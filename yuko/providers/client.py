"""OpenAI 兼容 + Anthropic 兼容模型客户端。

向 Engine 暴露统一的 `complete()` 接口：
- 输入：messages 列表 + tools 定义 + 配置
- 输出：模型文本（可能包含 tool_calls JSON）
"""

import json
import time
from typing import Any

import httpx

from .base import ModelResult, ProviderError

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_TIMEOUT = 120  # 秒


# ---- 工具函数 ----

def _normalize_base_url(url: str) -> str:
    """确保 base_url 以 /v1 结尾（OpenAI 兼容后端需要）。"""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _extract_text_from_openai(data: dict) -> str:
    """从 OpenAI chat/completions 响应中提取文本。"""
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
    return ""


def _extract_tool_calls_from_openai(data: dict) -> list[dict]:
    """从 OpenAI chat/completions 响应中提取 tool_calls。"""
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            return [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                }
                for tc in tool_calls
            ]
    return []


def _extract_text_from_anthropic(data: dict) -> str:
    """从 Anthropic /messages 响应中提取文本。"""
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _extract_tool_use_from_anthropic(data: dict) -> list[dict]:
    """从 Anthropic /messages 响应中提取 tool_use 块。"""
    tool_uses = []
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_uses.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {})),
            })
    return tool_uses


def _retry_delay(attempt: int) -> float:
    """指数退避延迟。"""
    return min(0.5 * (2 ** attempt), 8.0)


# ---- OpenAI 兼容客户端 ----

class OpenAICompatibleModelClient:
    """OpenAI chat/completions 兼容客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.last_metadata: dict = {}

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResult:
        """发送请求到 /chat/completions，返回模型结果。"""
        self.last_metadata = {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                }
                for t in tools
            ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(3):  # 最多 3 次尝试
            try:
                response = httpx.post(
                    self.base_url + "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                data = response.json()

                if response.is_success:
                    usage = data.get("usage", {})
                    self.last_metadata = {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "attempts": attempt + 1,
                        "retry_count": retry_count,
                    }
                    text = _extract_text_from_openai(data)
                    tool_calls = _extract_tool_calls_from_openai(data)
                    return ModelResult(
                        text=text,
                        tool_calls=tool_calls,
                        metadata=dict(self.last_metadata),
                    )

                # HTTP 错误
                retryable = response.status_code in RETRYABLE_HTTP_STATUS
                if retryable and attempt < 2:
                    retry_count += 1
                    time.sleep(_retry_delay(attempt))
                    continue

                raise ProviderError(
                    f"OpenAI 请求失败 (HTTP {response.status_code})",
                    provider="openai",
                    model=self.model,
                    base_url=self.base_url,
                    http_status=response.status_code,
                    retryable=retryable,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < 2:
                    retry_count += 1
                    time.sleep(_retry_delay(attempt))
                    continue
                raise ProviderError(
                    f"OpenAI 网络错误: {e}",
                    provider="openai",
                    model=self.model,
                    base_url=self.base_url,
                    retryable=True,
                ) from e

            except ProviderError:
                raise

        raise ProviderError(
            f"OpenAI 请求失败: {last_error}",
            provider="openai",
            model=self.model,
            base_url=self.base_url,
            retryable=True,
        )


# ---- Anthropic 兼容客户端 ----

class AnthropicCompatibleModelClient:
    """Anthropic /messages 兼容客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.last_metadata: dict = {}

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResult:
        """发送请求到 /messages，返回模型结果。"""
        self.last_metadata = {}

        # Anthropic 格式要求 messages 中的 content 是字符串数组或内容块
        anthropic_messages = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            anthropic_messages.append({
                "role": m["role"],
                "content": content,
            })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = httpx.post(
                    self.base_url + "/messages",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                data = response.json()

                if response.is_success:
                    usage = data.get("usage", {})
                    self.last_metadata = {
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "attempts": attempt + 1,
                        "retry_count": retry_count,
                    }
                    text = _extract_text_from_anthropic(data)
                    tool_calls = _extract_tool_use_from_anthropic(data)
                    return ModelResult(
                        text=text,
                        tool_calls=tool_calls,
                        metadata=dict(self.last_metadata),
                    )

                retryable = response.status_code in RETRYABLE_HTTP_STATUS
                if retryable and attempt < 2:
                    retry_count += 1
                    time.sleep(_retry_delay(attempt))
                    continue

                raise ProviderError(
                    f"Anthropic 请求失败 (HTTP {response.status_code})",
                    provider="anthropic",
                    model=self.model,
                    base_url=self.base_url,
                    http_status=response.status_code,
                    retryable=retryable,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < 2:
                    retry_count += 1
                    time.sleep(_retry_delay(attempt))
                    continue
                raise ProviderError(
                    f"Anthropic 网络错误: {e}",
                    provider="anthropic",
                    model=self.model,
                    base_url=self.base_url,
                    retryable=True,
                ) from e

            except ProviderError:
                raise

        raise ProviderError(
            f"Anthropic 请求失败: {last_error}",
            provider="anthropic",
            model=self.model,
            base_url=self.base_url,
            retryable=True,
        )

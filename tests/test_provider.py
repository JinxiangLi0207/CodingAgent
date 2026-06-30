"""Provider 层单元测试。"""

from unittest.mock import Mock, patch

import httpx
import pytest

from yuko.providers.base import ModelResult, ProviderError
from yuko.providers.client import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
    _extract_text_from_anthropic,
    _extract_text_from_openai,
    _extract_tool_calls_from_openai,
    _extract_tool_use_from_anthropic,
    _normalize_base_url,
)


class TestNormalizeBaseUrl:
    def test_already_v1(self):
        assert _normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_missing_v1(self):
        assert _normalize_base_url("https://api.openai.com") == "https://api.openai.com/v1"

    def test_trailing_slash(self):
        assert _normalize_base_url("https://api.openai.com/") == "https://api.openai.com/v1"


class TestExtractTextOpenAI:
    def test_normal_response(self):
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello, world!"}
                }
            ]
        }
        assert _extract_text_from_openai(data) == "Hello, world!"

    def test_empty_response(self):
        assert _extract_text_from_openai({}) == ""

    def test_no_content(self):
        data = {"choices": [{"message": {"role": "assistant"}}]}
        assert _extract_text_from_openai(data) == ""


class TestExtractToolCallsOpenAI:
    def test_with_tool_calls(self):
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/test.txt"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        result = _extract_tool_calls_from_openai(data)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["id"] == "call_123"

    def test_no_tool_calls(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        assert _extract_tool_calls_from_openai(data) == []


class TestExtractTextAnthropic:
    def test_normal_response(self):
        data = {
            "content": [
                {"type": "text", "text": "Hello from Claude"},
                {"type": "tool_use", "name": "read", "input": {}},
            ]
        }
        assert _extract_text_from_anthropic(data) == "Hello from Claude"

    def test_empty(self):
        assert _extract_text_from_anthropic({}) == ""


class TestExtractToolUseAnthropic:
    def test_with_tool_use(self):
        data = {
            "content": [
                {"type": "text", "text": "Let me read that"},
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"},
                },
            ]
        }
        result = _extract_tool_use_from_anthropic(data)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["id"] == "toolu_001"


class TestOpenAIClient:
    @pytest.fixture
    def client(self):
        return OpenAICompatibleModelClient(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )

    def test_complete_success(self, client):
        mock_response = Mock(spec=httpx.Response)
        mock_response.is_success = True
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch("httpx.post", return_value=mock_response):
            result = client.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert isinstance(result, ModelResult)
        assert result.text == "Hi!"
        assert result.metadata["input_tokens"] == 10

    def test_complete_with_tools(self, client):
        mock_response = Mock(spec=httpx.Response)
        mock_response.is_success = True
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/x"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }

        with patch("httpx.post", return_value=mock_response):
            result = client.complete(
                messages=[{"role": "user", "content": "Read /x"}],
                tools=[{"name": "read_file", "description": "Read a file", "parameters": {}}],
            )

        assert result.text == ""  # 工具调用没有文本内容

    def test_http_error(self, client):
        mock_response = Mock(spec=httpx.Response)
        mock_response.is_success = False
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ProviderError) as exc:
                client.complete(messages=[{"role": "user", "content": "x"}])
            assert exc.value.http_status == 401

    def test_retry_on_429(self, client):
        """429 应触发重试。"""
        mock_fail = Mock(spec=httpx.Response)
        mock_fail.is_success = False
        mock_fail.status_code = 429
        mock_fail.json.return_value = {"error": "Rate limited"}

        mock_ok = Mock(spec=httpx.Response)
        mock_ok.is_success = True
        mock_ok.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Recovered"}}],
            "usage": {},
        }

        with patch("httpx.post", side_effect=[mock_fail, mock_ok]):
            result = client.complete(messages=[{"role": "user", "content": "x"}])

        assert result.text == "Recovered"


class TestAnthropicClient:
    @pytest.fixture
    def client(self):
        return AnthropicCompatibleModelClient(
            model="claude-sonnet-4-6",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
        )

    def test_complete_success(self, client):
        mock_response = Mock(spec=httpx.Response)
        mock_response.is_success = True
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch("httpx.post", return_value=mock_response):
            result = client.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert result.text == "Hello from Claude"

    def test_http_error(self, client):
        mock_response = Mock(spec=httpx.Response)
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Server error"}

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ProviderError):
                client.complete(messages=[{"role": "user", "content": "x"}])


class TestProviderError:
    def test_basic(self):
        err = ProviderError(
            "Something went wrong",
            provider="openai",
            model="gpt-4o",
            base_url="https://api.openai.com",
            http_status=500,
            retryable=True,
        )
        assert str(err) == "Something went wrong"
        assert err.provider == "openai"
        assert err.http_status == 500
        assert err.retryable is True

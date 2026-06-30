"""引擎控制循环 —— yuko 的核心。

Engine 负责将用户请求转化为模型调用 → 工具执行 → 继续对话的循环，
直到获得最终答案或达到步数上限。
"""

import json
from pathlib import Path

from .providers.client import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)
from .tools.registry import TOOL_DEFINITIONS, build_tool_registry

# ---- System Prompt ----

SYSTEM_PROMPT = """你是一个 AI 编程助手 yuko。你可以使用工具来读取、搜索、修改代码和执行 Shell 命令。

可用工具：
{formatted_tools}

你需要直接给出回答或者调用工具。如果要调用工具，在你的回复中使用 tool_calls。
"""


def _format_tools_as_text() -> str:
    """将工具定义格式化为文本说明。"""
    lines = []
    for name, defn in TOOL_DEFINITIONS.items():
        desc = defn["description"]
        params = defn["parameters"]["properties"]
        required = defn["parameters"].get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = " (必填)" if pname in required else ""
            param_strs.append(
                f"    {pname}: {pinfo.get('type', 'str')}{req}"
                f" — {pinfo.get('description', '')}"
            )
        lines.append(f"- {name}: {desc}\n" + "\n".join(param_strs))
    return "\n".join(lines)


# ---- Engine ----

class Engine:
    """控制循环引擎。

    输入 → 调模型 → 解析工具调用 → 执行 → 循环 → 最终答案。
    """

    def __init__(
        self,
        model_client: OpenAICompatibleModelClient | AnthropicCompatibleModelClient,
        cwd: str | Path = ".",
        max_steps: int = 10,
    ):
        self.client = model_client
        self.cwd = Path(cwd).resolve()
        self.max_steps = max_steps
        self.tools = build_tool_registry(self.cwd)
        self._is_openai = isinstance(self.client, OpenAICompatibleModelClient)

    def ask(self, user_message: str) -> str:
        """同步入口：返回最终答案字符串。"""
        final = ""
        for event in self.run(user_message):
            if event["type"] == "final_answer":
                final = event["content"]
            elif event["type"] == "error":
                final = f"错误: {event['content']}"
        return final

    # ---- 事件生成器 ----

    def run(self, user_message: str):
        """生成器：逐步 yield 事件，供 CLI 渐进式渲染。

        yield 的事件：
        - {"type": "thinking", "step": N} — 开始第 N 步思考
        - {"type": "tool_call", "name": ..., "arguments": {...}}
        - {"type": "tool_result", "name": ..., "content": "..."}
        - {"type": "final_answer", "content": "...", "metadata": {...}}
        - {"type": "error", "content": "..."}
        """
        messages: list[dict] = self._build_initial_messages(user_message)
        tool_defs = self._build_tool_defs()

        for step in range(self.max_steps):
            yield {"type": "thinking", "step": step + 1}

            # 1. 调用模型
            try:
                result = self.client.complete(
                    messages=messages,
                    tools=tool_defs,
                )
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            # 2. 检查工具调用
            if not result.tool_calls:
                yield {
                    "type": "final_answer",
                    "content": result.text,
                    "metadata": result.metadata,
                }
                return

            # 3. 执行工具调用
            for tc in result.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                yield {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_args,
                }

                # 执行
                tool = self.tools.get(tool_name)
                if tool is None:
                    tool_result = f"错误: 未知工具 '{tool_name}'"
                else:
                    try:
                        tr = tool.execute(tool_args)
                        tool_result = tr.content
                    except ValueError as e:
                        tool_result = f"错误: {e}"

                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "content": tool_result,
                }

                # 追加到消息历史
                if self._is_openai:
                    messages.append({
                        "role": "assistant",
                        "content": result.text or None,
                        "tool_calls": [{
                            "id": tc.get("id", f"call_{step}"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step}"),
                        "content": tool_result,
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tc.get("id", f"call_{step}"),
                                "name": tool_name,
                                "input": tool_args,
                            },
                        ],
                    })
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.get("id", f"call_{step}"),
                                "content": tool_result,
                            },
                        ],
                    })

        # 达到 max_steps
        yield {
            "type": "final_answer",
            "content": f"已达到最大步数 ({self.max_steps})。",
        }

    # ---- 内部方法 ----

    def _build_initial_messages(self, user_message: str) -> list[dict]:
        """构建初始消息列表。"""
        system_content = SYSTEM_PROMPT.format(
            formatted_tools=_format_tools_as_text()
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    def _build_tool_defs(self) -> list[dict]:
        """构建与 provider 协议匹配的工具定义列表。"""
        if self._is_openai:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": defn["description"],
                        "parameters": defn["parameters"],
                    },
                }
                for name, defn in TOOL_DEFINITIONS.items()
            ]
        else:
            return [
                {
                    "name": name,
                    "description": defn["description"],
                    "input_schema": defn["parameters"],
                }
                for name, defn in TOOL_DEFINITIONS.items()
            ]

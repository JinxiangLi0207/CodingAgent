"""工具抽象 —— RegisteredTool 和 ToolResult。"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果。"""
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class RegisteredTool:
    """注册的工具定义。"""

    name: str
    description: str
    parameters: dict  # JSON Schema 格式的参数定义
    risky: bool  # True = 写操作，需要审批
    runner: Callable[..., ToolResult]  # 执行函数

    @property
    def read_only(self) -> bool:
        return not self.risky

    def execute(self, args: dict[str, Any]) -> ToolResult:
        """执行工具，确保返回 ToolResult。"""
        result = self.runner(args)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=str(result))

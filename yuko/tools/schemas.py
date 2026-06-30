"""工具的 Pydantic 参数模型 —— 每个工具的参数单源真理。"""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class ReadFileArgs(BaseModel):
    """读取文件的参数。"""
    path: str
    start: int = 1
    end: int = 200

    @field_validator("start")
    @classmethod
    def start_ge_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("start 必须 >= 1")
        return v

    @model_validator(mode="after")
    def end_ge_start(self) -> "ReadFileArgs":
        if self.end < self.start:
            raise ValueError("无效的行范围")
        return self


class WriteFileArgs(BaseModel):
    """写入文件的参数。"""
    path: str
    content: str


class ListFilesArgs(BaseModel):
    """列出目录的参数。"""
    path: str = "."


class SearchArgs(BaseModel):
    """搜索文件内容的参数。"""
    pattern: str
    path: str = "."

    @field_validator("pattern")
    @classmethod
    def pattern_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pattern 不能为空")
        return v


class RunShellArgs(BaseModel):
    """执行 Shell 命令的参数。"""
    command: str
    timeout: int = 20

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command 不能为空")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_in_range(cls, v: int) -> int:
        if v < 1 or v > 120:
            raise ValueError("timeout 必须在 [1, 120] 之间")
        return v


# 工具名 → Pydantic 模型 的映射
TOOL_SCHEMAS = {
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "list_files": ListFilesArgs,
    "search": SearchArgs,
    "run_shell": RunShellArgs,
}

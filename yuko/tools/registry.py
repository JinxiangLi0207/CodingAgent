"""工具注册、校验与执行。

这是 agent 的能力白名单：模型能申请哪些动作、如何校验参数、如何执行。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .base import RegisteredTool, ToolResult
from .schemas import TOOL_SCHEMAS

# 被忽略的目录/文件名
IGNORED_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                  ".yuko", "node_modules", ".DS_Store"}

# ---- 工具参数定义（JSON Schema 格式）----

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "read_file": {
        "description": "读取文件内容，支持行范围。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "start": {"type": "integer", "description": "起始行号（从1开始）", "default": 1},
                "end": {"type": "integer", "description": "结束行号", "default": 200},
            },
            "required": ["path"],
        },
        "risky": False,
    },
    "write_file": {
        "description": "写入/创建文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
        "risky": True,
    },
    "list_files": {
        "description": "列出目录内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径", "default": "."},
            },
        },
        "risky": False,
    },
    "search": {
        "description": "搜索文件内容（优先使用 ripgrep）。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索模式"},
                "path": {"type": "string", "description": "搜索路径", "default": "."},
            },
            "required": ["pattern"],
        },
        "risky": False,
    },
    "run_shell": {
        "description": "执行 Shell 命令。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout": {"type": "integer", "description": "超时秒数 [1, 120]", "default": 20},
            },
            "required": ["command"],
        },
        "risky": True,
    },
}


# ---- 工具注册表构建 ----

def build_tool_registry(cwd: str | Path) -> dict[str, RegisteredTool]:
    """构建工具注册表。

    工具是显式注册的，不是动态发现的。这样模型看到的是一个有边界、
    可审计的动作集合。
    """
    root = Path(cwd).resolve()
    return {
        name: RegisteredTool(
            name=name,
            description=defn["description"],
            parameters=defn["parameters"],
            risky=defn["risky"],
            runner=_make_runner(name, root),
        )
        for name, defn in TOOL_DEFINITIONS.items()
    }


def _make_runner(name: str, root: Path):
    """创建绑定了 cwd 的工具执行函数。"""
    runners = {
        "read_file": _tool_read_file,
        "write_file": _tool_write_file,
        "list_files": _tool_list_files,
        "search": _tool_search,
        "run_shell": _tool_run_shell,
    }

    def runner(args: dict) -> ToolResult:
        return runners[name](root, args)

    return runner


# ---- 参数校验 ----

def validate_tool_args(name: str, args: dict) -> None:
    """校验工具参数（Pydantic + 工作区检查）。"""
    schema_cls = TOOL_SCHEMAS.get(name)
    if schema_cls is not None:
        try:
            schema_cls.model_validate(args)
        except ValidationError as exc:
            raise ValueError(_first_error(exc)) from exc


def _first_error(exc: ValidationError) -> str:
    """从 Pydantic ValidationError 中提取第一条错误信息。"""
    errors = exc.errors(include_url=False)
    if not errors:
        return str(exc)
    err = errors[0]
    msg = str(err.get("msg", "")).removeprefix("Value error, ")
    if err.get("type") == "missing":
        loc = err.get("loc", ())
        field = loc[-1] if loc else ""
        if field:
            return f"缺少参数: '{field}'"
    return msg


# ---- 工具实现 ----

def _resolve(root: Path, rel: str) -> Path:
    """解析相对路径为绝对路径，防止路径穿越。"""
    path = (root / rel).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError(f"不允许访问工作区外的路径: {rel}")
    return path


def _visible_entries(path: Path) -> list[Path]:
    """列出可见的目录条目（过滤忽略的文件）。"""
    items = []
    try:
        for item in path.iterdir():
            if item.name not in IGNORED_NAMES and not item.name.startswith("."):
                items.append(item)
    except PermissionError:
        pass
    return sorted(items, key=lambda x: (x.is_file(), x.name.lower()))


def _tool_list_files(root: Path, args: dict) -> ToolResult:
    path = _resolve(root, args.get("path", "."))
    if not path.is_dir():
        return ToolResult(content=f"错误: '{args['path']}' 不是目录", is_error=True)

    entries = _visible_entries(path)
    if not entries:
        return ToolResult(content="(空)")

    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(root)}")
        if entry.is_dir():
            for child in _visible_entries(entry)[:12]:
                child_kind = "[D]" if child.is_dir() else "[F]"
                lines.append(f"    {child_kind} {child.relative_to(root)}")
    return ToolResult(content="\n".join(lines))


def _tool_read_file(root: Path, args: dict) -> ToolResult:
    path = _resolve(root, args["path"])
    if not path.is_file():
        return ToolResult(content=f"错误: '{args['path']}' 不是文件", is_error=True)

    start = int(args.get("start", 1))
    end = int(args.get("end", 200))

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return ToolResult(content=f"错误: 无法读取文件: {e}", is_error=True)

    body = "\n".join(
        f"{num:>4}: {line}"
        for num, line in enumerate(lines[start - 1 : end], start=start)
    )
    return ToolResult(content=f"# {path.relative_to(root)}\n{body}")


def _tool_write_file(root: Path, args: dict) -> ToolResult:
    path = _resolve(root, args["path"])
    content = str(args["content"])

    if path.exists() and path.is_dir():
        return ToolResult(content=f"错误: '{args['path']}' 是目录", is_error=True)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult(content=f"错误: 写入失败: {e}", is_error=True)

    return ToolResult(content=f"已写入 {path.relative_to(root)} ({len(content)} 字符)")


def _tool_search(root: Path, args: dict) -> ToolResult:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return ToolResult(content="错误: pattern 不能为空", is_error=True)

    path = _resolve(root, args.get("path", "."))
    max_matches = 200

    # 优先使用 ripgrep
    rg = shutil.which("rg")
    if rg:
        result = subprocess.run(
            [rg, "-n", "--smart-case", "--max-count", str(max_matches), pattern, str(path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return ToolResult(content=output or "(无匹配)")

    # 简单回退搜索
    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file()
        and not any(p in IGNORED_NAMES or p.startswith(".") for p in item.relative_to(root).parts)
    ]
    for file_path in files:
        try:
            for num, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(root)}:{num}:{line}")
                    if len(matches) >= max_matches:
                        break
        except Exception:
            continue
        if len(matches) >= max_matches:
            break

    return ToolResult(content="\n".join(matches) or "(无匹配)")


def _tool_run_shell(root: Path, args: dict) -> ToolResult:
    command = str(args.get("command", "")).strip()
    if not command:
        return ToolResult(content="错误: command 不能为空", is_error=True)

    timeout = int(args.get("timeout", 20))
    timeout = max(1, min(timeout, 120))

    try:
        result = subprocess.run(
            command,
            cwd=str(root),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(content=f"错误: 命令超时 ({timeout}s)", is_error=True)
    except Exception as e:
        return ToolResult(content=f"错误: 执行失败: {e}", is_error=True)

    return ToolResult(content=textwrap.dedent(f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(空)"}
        stderr:
        {result.stderr.strip() or "(空)"}
        """).strip())

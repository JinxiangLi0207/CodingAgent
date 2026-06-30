"""工具系统单元测试。"""

import tempfile
from pathlib import Path

import pytest

from yuko.tools.base import RegisteredTool, ToolResult
from yuko.tools.registry import TOOL_DEFINITIONS, build_tool_registry, validate_tool_args
from yuko.tools.schemas import (
    ListFilesArgs,
    ReadFileArgs,
    RunShellArgs,
    SearchArgs,
    WriteFileArgs,
)


class TestSchemas:
    def test_read_file_valid(self):
        args = ReadFileArgs(path="test.py")
        assert args.path == "test.py"
        assert args.start == 1
        assert args.end == 200

    def test_read_file_invalid_start(self):
        with pytest.raises(ValueError):
            ReadFileArgs(path="test.py", start=0)

    def test_read_file_end_lt_start(self):
        with pytest.raises(ValueError):
            ReadFileArgs(path="test.py", start=10, end=5)

    def test_search_empty_pattern(self):
        with pytest.raises(ValueError):
            SearchArgs(pattern="")

    def test_run_shell_empty_command(self):
        with pytest.raises(ValueError):
            RunShellArgs(command="")

    def test_run_shell_timeout_range(self):
        with pytest.raises(ValueError):
            RunShellArgs(command="ls", timeout=0)


class TestToolDefinitions:
    def test_all_core_tools_registered(self):
        """确保5个核心工具都已定义。"""
        assert set(TOOL_DEFINITIONS.keys()) == {
            "read_file", "write_file", "list_files", "search", "run_shell"
        }

    def test_each_tool_has_description(self):
        for name, defn in TOOL_DEFINITIONS.items():
            assert defn["description"], f"{name} 缺少 description"

    def test_each_tool_has_parameters(self):
        for name, defn in TOOL_DEFINITIONS.items():
            assert "parameters" in defn, f"{name} 缺少 parameters"


class TestBuildToolRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return build_tool_registry(tmp_path)

    def test_returns_dict_of_registered_tools(self, registry):
        assert len(registry) == 5
        for name, tool in registry.items():
            assert isinstance(tool, RegisteredTool)
            assert tool.name == name

    def test_tools_have_runner(self, registry):
        for tool in registry.values():
            assert callable(tool.runner)

    def test_risky_flags(self, registry):
        assert not registry["read_file"].risky
        assert not registry["list_files"].risky
        assert not registry["search"].risky
        assert registry["write_file"].risky
        assert registry["run_shell"].risky


class TestToolExecution:
    @pytest.fixture
    def registry(self, tmp_path):
        return build_tool_registry(tmp_path)

    def test_list_files_empty(self, registry):
        result = registry["list_files"].execute({"path": "."})
        assert not result.is_error

    def test_list_files_with_content(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hi')")
        (tmp_path / "data").mkdir()
        registry = build_tool_registry(tmp_path)
        result = registry["list_files"].execute({"path": "."})
        assert "hello.py" in result.content
        assert "data" in result.content

    def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        registry = build_tool_registry(tmp_path)
        result = registry["read_file"].execute({"path": "test.txt", "start": 2, "end": 4})
        assert result.content
        assert "line2" in result.content
        assert "line4" in result.content

    def test_read_file_not_found(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        result = registry["read_file"].execute({"path": "nonexistent.txt"})
        assert result.is_error

    def test_write_file(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        result = registry["write_file"].execute({
            "path": "output.txt",
            "content": "hello world"
        })
        assert not result.is_error
        assert (tmp_path / "output.txt").read_text() == "hello world"

    def test_write_file_creates_dirs(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        result = registry["write_file"].execute({
            "path": "a/b/c/file.txt",
            "content": "nested"
        })
        assert not result.is_error
        assert (tmp_path / "a" / "b" / "c" / "file.txt").read_text() == "nested"

    def test_write_file_on_directory(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        registry = build_tool_registry(tmp_path)
        result = registry["write_file"].execute({
            "path": "mydir",
            "content": "x"
        })
        assert result.is_error

    def test_search_finds_match(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    return 'world'\n")
        registry = build_tool_registry(tmp_path)
        result = registry["search"].execute({"pattern": "hello", "path": "."})
        assert "code.py" in result.content

    def test_search_no_match(self, tmp_path):
        (tmp_path / "code.py").write_text("print('hi')")
        registry = build_tool_registry(tmp_path)
        result = registry["search"].execute({"pattern": "zzz_nonexistent_zzz", "path": "."})
        assert "(无匹配)" in result.content

    def test_run_shell_echo(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        result = registry["run_shell"].execute({"command": "echo hello"})
        assert not result.is_error
        assert "hello" in result.content

    def test_run_shell_timeout(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        result = registry["run_shell"].execute({
            "command": "sleep 10",
            "timeout": 1,
        })
        assert result.is_error
        assert "超时" in result.content

    def test_path_traversal_blocked(self, tmp_path):
        registry = build_tool_registry(tmp_path)
        with pytest.raises(ValueError, match="工作区外"):
            registry["read_file"].execute({"path": "C:/Windows/System32/drivers/etc/hosts"})


class TestValidateToolArgs:
    def test_valid_args(self):
        validate_tool_args("read_file", {"path": "test.py"})

    def test_missing_required(self):
        with pytest.raises(ValueError):
            validate_tool_args("read_file", {})

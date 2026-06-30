"""CLI 单元测试。"""

import pytest

from yuko.cli import build_arg_parser


class TestArgParser:
    def test_help(self):
        p = build_arg_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--help"])

    def test_message(self):
        p = build_arg_parser()
        args = p.parse_args(["hello", "world"])
        assert args.message == ["hello", "world"]

    def test_multi_word_message(self):
        p = build_arg_parser()
        args = p.parse_args(["帮我", "读取", "README.md"])
        assert " ".join(args.message) == "帮我 读取 README.md"

    def test_provider_short(self):
        p = build_arg_parser()
        args = p.parse_args(["-p", "anthropic", "hi"])
        assert args.provider == "anthropic"

    def test_model_short(self):
        p = build_arg_parser()
        args = p.parse_args(["-m", "gpt-4o", "hi"])
        assert args.model == "gpt-4o"

    def test_api_key(self):
        p = build_arg_parser()
        args = p.parse_args(["-k", "sk-test", "hi"])
        assert args.api_key == "sk-test"

    def test_max_steps(self):
        p = build_arg_parser()
        args = p.parse_args(["--max-steps", "5", "hi"])
        assert args.max_steps == 5

    def test_repl_flag(self):
        p = build_arg_parser()
        args = p.parse_args(["--repl"])
        assert args.repl is True

    def test_cwd(self):
        p = build_arg_parser()
        args = p.parse_args(["-C", "/tmp", "hi"])
        assert args.cwd == "/tmp"

    def test_defaults(self):
        p = build_arg_parser()
        args = p.parse_args([])
        assert args.message == []
        assert args.provider is None
        assert args.repl is False
        assert args.max_steps == 10

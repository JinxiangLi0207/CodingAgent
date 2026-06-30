"""CLI 入口 —— 命令行解析、配置组装、模式分发。"""

import argparse
import os
import sys
from pathlib import Path

from .config import resolve_provider_config
from .engine import Engine
from .providers.client import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)

WELCOME = r"""
  __  ____  __ __  ____
  \ \/ /\ \/ //  |/  _ \
   \  /  \  // /| // // /
   / /   / // ___// // /_
  /_/   /_//_/  /____/(_)

  yuko — 轻量级 AI 编程助手
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    p = argparse.ArgumentParser(
        prog="yuko",
        description="yuko — 轻量级 AI 编程助手",
    )
    p.add_argument(
        "message",
        nargs="*",
        help="发送给 yuko 的消息（省略则进入 REPL 模式）",
    )
    p.add_argument(
        "--repl", action="store_true",
        help="强制进入 REPL 交互模式",
    )
    p.add_argument(
        "--provider", "-p", type=str, default=None,
        help="Provider 名称 (openai/anthropic/deepseek)",
    )
    p.add_argument(
        "--model", "-m", type=str, default=None,
        help="模型名称",
    )
    p.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="API Key",
    )
    p.add_argument(
        "--base-url", type=str, default=None,
        help="API Base URL",
    )
    p.add_argument(
        "--cwd", "-C", type=str, default=".",
        help="工作目录",
    )
    p.add_argument(
        "--max-steps", type=int, default=10,
        help="最大工具调用步数 (默认: 10)",
    )
    return p


def create_client(provider_config):
    """根据配置创建对应的模型客户端。"""
    if provider_config.protocol == "openai":
        return OpenAICompatibleModelClient(
            model=provider_config.model,
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
        )
    else:
        return AnthropicCompatibleModelClient(
            model=provider_config.model,
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
        )


def main(argv: list[str] | None = None) -> None:
    """yuko 主入口点。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 组装用户消息
    user_message = " ".join(args.message).strip() if args.message else ""

    # 解析配置
    try:
        provider_config = resolve_provider_config(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            start=args.cwd,
        )
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    if not provider_config.api_key:
        print(
            "错误: 未设置 API Key。请通过以下方式之一设置:\n"
            "  - 环境变量 YUKO_API_KEY\n"
            "  - 环境变量 OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY\n"
            "  - .yuko.toml 配置文件\n"
            "  - --api-key 命令行参数",
            file=sys.stderr,
        )
        sys.exit(1)

    # 创建客户端和引擎
    client = create_client(provider_config)
    engine = Engine(client, cwd=args.cwd, max_steps=args.max_steps)

    # 显示启动信息
    print(f"yuko v0.1.0  [{provider_config.name}] {provider_config.model}")
    print(f"工作目录: {Path(args.cwd).resolve()}")
    print()

    if user_message and not args.repl:
        # one-shot 模式
        _run_oneshot(engine, user_message)
    else:
        # REPL 模式
        _run_repl(engine, user_message)


def _run_oneshot(engine: Engine, message: str) -> None:
    """执行一次性消息。"""
    for event in engine.run(message):
        etype = event["type"]
        if etype == "thinking":
            pass  # 不显示进度
        elif etype == "tool_call":
            print(f"🔧 {event['name']}({event['arguments']})")
        elif etype == "tool_result":
            content = str(event.get("content", ""))
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"   → {content[:100]}")
        elif etype == "final_answer":
            print(f"\n{event['content']}")
        elif etype == "error":
            print(f"\n❌ 错误: {event['content']}", file=sys.stderr)


def _run_repl(engine: Engine, initial_message: str = "") -> None:
    """REPL 交互模式。"""
    print(WELCOME)
    print("输入消息与 yuko 对话，输入 /help 查看帮助，Ctrl+C 退出。\n")

    if initial_message:
        _run_oneshot(engine, initial_message)

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 斜杠命令
        if user_input.startswith("/"):
            _handle_slash(user_input, engine)
            continue

        _run_oneshot(engine, user_input)


def _handle_slash(command: str, engine: Engine) -> None:
    """处理斜杠命令。"""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/h"):
        print("可用命令:")
        print("  /help       — 显示帮助")
        print("  /info       — 显示当前配置信息")
        print("  /clear      — 清屏（重启引擎）")
        print("  /exit, /q   — 退出")
        print("直接输入消息即可与 yuko 对话。")
    elif cmd == "/info":
        print(f"工作目录: {engine.cwd}")
        print(f"最大步数: {engine.max_steps}")
        print(f"已注册工具: {', '.join(engine.tools.keys())}")
    elif cmd == "/clear":
        print("（引擎状态已重置）")
    elif cmd in ("/exit", "/q"):
        print("再见！")
        sys.exit(0)
    else:
        print(f"未知命令: {cmd}，输入 /help 查看帮助")


if __name__ == "__main__":
    main()

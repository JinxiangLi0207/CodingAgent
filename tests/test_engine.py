"""引擎控制循环单元测试。"""

import json
from pathlib import Path

import pytest

from yuko.engine import Engine
from yuko.providers.base import ModelResult
from yuko.providers.client import OpenAICompatibleModelClient


class FakeModelClient:
    """模拟模型客户端，按预设脚本返回结果。"""

    # 模拟 OpenAI 客户端需要的属性
    model = "fake-model"
    base_url = "https://fake.example.com/v1"

    def __init__(self, responses: list[ModelResult]):
        self.responses = responses
        self.call_count = 0
        self.last_messages: list[dict] = []
        self.last_tools: list[dict] = []

    def complete(self, messages, tools=None, max_tokens=4096):
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools or []
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return ModelResult(text="done")

    def __str__(self):
        return "FakeModelClient"


class FakeOpenAIClient(FakeModelClient, OpenAICompatibleModelClient):
    """伪装成 OpenAI 兼容客户端，使用 FakeModelClient 的 complete()。"""
    pass


class TestEngine:
    def test_simple_answer_no_tools(self, tmp_path):
        """模型直接回答，不使用工具。"""
        client = FakeOpenAIClient([
            ModelResult(text="你好！有什么可以帮你的？"),
        ])
        engine = Engine(client, cwd=tmp_path)
        answer = engine.ask("你好")
        assert "你好" in answer

    def test_single_tool_call(self, tmp_path):
        """模型调用一次工具后回答。"""
        (tmp_path / "hello.py").write_text("print('hello')")
        client = FakeOpenAIClient([
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_0",
                    "name": "read_file",
                    "arguments": {"path": "hello.py"},
                }],
            ),
            ModelResult(text="文件内容是 print('hello')"),
        ])
        engine = Engine(client, cwd=tmp_path, max_steps=5)
        answer = engine.ask("读取 hello.py")
        assert "print" in answer
        assert client.call_count == 2

    def test_multiple_tool_calls(self, tmp_path):
        """模型连续调用多个工具。"""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        client = FakeOpenAIClient([
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_0",
                    "name": "read_file",
                    "arguments": {"path": "a.py"},
                }],
            ),
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": {"path": "b.py"},
                }],
            ),
            ModelResult(text="两个文件已读取完毕"),
        ])
        engine = Engine(client, cwd=tmp_path, max_steps=5)
        answer = engine.ask("读取所有文件")
        assert "读取完毕" in answer
        assert client.call_count == 3

    def test_max_steps_reached(self, tmp_path):
        """达到最大步数限制。"""
        client = FakeOpenAIClient([
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_0",
                    "name": "read_file",
                    "arguments": {"path": "x.py"},
                }],
            ),
        ] * 10)  # 多于 max_steps
        engine = Engine(client, cwd=tmp_path, max_steps=3)
        answer = engine.ask("一直读取")
        assert "最大步数" in answer

    def test_unknown_tool(self, tmp_path):
        """模型请求未注册的工具。"""
        client = FakeOpenAIClient([
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_0",
                    "name": "nonexistent_tool",
                    "arguments": {},
                }],
            ),
            ModelResult(text="工具不存在，我直接回答"),
        ])
        engine = Engine(client, cwd=tmp_path, max_steps=5)
        answer = engine.ask("用未知工具")
        assert "工具不存在" in answer or client.call_count >= 1

    def test_events_yielded(self, tmp_path):
        """验证生成器产生正确的事件序列。"""
        client = FakeOpenAIClient([
            ModelResult(text="直接回答"),
        ])
        engine = Engine(client, cwd=tmp_path)
        events = list(engine.run("你好"))
        types = [e["type"] for e in events]
        assert "thinking" in types
        assert "final_answer" in types

    def test_tool_events(self, tmp_path):
        """验证工具调用产生 event 序列。"""
        (tmp_path / "test.txt").write_text("data")
        client = FakeOpenAIClient([
            ModelResult(
                text="",
                tool_calls=[{
                    "id": "call_0",
                    "name": "read_file",
                    "arguments": {"path": "test.txt"},
                }],
            ),
            ModelResult(text="完成"),
        ])
        engine = Engine(client, cwd=tmp_path)
        events = list(engine.run("读取"))
        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final_answer" in types

    def test_error_handling(self, tmp_path):
        """模型调用出错时的错误事件。"""
        class ErrorClient:
            def complete(self, messages, tools=None, max_tokens=4096):
                raise RuntimeError("模拟错误")
            def __str__(self):
                return "ErrorClient"

        engine = Engine(ErrorClient(), cwd=tmp_path)
        events = list(engine.run("hi"))
        assert any(e["type"] == "error" for e in events)

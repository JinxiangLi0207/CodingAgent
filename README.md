# yuko

一个轻量级 AI Coding Agent，从零搭建，学习 [pico](https://gitee.com/htxoffical/pico) 架构设计。

## 安装

```bash
pip install -e .
```

## 使用

```bash
yuko "你的消息"    # one-shot 模式
python -m yuko     # 入口点
```

## 开发

```bash
pip install -e ".[dev]"
pytest tests/
ruff check yuko/
```

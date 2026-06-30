"""Provider 层 —— 模型后端适配。"""

from .base import ModelResult, ProviderError
from .client import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
    _extract_text_from_openai,
    _extract_tool_calls_from_openai,
    _extract_text_from_anthropic,
    _extract_tool_use_from_anthropic,
)

__all__ = [
    "AnthropicCompatibleModelClient",
    "ModelResult",
    "OpenAICompatibleModelClient",
    "ProviderError",
]

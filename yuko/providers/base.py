"""Provider 基础类型 —— ModelResult 和统一的调用接口。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelResult:
    """模型返回结果。"""
    text: str
    metadata: dict = field(default_factory=dict)


class ProviderError(Exception):
    """Provider 错误，包含调试所需的上下文。"""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        http_status: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.http_status = http_status
        self.retryable = retryable

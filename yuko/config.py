"""项目配置解析 —— Provider 配置、.yuko.toml、环境变量。"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# ---- 常量 ----

DEFAULT_PROVIDER = "openai"
PROJECT_CONFIG_NAME = ".yuko.toml"
PROTOCOLS = {"openai", "anthropic"}

# Provider 默认值
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "anthropic": {
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-6",
    },
    "deepseek": {
        "protocol": "anthropic",
        "base_url": "https://api.deepseek.com/anthropic",
        "model": "deepseek-v4-pro",
    },
}

PROVIDER_ALIASES = {
    "gpt": "openai",
    "claude": "anthropic",
}

# Provider 各自的环境变量名
PROVIDER_ENV_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "openai": {
        "api_key": ("OPENAI_API_KEY",),
        "base_url": ("OPENAI_BASE_URL",),
        "model": ("OPENAI_MODEL",),
    },
    "anthropic": {
        "api_key": ("ANTHROPIC_API_KEY",),
        "base_url": ("ANTHROPIC_BASE_URL",),
        "model": ("ANTHROPIC_MODEL",),
    },
    "deepseek": {
        "api_key": ("DEEPSEEK_API_KEY",),
        "base_url": ("DEEPSEEK_BASE_URL",),
        "model": ("DEEPSEEK_MODEL",),
    },
}

# 通用环境变量（优先级最高）
ENV_PROVIDER = "YUKO_PROVIDER"
ENV_API_KEY = "YUKO_API_KEY"
ENV_BASE_URL = "YUKO_BASE_URL"
ENV_MODEL = "YUKO_MODEL"


# ---- 数据类 ----

@dataclass(frozen=True)
class ProviderConfig:
    """Provider 配置，包含连接 LLM 所需的所有信息。"""

    name: str
    protocol: str
    api_key: str
    base_url: str
    model: str


# ---- 工具函数 ----

def normalize_provider_name(name: str | None) -> str:
    """标准化 provider 名称，支持别名映射。"""
    if not name:
        return DEFAULT_PROVIDER
    normalized = name.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def _first_value(*values: Any) -> str:
    """返回第一个非空值。"""
    for v in values:
        if v:
            return str(v)
    return ""


def _find_project_config(start: str | Path = ".") -> Path | None:
    """向上查找项目级 .yuko.toml。"""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        config_path = path / PROJECT_CONFIG_NAME
        if config_path.exists():
            return config_path
    return None


def _read_toml(path: Path) -> dict[str, Any]:
    """读取 TOML 配置文件。"""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise ValueError(f"无法读取配置文件 {path}: {e}") from e


def _env_values_for_provider(provider_name: str) -> dict[str, str]:
    """从 Provider 专属环境变量读取配置。"""
    values: dict[str, str] = {}
    for key, names in PROVIDER_ENV_NAMES.get(provider_name, {}).items():
        for name in names:
            val = os.environ.get(name)
            if val:
                values[key] = val
                break
    return values


def resolve_provider_config(
    provider: str | None = None,
    *,
    start: str | Path = ".",
    config_path: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    """解析 Provider 配置。

    优先级：CLI 参数 > 通用环境变量 > Provider 环境变量 > 项目 TOML > 默认值
    """
    # 1. 读取 TOML 配置文件
    toml_data: dict[str, Any] = {}
    if config_path:
        toml_data = _read_toml(Path(config_path))
    else:
        found = _find_project_config(start)
        if found:
            toml_data = _read_toml(found)

    # 2. 确定 provider 名称（CLI > 通用环境变量 > TOML 顶层 provider > 默认值）
    toml_top_provider = toml_data.get("provider", "")
    provider_name = normalize_provider_name(
        provider
        or os.environ.get(ENV_PROVIDER)
        or toml_top_provider
    )

    # 提取 TOML 中当前 provider 的配置段
    toml_provider: dict[str, Any] = {}
    if "providers" in toml_data and provider_name in toml_data["providers"]:
        toml_provider = toml_data["providers"][provider_name]

    # 3. 默认值
    defaults = PROVIDER_DEFAULTS.get(provider_name, {})

    # 4. Provider 环境变量
    env_vals = _env_values_for_provider(provider_name)

    # 5. 按优先级合并
    resolved_model = _first_value(
        model,
        os.environ.get(ENV_MODEL),
        env_vals.get("model"),
        toml_provider.get("model"),
        defaults.get("model"),
    )

    resolved_base_url = _first_value(
        base_url,
        os.environ.get(ENV_BASE_URL),
        env_vals.get("base_url"),
        toml_provider.get("base_url"),
        defaults.get("base_url"),
    )

    resolved_api_key = _first_value(
        api_key,
        os.environ.get(ENV_API_KEY),
        env_vals.get("api_key"),
        toml_provider.get("api_key"),
        "",
    )

    protocol = _first_value(
        toml_provider.get("protocol"),
        defaults.get("protocol"),
    )
    if protocol not in PROTOCOLS:
        raise ValueError(f"Provider '{provider_name}' 使用了不支持的协议: {protocol}")

    return ProviderConfig(
        name=provider_name,
        protocol=protocol,
        api_key=str(resolved_api_key),
        base_url=str(resolved_base_url),
        model=str(resolved_model),
    )

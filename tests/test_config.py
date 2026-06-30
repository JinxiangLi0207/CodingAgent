"""配置系统单元测试。"""

import os
import tempfile
from pathlib import Path

import pytest

from yuko.config import (
    DEFAULT_PROVIDER,
    ProviderConfig,
    normalize_provider_name,
    resolve_provider_config,
)


class TestNormalizeProviderName:
    def test_default(self):
        assert normalize_provider_name(None) == DEFAULT_PROVIDER

    def test_alias_gpt(self):
        assert normalize_provider_name("gpt") == "openai"

    def test_alias_claude(self):
        assert normalize_provider_name("claude") == "anthropic"

    def test_passthrough(self):
        assert normalize_provider_name("deepseek") == "deepseek"

    def test_case_insensitive(self):
        assert normalize_provider_name("OpenAI") == "openai"


class TestResolveProviderConfig:
    def test_defaults_only(self):
        """纯默认值，无 TOML、无环境变量。"""
        config = resolve_provider_config(start="/tmp/nonexistent")
        assert config.name == "openai"
        assert config.protocol == "openai"
        assert config.model == "gpt-4o"
        assert "api.openai.com" in config.base_url

    def test_cli_override(self):
        """CLI 参数覆盖一切。"""
        config = resolve_provider_config(
            provider="anthropic",
            model="claude-opus-4-8",
            base_url="https://custom.example.com",
            api_key="sk-cli-key",
            start="/tmp/nonexistent",
        )
        assert config.name == "anthropic"
        assert config.protocol == "anthropic"
        assert config.model == "claude-opus-4-8"
        assert config.base_url == "https://custom.example.com"
        assert config.api_key == "sk-cli-key"

    def test_toml_config(self):
        """从 TOML 文件读取配置。"""
        toml_content = """
provider = "deepseek"

[providers.deepseek]
protocol = "anthropic"
api_key = "sk-toml-key"
base_url = "https://toml.example.com"
model = "toml-model"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            config = resolve_provider_config(config_path=toml_path)
            assert config.name == "deepseek"
            assert config.protocol == "anthropic"
            assert config.api_key == "sk-toml-key"
            assert config.base_url == "https://toml.example.com"
            assert config.model == "toml-model"
        finally:
            os.unlink(toml_path)

    def test_env_override_toml(self):
        """通用环境变量覆盖 TOML。"""
        toml_content = """
[providers.openai]
api_key = "sk-toml-key"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            os.environ["YUKO_API_KEY"] = "sk-env-key"
            config = resolve_provider_config(config_path=toml_path)
            assert config.api_key == "sk-env-key"
        finally:
            os.unlink(toml_path)
            os.environ.pop("YUKO_API_KEY", None)

    def test_provider_env_vars(self):
        """Provider 专属环境变量。"""
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        try:
            config = resolve_provider_config(start="/tmp/nonexistent")
            assert config.api_key == "sk-openai-env"
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_unsupported_protocol(self):
        """不支持的协议应抛出错误。"""
        toml_content = """
[providers.openai]
protocol = "grpc"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            with pytest.raises(ValueError, match="协议"):
                resolve_provider_config(config_path=toml_path)
        finally:
            os.unlink(toml_path)

    def test_env_provider_selection(self):
        """环境变量 YUKO_PROVIDER 选择 provider。"""
        os.environ["YUKO_PROVIDER"] = "anthropic"
        try:
            config = resolve_provider_config(start="/tmp/nonexistent")
            assert config.name == "anthropic"
            assert config.protocol == "anthropic"
        finally:
            os.environ.pop("YUKO_PROVIDER", None)

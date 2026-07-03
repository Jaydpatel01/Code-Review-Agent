"""Unit tests for config.py settings loading."""

import os
from unittest.mock import patch
from code_reviewer.config import load_settings, Settings


def test_settings_defaults():
    """Test default Settings options when no config file is present."""
    settings = Settings()
    assert settings.model == "gemini/gemini-3.1-flash-lite"
    assert settings.max_tokens == 2048
    assert settings.severity_threshold == "MEDIUM"
    assert settings.rules.complexity.enabled is True
    assert settings.rules.docs.enabled is False
    assert settings.output.format == "pretty"


def test_load_settings_from_yaml(tmp_path):
    """Test loading Settings from a YAML file."""
    config_file = tmp_path / ".codereviewer.yaml"
    config_content = """
model: custom-model
max_tokens: 1024
severity_threshold: HIGH
rules:
  complexity:
    enabled: false
  docs:
    enabled: true
output:
  format: json
"""
    config_file.write_text(config_content)

    settings = load_settings(str(config_file))

    assert settings.model == "custom-model"
    assert settings.max_tokens == 1024
    assert settings.severity_threshold == "HIGH"
    assert settings.rules.complexity.enabled is False
    assert settings.rules.docs.enabled is True
    assert settings.output.format == "json"


def test_settings_env_override():
    """Test environment variables overriding settings attributes."""
    with patch.dict(
        os.environ, {"CODEREVIEWER_MODEL": "env-model", "CODEREVIEWER_SEVERITY_THRESHOLD": "LOW"}
    ):
        settings = Settings()
        assert settings.model == "env-model"
        assert settings.severity_threshold == "LOW"

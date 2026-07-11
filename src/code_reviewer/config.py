"""Configuration loader for the code reviewer using pydantic-settings."""

import os
from typing import Literal, Dict, Any, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file if present
load_dotenv()


class ComplexityRules(BaseModel):
    """Rules for code complexity checks."""
    enabled: bool = True
    max_cyclomatic_complexity: int = 10
    max_function_length: int = 50

class NestingRules(BaseModel):
    """Rules for code nesting depth."""
    enabled: bool = True
    max_nesting_depth: int = 4

class MutableDefaultsRules(BaseModel):
    """Rules for mutable default arguments."""
    enabled: bool = True

class MagicNumbersRules(BaseModel):
    """Rules for magic numbers."""
    enabled: bool = False

class SecurityRules(BaseModel):
    """Rules for security analysis."""
    enabled: bool = True


class StyleRules(BaseModel):
    """Rules for style checks."""
    enabled: bool = True


class DocsRules(BaseModel):
    """Rules for checking documentation coverage."""
    enabled: bool = False


class RulesSettings(BaseModel):
    """Nested rules configurations."""
    complexity: ComplexityRules = Field(default_factory=ComplexityRules)
    nesting: NestingRules = Field(default_factory=NestingRules)
    mutable_defaults: MutableDefaultsRules = Field(default_factory=MutableDefaultsRules)
    magic_numbers: MagicNumbersRules = Field(default_factory=MagicNumbersRules)
    security: SecurityRules = Field(default_factory=SecurityRules)
    style: StyleRules = Field(default_factory=StyleRules)
    docs: DocsRules = Field(default_factory=DocsRules)


class OutputSettings(BaseModel):
    """Configuration for CLI output formats."""
    format: Literal["pretty", "json", "github"] = "pretty"
    show_suggestions: bool = True


class Settings(BaseSettings):
    """Main Settings configuration class backed by pydantic-settings."""
    model: str = "gemini/gemini-3.1-flash-lite"
    max_tokens: int = 2048
    severity_threshold: Literal["HIGH", "MEDIUM", "LOW", "INFO"] = "MEDIUM"
    rules: RulesSettings = Field(default_factory=RulesSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    # Configuration for pydantic-settings to allow environment overrides
    model_config = SettingsConfigDict(
        env_prefix="CODEREVIEWER_",
        env_nested_delimiter="__",
        extra="ignore"
    )


def load_settings(config_path: str = ".codereviewer.yaml") -> Settings:
    """
    Load settings from a YAML configuration file, with fallback/override from env vars.

    Args:
        config_path: Filepath to the YAML settings file.

    Returns:
        Settings: Configured settings instance.
    """
    config_data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
            except yaml.YAMLError:
                # Fall back to empty config data if file is invalid YAML
                pass

    return Settings(**config_data)

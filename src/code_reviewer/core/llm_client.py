"""LiteLLM abstraction client with retries and structured output support."""

import time
import logging
from typing import Type, TypeVar, List, Dict, Any, Optional
import litellm
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLMClient operations."""
    pass


class LLMClient:
    """Handles LLM completions via LiteLLM with structured outputs and retry logic."""

    def __init__(
        self,
        model: str = "gemini/gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize the LLM Client.

        Args:
            model: The target LLM model identifier.
            api_key: Optional API key. Resolves from environment variables if None.
            api_base: Optional custom API endpoint URL.
            max_tokens: Maximum completion tokens.
            temperature: Randomness control.
            max_retries: Number of transient retries.
            backoff_factor: Multiplier for exponential backoff delay.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[T]] = None,
    ) -> T | str:
        """
        Generate completion content, potentially parsing into a Pydantic structure.

        Args:
            messages: Role-content formatted messages.
            response_format: Target Pydantic model for structured output validation.

        Returns:
            The raw text response, or an instance of the response_format Pydantic model.

        Raises:
            LLMClientError: If completion fails or output validation fails after retries.
        """
        attempts = 0
        delay = 1.0

        while attempts <= self.max_retries:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_base:
                    kwargs["api_base"] = self.api_base

                if response_format:
                    kwargs["response_format"] = response_format

                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                if response_format:
                    if isinstance(content, str):
                        try:
                            return response_format.model_validate_json(content)
                        except Exception as parse_err:
                            logger.warning(f"Direct JSON validation failed: {parse_err}. Attempting sanitization...")
                            # Sanitize potential markdown wrap
                            clean = content.strip()
                            if clean.startswith("```json"):
                                clean = clean[7:]
                            if clean.endswith("```"):
                                clean = clean[:-3]
                            return response_format.model_validate_json(clean.strip())
                    elif isinstance(content, response_format):
                        return content
                    else:
                        return response_format.model_validate(content)

                return content

            except Exception as e:
                attempts += 1
                if attempts > self.max_retries:
                    raise LLMClientError(f"LLM execution failed after {self.max_retries} attempts: {str(e)}") from e
                logger.warning(f"Transient error: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= self.backoff_factor

        raise LLMClientError("LLM execution fell through retry logic.")

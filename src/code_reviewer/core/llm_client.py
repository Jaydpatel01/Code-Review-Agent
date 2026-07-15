"""LiteLLM abstraction client with retries and structured output support."""

import time
import logging
from typing import Type, TypeVar, List, Dict, Any, Optional
import litellm
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Exceptions that represent transient API/network problems and are safe to retry.
_RETRYABLE_EXCEPTIONS = (
    litellm.APIError,
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
)


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
        initial_delay: float = 1.0,
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
            initial_delay: Starting delay in seconds before the first retry.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[T]],
    ) -> Dict[str, Any]:
        """Build the keyword-argument dict for litellm.completion."""
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
        return kwargs

    def _parse_response(self, content: Any, response_format: Type[T]) -> T:
        """
        Parse raw completion content into the requested Pydantic model.

        Handles three cases:
          1. content is a plain string — attempt direct JSON validation.
          2. Direct validation fails — strip Markdown code fences and retry.
          3. content is already the target type — return it directly.
          4. content is another dict/object — fall back to model_validate.

        Args:
            content: Raw content from the LiteLLM response.
            response_format: Target Pydantic model class.

        Returns:
            A validated instance of response_format.

        Raises:
            ValueError: If the content cannot be parsed.
        """
        if isinstance(content, str):
            try:
                return response_format.model_validate_json(content)
            except Exception as parse_err:
                logger.warning(
                    "Direct JSON validation failed: %s. Attempting sanitization...",
                    parse_err,
                )
                clean = content.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                return response_format.model_validate_json(clean.strip())

        if isinstance(content, response_format):
            return content

        return response_format.model_validate(content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[T]] = None,
    ) -> T | str:
        """
        Generate completion content, potentially parsing into a Pydantic structure.

        Only transient API/network errors are retried (litellm.APIError,
        litellm.Timeout, litellm.RateLimitError, etc.). Programming errors
        (NameError, TypeError, …) propagate immediately.

        Args:
            messages: Role-content formatted messages.
            response_format: Target Pydantic model for structured output validation.

        Returns:
            The raw text response, or an instance of the response_format Pydantic model.

        Raises:
            LLMClientError: If completion fails or output validation fails after retries.
        """
        attempts = 0
        delay = self.initial_delay
        kwargs = self._build_kwargs(messages, response_format)

        while attempts <= self.max_retries:
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                if response_format:
                    return self._parse_response(content, response_format)

                return content

            except _RETRYABLE_EXCEPTIONS as e:
                attempts += 1
                if attempts > self.max_retries:
                    raise LLMClientError(
                        f"LLM execution failed after {self.max_retries} attempts: {e}"
                    ) from e
                logger.warning("Transient error: %s. Retrying in %.1fs...", e, delay)
                time.sleep(delay)
                delay *= self.backoff_factor

        raise LLMClientError("LLM execution fell through retry logic.")

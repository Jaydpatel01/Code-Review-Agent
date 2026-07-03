"""Unit tests for LiteLLM client abstraction."""

import pytest
from unittest.mock import MagicMock
from code_reviewer.core.llm_client import LLMClient, LLMClientError
from code_reviewer.core.models import LLMReviewResponse, LLMFinding


def test_llm_client_initialization():
    """Test LLMClient config settings initialization."""
    client = LLMClient(model="test-model", temperature=0.5)
    assert client.model == "test-model"
    assert client.temperature == 0.5
    assert client.max_retries == 3


def test_llm_client_string_completion(mocker):
    """Test LLMClient successful raw text response."""
    mock_completion = mocker.patch("litellm.completion")

    # Mock return value of litellm.completion
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Review completed."
    mock_completion.return_value = mock_response

    client = LLMClient(max_retries=1)
    res = client.generate_completion([{"role": "user", "content": "hello"}])

    assert res == "Review completed."
    mock_completion.assert_called_once()


def test_llm_client_structured_completion(mocker):
    """Test LLMClient returning parsed Pydantic structures."""
    mock_completion = mocker.patch("litellm.completion")

    # Mock JSON string representation returned by LLM
    json_data = (
        '{"findings": [{"line_number": 5, "severity": "HIGH", "category": "security", '
        '"message": "SQL Injection", "suggestion": "Use parameterized query"}], '
        '"summary": "One vulnerability found."}'
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json_data
    mock_completion.return_value = mock_response

    client = LLMClient(max_retries=1)
    res = client.generate_completion(
        [{"role": "user", "content": "hello"}], response_format=LLMReviewResponse
    )

    assert isinstance(res, LLMReviewResponse)
    assert res.summary == "One vulnerability found."
    assert len(res.findings) == 1
    assert res.findings[0].category == "security"


def test_llm_client_retry_logic(mocker):
    """Test LLMClient retry logic executing for transient errors."""
    mock_completion = mocker.patch("litellm.completion")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success"

    mock_completion.side_effect = [Exception("API Error"), Exception("API Error"), mock_response]

    # Setup sleep mock so the test runs instantly
    mocker.patch("time.sleep")

    client = LLMClient(max_retries=2, backoff_factor=1.0)
    res = client.generate_completion([{"role": "user", "content": "hello"}])

    assert res == "Success"
    assert mock_completion.call_count == 3



def test_llm_client_failure_exhaustion(mocker):
    """Test LLMClient raising LLMClientError on total retry failure."""
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.side_effect = Exception("API Connection failure")
    mocker.patch("time.sleep")

    client = LLMClient(max_retries=2)
    with pytest.raises(LLMClientError) as excinfo:
        client.generate_completion([{"role": "user", "content": "hello"}])

    assert "LLM execution failed after" in str(excinfo.value)
    assert mock_completion.call_count == 3

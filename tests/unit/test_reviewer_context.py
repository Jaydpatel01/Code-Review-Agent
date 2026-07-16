"""Unit tests for context integration in FileReviewer and DiffReviewer."""

import pytest
from pathlib import Path

from code_reviewer.core.reviewer import FileReviewer, DiffReviewer
from code_reviewer.core.models import (
    LLMReviewResponse,
    LLMFinding,
    ReviewResult,
    DiffHunk,
    Finding,
)
from code_reviewer.config import Settings


def test_file_reviewer_with_context(mocker, tmp_path):
    """Test FileReviewer correctly prepends additional_context to the user message."""
    # Create a dummy file
    dummy_file = tmp_path / "app.py"
    dummy_file.write_text("def main():\n    print('hello')\n")

    # Mock LLMClient
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"
    
    # Capture the messages sent to the LLM
    captured_messages = []
    
    def capture_messages(messages, response_format):
        captured_messages.extend(messages)
        return LLMReviewResponse(findings=[], summary="OK")
    
    mock_client.generate_completion.side_effect = capture_messages

    settings = Settings()
    reviewer = FileReviewer(llm_client=mock_client, settings=settings)
    
    # Test with context
    context = "## Codebase Context\n\nSimilar functions found:\n- helper() in utils.py"
    result = reviewer.review_file(str(dummy_file), additional_context=context)

    assert isinstance(result, ReviewResult)
    assert len(captured_messages) == 2  # system + user message
    
    user_message = captured_messages[1]["content"]
    
    # Context should be prepended to the user message
    assert user_message.startswith(context)
    assert "## Codebase Context" in user_message
    assert str(dummy_file) in user_message
    assert "def main():" in user_message


def test_file_reviewer_without_context(mocker, tmp_path):
    """Test FileReviewer works normally when no context is provided."""
    # Create a dummy file
    dummy_file = tmp_path / "app.py"
    dummy_file.write_text("def main():\n    print('hello')\n")

    # Mock LLMClient
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"
    
    captured_messages = []
    
    def capture_messages(messages, response_format):
        captured_messages.extend(messages)
        return LLMReviewResponse(findings=[], summary="OK")
    
    mock_client.generate_completion.side_effect = capture_messages

    settings = Settings()
    reviewer = FileReviewer(llm_client=mock_client, settings=settings)
    
    # Test without context (default behavior)
    result = reviewer.review_file(str(dummy_file))

    assert isinstance(result, ReviewResult)
    assert len(captured_messages) == 2  # system + user message
    
    user_message = captured_messages[1]["content"]
    
    # Should NOT have context
    assert "## Codebase Context" not in user_message
    assert str(dummy_file) in user_message
    assert "def main():" in user_message


def test_file_reviewer_with_empty_context(mocker, tmp_path):
    """Test FileReviewer handles empty context string correctly."""
    # Create a dummy file
    dummy_file = tmp_path / "app.py"
    dummy_file.write_text("def main():\n    print('hello')\n")

    # Mock LLMClient
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"
    
    captured_messages = []
    
    def capture_messages(messages, response_format):
        captured_messages.extend(messages)
        return LLMReviewResponse(findings=[], summary="OK")
    
    mock_client.generate_completion.side_effect = capture_messages

    settings = Settings()
    reviewer = FileReviewer(llm_client=mock_client, settings=settings)
    
    # Test with empty context string
    result = reviewer.review_file(str(dummy_file), additional_context="")

    assert isinstance(result, ReviewResult)
    user_message = captured_messages[1]["content"]
    
    # Empty context should not add extra newlines
    assert not user_message.startswith("\n\n")


def test_diff_reviewer_with_context_integration(mocker):
    """Test DiffReviewer retrieves context and passes it to run_agent_review."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    # Mock run_agent_review to capture the codebase_context parameter
    captured_context = []
    
    async def _fake_run(hunks, model, settings, codebase_context=""):
        captured_context.append(codebase_context)
        return [
            Finding(
                file_path="app.py",
                line_number=11,
                severity="HIGH",
                category="security",
                message="Test finding",
                suggestion="Fix it",
                source="llm",
            )
        ]

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever to return test context
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    mock_retriever.get_context_for_file.return_value = (
        "## Codebase Context\n\nSimilar functions:\n- helper() in utils.py"
    )
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    hunk = DiffHunk(
        file_path="app.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, 'print("World")')],
        removed_lines=[],
        context_lines=[(10, 'print("Hello")')],
        raw_hunk="",
    )

    result = reviewer.review_hunks("app.py", [hunk])

    assert isinstance(result, ReviewResult)
    assert len(captured_context) == 1
    
    # Context should have been passed to run_agent_review
    assert "## Codebase Context" in captured_context[0]
    assert "helper() in utils.py" in captured_context[0]


def test_diff_reviewer_without_index(mocker):
    """Test DiffReviewer works normally when no index exists."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    # Mock run_agent_review to capture the codebase_context parameter
    captured_context = []
    
    async def _fake_run(hunks, model, settings, codebase_context=""):
        captured_context.append(codebase_context)
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever to return no index
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = False
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    hunk = DiffHunk(
        file_path="app.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, 'print("World")')],
        removed_lines=[],
        context_lines=[(10, 'print("Hello")')],
        raw_hunk="",
    )

    result = reviewer.review_hunks("app.py", [hunk])

    assert isinstance(result, ReviewResult)
    assert len(captured_context) == 1
    
    # Context should be empty string when no index exists
    assert captured_context[0] == ""
    
    # get_context_for_file should not have been called
    mock_retriever.get_context_for_file.assert_not_called()


def test_diff_reviewer_limits_context_retrieval(mocker):
    """Test DiffReviewer limits context to 3 files max, 2 similar per file."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    async def _fake_run(hunks, model, settings, codebase_context=""):
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    mock_retriever.get_context_for_file.return_value = "Context for file"
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    # Create hunks for 5 different files
    hunks = [
        DiffHunk(
            file_path=f"file{i}.py",
            start_line=10,
            end_line=13,
            added_lines=[(11, 'print("test")')],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        )
        for i in range(5)
    ]

    result = reviewer.review_hunks("app.py", hunks)

    assert isinstance(result, ReviewResult)
    
    # get_context_for_file should be called max 3 times (limit)
    assert mock_retriever.get_context_for_file.call_count <= 3
    
    # Each call should request n_similar=2
    for call in mock_retriever.get_context_for_file.call_args_list:
        assert call.kwargs["n_similar"] == 2


def test_diff_reviewer_deduplicates_file_paths(mocker):
    """Test DiffReviewer deduplicates hunks from the same file."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    async def _fake_run(hunks, model, settings, codebase_context=""):
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    mock_retriever.get_context_for_file.return_value = "Context"
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    # Create multiple hunks for the same file
    hunks = [
        DiffHunk(
            file_path="app.py",
            start_line=10 + i * 10,
            end_line=13 + i * 10,
            added_lines=[(11 + i * 10, f'print("test{i}")')],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        )
        for i in range(5)
    ]

    result = reviewer.review_hunks("app.py", hunks)

    assert isinstance(result, ReviewResult)
    
    # get_context_for_file should only be called once for app.py
    assert mock_retriever.get_context_for_file.call_count == 1
    assert mock_retriever.get_context_for_file.call_args[0][0] == "app.py"


def test_diff_reviewer_handles_context_retrieval_error(mocker):
    """Test DiffReviewer continues gracefully when context retrieval fails."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    captured_context = []
    
    async def _fake_run(hunks, model, settings, codebase_context=""):
        captured_context.append(codebase_context)
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever to raise exception
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    mock_retriever.get_context_for_file.side_effect = Exception("Retrieval failed")
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    hunk = DiffHunk(
        file_path="app.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, 'print("World")')],
        removed_lines=[],
        context_lines=[],
        raw_hunk="",
    )

    # Should not crash - exception is caught in the implementation
    result = reviewer.review_hunks("app.py", [hunk])

    assert isinstance(result, ReviewResult)
    # Context should be empty due to exception
    assert len(captured_context) == 1


def test_diff_reviewer_combines_multiple_file_contexts(mocker):
    """Test DiffReviewer combines context from multiple files with newlines."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    captured_context = []
    
    async def _fake_run(hunks, model, settings, codebase_context=""):
        captured_context.append(codebase_context)
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever to return different context for each file
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    
    def get_context_side_effect(file_path, n_similar):
        return f"Context for {file_path}"
    
    mock_retriever.get_context_for_file.side_effect = get_context_side_effect
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    # Create hunks for 2 different files
    hunks = [
        DiffHunk(
            file_path="file1.py",
            start_line=10,
            end_line=13,
            added_lines=[(11, "code")],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        ),
        DiffHunk(
            file_path="file2.py",
            start_line=20,
            end_line=23,
            added_lines=[(21, "code")],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        ),
    ]

    result = reviewer.review_hunks("app.py", hunks)

    assert isinstance(result, ReviewResult)
    assert len(captured_context) == 1
    
    combined = captured_context[0]
    # Both file contexts should be present, separated by double newline
    assert "Context for file1.py" in combined
    assert "Context for file2.py" in combined
    assert "\n\n" in combined


def test_diff_reviewer_skips_empty_contexts(mocker):
    """Test DiffReviewer filters out empty context strings."""
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    captured_context = []
    
    async def _fake_run(hunks, model, settings, codebase_context=""):
        captured_context.append(codebase_context)
        return []

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )
    
    # Mock ContextRetriever to return mix of empty and non-empty contexts
    mock_retriever = mocker.MagicMock()
    mock_retriever.index_exists.return_value = True
    
    def get_context_side_effect(file_path, n_similar):
        if file_path == "file1.py":
            return "Valid context"
        return ""  # Empty context for other files
    
    mock_retriever.get_context_for_file.side_effect = get_context_side_effect
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.ContextRetriever",
        return_value=mock_retriever,
    )

    settings = Settings()
    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    # Create hunks for 2 files
    hunks = [
        DiffHunk(
            file_path="file1.py",
            start_line=10,
            end_line=13,
            added_lines=[(11, "code")],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        ),
        DiffHunk(
            file_path="file2.py",
            start_line=20,
            end_line=23,
            added_lines=[(21, "code")],
            removed_lines=[],
            context_lines=[],
            raw_hunk="",
        ),
    ]

    result = reviewer.review_hunks("app.py", hunks)

    assert isinstance(result, ReviewResult)
    assert len(captured_context) == 1
    
    combined = captured_context[0]
    # Only non-empty context should be included
    assert "Valid context" in combined
    assert combined.count("\n\n") == 0  # No double newlines from empty contexts

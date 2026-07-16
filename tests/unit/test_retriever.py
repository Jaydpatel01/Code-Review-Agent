"""Unit tests for ContextRetriever."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from code_reviewer.retrieval.retriever import ContextRetriever


@pytest.fixture
def mock_store(mocker):
    """Mock CodebaseStore."""
    store = mocker.MagicMock()
    return store


@pytest.fixture
def mock_embedder(mocker):
    """Mock ChunkEmbedder."""
    embedder = mocker.MagicMock()
    embedder.embed_text.return_value = [0.1] * 384  # MiniLM embedding dimension
    return embedder


@pytest.fixture
def retriever(mocker, mock_store, mock_embedder, tmp_path):
    """Create a ContextRetriever with mocked dependencies."""
    mocker.patch(
        "code_reviewer.retrieval.retriever.CodebaseStore",
        return_value=mock_store,
    )
    mocker.patch(
        "code_reviewer.retrieval.retriever.ChunkEmbedder",
        return_value=mock_embedder,
    )
    return ContextRetriever(tmp_path)


def test_index_exists_no_directory(tmp_path):
    """Test index_exists returns False when index directory doesn't exist."""
    retriever = ContextRetriever(tmp_path)
    # Directory doesn't exist yet
    assert not retriever.index_exists()


def test_index_exists_empty_index(mocker, tmp_path):
    """Test index_exists returns False when index has no chunks."""
    # Create the index directory
    index_dir = tmp_path / ".code-reviewer" / "index"
    index_dir.mkdir(parents=True)
    
    # Mock store to return 0 chunks
    mock_store = mocker.MagicMock()
    mock_store.get_stats.return_value = {"total_chunks": 0, "total_files": 0}
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.CodebaseStore",
        return_value=mock_store,
    )
    
    retriever = ContextRetriever(tmp_path)
    assert not retriever.index_exists()


def test_index_exists_valid_index(mocker, tmp_path):
    """Test index_exists returns True when index has chunks."""
    # Create the index directory
    index_dir = tmp_path / ".code-reviewer" / "index"
    index_dir.mkdir(parents=True)
    
    # Mock store to return chunks
    mock_store = mocker.MagicMock()
    mock_store.get_stats.return_value = {"total_chunks": 42, "total_files": 10}
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.CodebaseStore",
        return_value=mock_store,
    )
    
    retriever = ContextRetriever(tmp_path)
    assert retriever.index_exists()


def test_index_exists_handles_exception(mocker, tmp_path):
    """Test index_exists returns False gracefully when store raises exception."""
    # Create the index directory
    index_dir = tmp_path / ".code-reviewer" / "index"
    index_dir.mkdir(parents=True)
    
    # Mock store to raise exception
    mock_store = mocker.MagicMock()
    mock_store.get_stats.side_effect = Exception("ChromaDB error")
    
    mocker.patch(
        "code_reviewer.retrieval.retriever.CodebaseStore",
        return_value=mock_store,
    )
    
    retriever = ContextRetriever(tmp_path)
    assert not retriever.index_exists()


def test_get_context_for_function_no_index(retriever, mocker):
    """Test get_context_for_function returns empty string when no index exists."""
    # Mock index_exists to return False
    mocker.patch.object(retriever, "index_exists", return_value=False)
    
    result = retriever.get_context_for_function("app.py", "main")
    assert result == ""


def test_get_context_for_function_file_not_found(retriever, mocker):
    """Test get_context_for_function returns empty string when file has no chunks."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    retriever.store.get_chunks_by_file.side_effect = Exception("File not found")
    
    result = retriever.get_context_for_function("app.py", "main")
    assert result == ""


def test_get_context_for_function_function_not_found(retriever, mocker):
    """Test get_context_for_function returns empty string when function not found."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Return chunks but none match the target function
    retriever.store.get_chunks_by_file.return_value = [
        {"name": "other_function", "source_code": "def other_function(): pass"}
    ]
    
    result = retriever.get_context_for_function("app.py", "main")
    assert result == ""


def test_get_context_for_function_success(retriever, mocker):
    """Test get_context_for_function successfully retrieves similar functions."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock the target function
    target_chunk = {
        "name": "main",
        "source_code": "def main():\n    print('Hello')",
        "file_path": "app.py",
        "start_line": 1,
        "end_line": 2,
    }
    
    # Mock similar functions
    similar_chunks = [
        target_chunk,  # Will be filtered out
        {
            "name": "helper",
            "source_code": "def helper():\n    print('Help')",
            "file_path": "utils.py",
            "start_line": 10,
            "end_line": 11,
        },
        {
            "name": "worker",
            "source_code": "def worker():\n    print('Work')",
            "file_path": "tasks.py",
            "start_line": 20,
            "end_line": 21,
        },
    ]
    
    retriever.store.get_chunks_by_file.return_value = [target_chunk]
    retriever.store.search_similar.return_value = similar_chunks
    
    result = retriever.get_context_for_function("app.py", "main", n_similar=2)
    
    assert result != ""
    assert "## Similar Functions in Codebase" in result
    assert "helper" in result
    assert "utils.py" in result
    assert "worker" in result
    assert "tasks.py" in result
    # Target function should be filtered out
    assert "app.py" not in result


def test_get_context_for_function_truncates_long_code(retriever, mocker):
    """Test get_context_for_function truncates source code over 30 lines."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Create a function with 40 lines
    long_source = "\n".join([f"line {i}" for i in range(40)])
    
    target_chunk = {
        "name": "main",
        "source_code": "def main(): pass",
        "file_path": "app.py",
    }
    
    similar_chunks = [
        {
            "name": "long_function",
            "source_code": long_source,
            "file_path": "utils.py",
            "start_line": 1,
            "end_line": 40,
        }
    ]
    
    retriever.store.get_chunks_by_file.return_value = [target_chunk]
    retriever.store.search_similar.return_value = similar_chunks
    
    result = retriever.get_context_for_function("app.py", "main")
    
    assert "... (truncated)" in result


def test_get_context_for_file_no_index(retriever, mocker):
    """Test get_context_for_file returns empty string when no index exists."""
    mocker.patch.object(retriever, "index_exists", return_value=False)
    
    result = retriever.get_context_for_file("app.py")
    assert result == ""


def test_get_context_for_file_no_chunks(retriever, mocker):
    """Test get_context_for_file returns empty string when file has no chunks."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    retriever.store.get_chunks_by_file.return_value = []
    
    result = retriever.get_context_for_file("app.py")
    assert result == ""


def test_get_context_for_file_success(retriever, mocker):
    """Test get_context_for_file successfully retrieves similar functions."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock chunks in the target file
    file_chunks = [
        {
            "name": "func1",
            "source_code": "def func1(): pass",
            "file_path": "app.py",
        },
        {
            "name": "func2",
            "source_code": "def func2(): pass",
            "file_path": "app.py",
        },
    ]
    
    # Mock similar functions from other files
    similar_chunks_1 = [
        {
            "name": "helper1",
            "source_code": "def helper1(): pass",
            "file_path": "utils.py",
            "start_line": 10,
            "end_line": 11,
        },
    ]
    
    similar_chunks_2 = [
        {
            "name": "helper2",
            "source_code": "def helper2(): pass",
            "file_path": "tasks.py",
            "start_line": 20,
            "end_line": 21,
        },
    ]
    
    retriever.store.get_chunks_by_file.return_value = file_chunks
    retriever.store.search_similar.side_effect = [similar_chunks_1, similar_chunks_2]
    
    result = retriever.get_context_for_file("app.py", n_similar=3)
    
    assert result != ""
    assert "## Codebase Context" in result
    assert "helper1" in result
    assert "utils.py" in result
    assert "helper2" in result
    assert "tasks.py" in result


def test_get_context_for_file_deduplicates_results(retriever, mocker):
    """Test get_context_for_file deduplicates functions found multiple times."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock chunks in the target file
    file_chunks = [
        {
            "name": "func1",
            "source_code": "def func1(): pass",
            "file_path": "app.py",
        },
        {
            "name": "func2",
            "source_code": "def func2(): pass",
            "file_path": "app.py",
        },
    ]
    
    # Both searches return the same similar function
    duplicate_result = {
        "name": "helper",
        "source_code": "def helper(): pass",
        "file_path": "utils.py",
        "start_line": 10,
        "end_line": 11,
    }
    
    retriever.store.get_chunks_by_file.return_value = file_chunks
    retriever.store.search_similar.side_effect = [
        [duplicate_result],
        [duplicate_result],
    ]
    
    result = retriever.get_context_for_file("app.py", n_similar=3)
    
    # Should only appear once in the headers (deduplication working)
    assert result.count("### helper") == 1
    assert result.count("utils.py:10") == 1


def test_get_context_for_file_filters_same_file(retriever, mocker):
    """Test get_context_for_file filters out results from the same file."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock chunks in the target file
    file_chunks = [
        {
            "name": "func1",
            "source_code": "def func1(): pass",
            "file_path": "app.py",
        },
    ]
    
    # Search returns functions from the same file (should be filtered)
    similar_chunks = [
        {
            "name": "func2",
            "source_code": "def func2(): pass",
            "file_path": "app.py",  # Same file
            "start_line": 10,
            "end_line": 11,
        },
        {
            "name": "helper",
            "source_code": "def helper(): pass",
            "file_path": "utils.py",  # Different file
            "start_line": 20,
            "end_line": 21,
        },
    ]
    
    retriever.store.get_chunks_by_file.return_value = file_chunks
    retriever.store.search_similar.return_value = similar_chunks
    
    result = retriever.get_context_for_file("app.py", n_similar=3)
    
    # Should only include helper from utils.py, not func2 from app.py
    assert "helper" in result
    assert "utils.py" in result
    assert result.count("app.py") == 0


def test_get_context_for_file_handles_embedding_error(retriever, mocker):
    """Test get_context_for_file continues gracefully when embedding fails."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock chunks in the target file
    file_chunks = [
        {
            "name": "func1",
            "source_code": "def func1(): pass",
            "file_path": "app.py",
        },
    ]
    
    retriever.store.get_chunks_by_file.return_value = file_chunks
    retriever.embedder.embed_text.side_effect = Exception("Embedding failed")
    
    result = retriever.get_context_for_file("app.py")
    
    # Should return empty string, not crash
    assert result == ""


def test_get_context_for_file_handles_search_error(retriever, mocker):
    """Test get_context_for_file continues gracefully when search fails."""
    mocker.patch.object(retriever, "index_exists", return_value=True)
    
    # Mock chunks in the target file
    file_chunks = [
        {
            "name": "func1",
            "source_code": "def func1(): pass",
            "file_path": "app.py",
        },
    ]
    
    retriever.store.get_chunks_by_file.return_value = file_chunks
    retriever.store.search_similar.side_effect = Exception("Search failed")
    
    result = retriever.get_context_for_file("app.py")
    
    # Should return empty string, not crash
    assert result == ""


def test_format_similar_functions_empty_list(retriever):
    """Test _format_similar_functions returns empty string for empty list."""
    result = retriever._format_similar_functions([])
    assert result == ""


def test_format_similar_functions_single_chunk(retriever):
    """Test _format_similar_functions formats a single chunk correctly."""
    chunks = [
        {
            "name": "helper",
            "source_code": "def helper():\n    return 42",
            "file_path": "utils.py",
            "start_line": 10,
            "end_line": 11,
        }
    ]
    
    result = retriever._format_similar_functions(chunks)
    
    assert "## Similar Functions in Codebase" in result
    assert "helper (utils.py:10–11)" in result
    assert "```python" in result
    assert "def helper():" in result
    assert "return 42" in result

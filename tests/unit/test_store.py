"""Unit tests for ChromaDB store."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from code_reviewer.indexer.chunker import CodeChunk
from code_reviewer.indexer.store import CodebaseStore


@pytest.fixture
def mock_chroma_client():
    """Provide a mocked ChromaDB client."""
    with patch("code_reviewer.indexer.store.chromadb.PersistentClient") as mock_client:
        # Mock the client instance
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock collections
        chunks_collection = MagicMock()
        files_collection = MagicMock()
        findings_collection = MagicMock()

        client_instance.get_or_create_collection.side_effect = [
            chunks_collection,
            files_collection,
            findings_collection,
        ]

        yield {
            "client": mock_client,
            "instance": client_instance,
            "chunks": chunks_collection,
            "files": files_collection,
            "findings": findings_collection,
        }


@pytest.fixture
def store(mock_chroma_client, tmp_path):
    """Provide a CodebaseStore instance with mocked ChromaDB."""
    return CodebaseStore(tmp_path)


@pytest.fixture
def sample_chunk():
    """Provide a sample CodeChunk for testing."""
    return CodeChunk(
        chunk_id="test_chunk_123",
        file_path="/test/file.py",
        name="test_function",
        chunk_type="function",
        start_line=10,
        end_line=20,
        source_code="def test_function():\n    pass",
        docstring="Test function docstring",
        calls=["helper", "print"],
        complexity=3,
        file_hash="abc123",
    )


def test_upsert_chunk_stores_without_error(store, sample_chunk, mock_chroma_client):
    """Test that upsert_chunk stores a chunk without raising errors."""
    embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector

    # Should not raise
    store.upsert_chunk(sample_chunk, embedding)

    # Verify upsert was called
    chunks_collection = mock_chroma_client["chunks"]
    chunks_collection.upsert.assert_called_once()

    # Verify correct data was passed
    call_args = chunks_collection.upsert.call_args
    assert call_args.kwargs["ids"] == [sample_chunk.chunk_id]
    assert call_args.kwargs["embeddings"] == [embedding]
    assert call_args.kwargs["documents"] == [sample_chunk.source_code]


def test_delete_by_file_removes_chunks(store, mock_chroma_client):
    """Test that delete_by_file removes all chunks for a file."""
    file_path = "/test/file.py"

    # Mock get() to return some chunk IDs
    chunks_collection = mock_chroma_client["chunks"]
    chunks_collection.get.return_value = {
        "ids": ["chunk1", "chunk2", "chunk3"]
    }

    store.delete_by_file(file_path)

    # Verify get was called with correct filter
    chunks_collection.get.assert_called_once()
    call_args = chunks_collection.get.call_args
    assert call_args.kwargs["where"] == {"file_path": file_path}

    # Verify delete was called with the IDs
    chunks_collection.delete.assert_called_once_with(
        ids=["chunk1", "chunk2", "chunk3"]
    )


def test_get_file_hash_returns_none_for_unknown_file(store, mock_chroma_client):
    """Test that get_file_hash returns None for unknown files."""
    files_collection = mock_chroma_client["files"]
    files_collection.get.return_value = {"ids": [], "metadatas": []}

    result = store.get_file_hash("/unknown/file.py")

    assert result is None


def test_get_file_hash_returns_stored_hash(store, mock_chroma_client):
    """Test that get_file_hash returns the stored hash."""
    file_path = "/test/file.py"
    expected_hash = "abc123def456"

    files_collection = mock_chroma_client["files"]
    files_collection.get.return_value = {
        "ids": [file_path],
        "metadatas": [{"file_hash": expected_hash}],
    }

    result = store.get_file_hash(file_path)

    assert result == expected_hash


def test_get_stats_returns_dict_with_expected_keys(store, mock_chroma_client):
    """Test that get_stats returns a dict with expected keys."""
    chunks_collection = mock_chroma_client["chunks"]
    files_collection = mock_chroma_client["files"]

    # Mock counts
    chunks_collection.count.return_value = 150
    files_collection.count.return_value = 10

    # Mock files metadata
    files_collection.get.return_value = {
        "metadatas": [
            {"indexed_at": "2024-01-01T10:00:00Z"},
            {"indexed_at": "2024-01-02T15:30:00Z"},
        ]
    }

    result = store.get_stats()

    assert isinstance(result, dict)
    assert "total_files" in result
    assert "total_chunks" in result
    assert "last_indexed" in result

    assert result["total_files"] == 10
    assert result["total_chunks"] == 150
    assert result["last_indexed"] == "2024-01-02T15:30:00Z"


def test_upsert_file_record_stores_metadata(store, mock_chroma_client):
    """Test that upsert_file_record stores file metadata."""
    file_path = "/test/file.py"
    file_hash = "test_hash_789"
    chunk_count = 5

    store.upsert_file_record(file_path, file_hash, chunk_count)

    files_collection = mock_chroma_client["files"]
    files_collection.upsert.assert_called_once()

    call_args = files_collection.upsert.call_args
    assert call_args.kwargs["ids"] == [file_path]
    assert call_args.kwargs["documents"] == [file_path]

    metadata = call_args.kwargs["metadatas"][0]
    assert metadata["file_path"] == file_path
    assert metadata["file_hash"] == file_hash
    assert metadata["chunk_count"] == chunk_count
    assert "indexed_at" in metadata


def test_get_chunks_by_file_returns_list(store, mock_chroma_client):
    """Test that get_chunks_by_file returns a list of chunk metadata."""
    file_path = "/test/file.py"

    chunks_collection = mock_chroma_client["chunks"]
    chunks_collection.get.return_value = {
        "ids": ["chunk1", "chunk2"],
        "documents": ["code1", "code2"],
        "metadatas": [
            {"name": "func1", "chunk_type": "function"},
            {"name": "func2", "chunk_type": "function"},
        ],
    }

    result = store.get_chunks_by_file(file_path)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "chunk1"
    assert result[0]["source_code"] == "code1"
    assert result[0]["name"] == "func1"


def test_search_similar_returns_list(store, mock_chroma_client):
    """Test that search_similar returns a list of similar chunks."""
    embedding = [0.5] * 384

    chunks_collection = mock_chroma_client["chunks"]
    chunks_collection.query.return_value = {
        "ids": [["chunk1", "chunk2"]],
        "documents": [["code1", "code2"]],
        "metadatas": [[{"name": "func1"}, {"name": "func2"}]],
        "distances": [[0.1, 0.2]],
    }

    result = store.search_similar(embedding, n=2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "chunk1"
    assert result[0]["distance"] == 0.1


def test_store_handles_errors_gracefully(store, mock_chroma_client):
    """Test that store methods don't raise on ChromaDB errors."""
    chunks_collection = mock_chroma_client["chunks"]
    chunks_collection.get.side_effect = Exception("ChromaDB error")

    # Should not raise, should return safe default
    result = store.get_chunks_by_file("/test/file.py")
    assert result == []


def test_store_creates_directory(tmp_path):
    """Test that store creates the index directory on initialization."""
    with patch("code_reviewer.indexer.store.chromadb.PersistentClient"):
        store = CodebaseStore(tmp_path)
        
        index_dir = tmp_path / ".code-reviewer" / "index"
        assert index_dir.exists()
        assert index_dir.is_dir()

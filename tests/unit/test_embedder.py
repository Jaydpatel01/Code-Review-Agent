"""Unit tests for code embedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_reviewer.indexer.chunker import CodeChunk
from code_reviewer.indexer.embedder import ChunkEmbedder


@pytest.fixture
def mock_sentence_transformer():
    """Provide a mocked SentenceTransformer."""
    with patch("code_reviewer.indexer.embedder.SentenceTransformer") as mock_st:
        # Create a mock model instance
        model_instance = MagicMock()
        mock_st.return_value = model_instance

        # Mock encode to return numpy arrays
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                # Single text -> single embedding
                return np.array([0.1] * 384)
            else:
                # List of texts -> list of embeddings
                return np.array([[0.1] * 384 for _ in text])

        model_instance.encode = MagicMock(side_effect=mock_encode)

        yield {"class": mock_st, "instance": model_instance}


@pytest.fixture
def embedder(mock_sentence_transformer):
    """Provide a ChunkEmbedder instance with mocked model."""
    return ChunkEmbedder()


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
        source_code="def test_function():\n    return 42",
        docstring="Test function docstring",
        calls=["print"],
        complexity=1,
        file_hash="abc123",
    )


def test_embed_text_returns_list_of_floats(embedder):
    """Test that embed_text returns a list of floats."""
    text = "sample code snippet"

    result = embedder.embed_text(text)

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_embed_chunk_returns_list_of_floats(embedder, sample_chunk):
    """Test that embed_chunk returns a list of floats."""
    result = embedder.embed_chunk(sample_chunk)

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_embed_chunks_returns_one_embedding_per_chunk(embedder, sample_chunk):
    """Test that embed_chunks returns one embedding per chunk."""
    # Create multiple chunks
    chunk1 = sample_chunk
    chunk2 = CodeChunk(
        chunk_id="test_chunk_456",
        file_path="/test/file.py",
        name="another_function",
        chunk_type="function",
        start_line=30,
        end_line=40,
        source_code="def another_function():\n    pass",
        docstring=None,
        calls=[],
        complexity=1,
        file_hash="abc123",
    )

    chunks = [chunk1, chunk2]

    result = embedder.embed_chunks(chunks)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(emb, list) for emb in result)
    assert all(len(emb) == 384 for emb in result)


def test_embed_chunks_empty_list_returns_empty_list(embedder):
    """Test that embed_chunks([]) returns []."""
    result = embedder.embed_chunks([])

    assert result == []


def test_embed_chunk_uses_name_docstring_and_code(embedder, sample_chunk, mock_sentence_transformer):
    """Test that embed_chunk builds text from name, docstring, and source code."""
    embedder.embed_chunk(sample_chunk)

    # Verify encode was called
    model = mock_sentence_transformer["instance"]
    model.encode.assert_called()

    # Get the text that was passed to encode
    call_args = model.encode.call_args
    text_arg = call_args[0][0]

    # Should contain name, docstring, and source code
    assert sample_chunk.name in text_arg
    assert sample_chunk.docstring in text_arg
    assert sample_chunk.source_code in text_arg


def test_embed_chunks_uses_batch_processing(embedder, sample_chunk, mock_sentence_transformer):
    """Test that embed_chunks uses batch processing."""
    chunks = [sample_chunk] * 5

    embedder.embed_chunks(chunks)

    # Verify encode was called once with a list (batch)
    model = mock_sentence_transformer["instance"]
    model.encode.assert_called_once()

    call_args = model.encode.call_args
    # First positional argument should be a list
    assert isinstance(call_args[0][0], list)
    assert len(call_args[0][0]) == 5

    # Verify batch_size parameter
    assert call_args.kwargs.get("batch_size") == 64
    assert call_args.kwargs.get("show_progress_bar") is False


def test_embedder_loads_model_on_init(mock_sentence_transformer):
    """Test that embedder loads the model on initialization."""
    ChunkEmbedder()

    # Verify model was loaded
    mock_st = mock_sentence_transformer["class"]
    mock_st.assert_called_once_with("all-MiniLM-L6-v2")


def test_embed_text_converts_to_list(embedder, mock_sentence_transformer):
    """Test that embed_text converts numpy array to list."""
    result = embedder.embed_text("test text")

    # Result should be a Python list, not numpy array
    assert isinstance(result, list)
    assert not isinstance(result, np.ndarray)


def test_embed_chunk_with_none_docstring(embedder):
    """Test that embed_chunk handles chunks with None docstring."""
    chunk = CodeChunk(
        chunk_id="test_chunk_789",
        file_path="/test/file.py",
        name="no_doc_function",
        chunk_type="function",
        start_line=50,
        end_line=60,
        source_code="def no_doc_function():\n    pass",
        docstring=None,  # No docstring
        calls=[],
        complexity=1,
        file_hash="def456",
    )

    # Should not raise
    result = embedder.embed_chunk(chunk)

    assert isinstance(result, list)
    assert len(result) == 384

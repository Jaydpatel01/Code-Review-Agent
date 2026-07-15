"""Unit tests for code chunker."""

from pathlib import Path

import pytest

from code_reviewer.indexer.chunker import CodeChunk, PythonChunker


@pytest.fixture
def chunker():
    """Provide a PythonChunker instance."""
    return PythonChunker()


@pytest.fixture
def good_code_path():
    """Path to sample good Python code."""
    return Path("tests/fixtures/sample_code/good_code.py")


@pytest.fixture
def bad_code_path():
    """Path to sample bad Python code."""
    return Path("tests/fixtures/sample_code/bad_code.py")


def test_chunk_good_code_returns_chunks(chunker, good_code_path):
    """Test that chunking good code returns at least one CodeChunk."""
    source = good_code_path.read_text()
    file_hash = "test_hash_123"

    chunks = chunker.chunk(good_code_path, source, file_hash)

    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)


def test_chunk_bad_code_has_complexity(chunker, bad_code_path):
    """Test that bad code chunks have complexity > 1."""
    source = bad_code_path.read_text()
    file_hash = "test_hash_456"

    chunks = chunker.chunk(bad_code_path, source, file_hash)

    # Bad code should have some chunks with higher complexity
    assert len(chunks) > 0
    # At least one chunk should have complexity > 1
    assert any(chunk.complexity > 1 for chunk in chunks)


def test_chunk_syntax_error_returns_empty_list(chunker):
    """Test that chunking code with SyntaxError returns empty list."""
    invalid_source = """
def broken_function(
    # Missing closing parenthesis and colon
    print("This won't parse")
"""
    file_path = Path("test_broken.py")
    file_hash = "test_hash_789"

    chunks = chunker.chunk(file_path, invalid_source, file_hash)

    assert chunks == []


def test_all_chunks_have_source_code(chunker, good_code_path):
    """Test that all returned chunks have non-empty source_code."""
    source = good_code_path.read_text()
    file_hash = "test_hash_abc"

    chunks = chunker.chunk(good_code_path, source, file_hash)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.source_code
        assert len(chunk.source_code.strip()) > 0


def test_chunk_ids_are_unique(chunker, good_code_path):
    """Test that chunk_id is unique across all chunks from same file."""
    source = good_code_path.read_text()
    file_hash = "test_hash_def"

    chunks = chunker.chunk(good_code_path, source, file_hash)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    # All chunk IDs should be unique
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_extracts_docstrings(chunker):
    """Test that docstrings are properly extracted."""
    source = '''
def example_function():
    """This is a docstring."""
    return 42
'''
    file_path = Path("test_docstring.py")
    file_hash = "test_hash_ghi"

    chunks = chunker.chunk(file_path, source, file_hash)

    assert len(chunks) > 0
    function_chunk = next(c for c in chunks if c.name == "example_function")
    assert function_chunk.docstring == "This is a docstring."


def test_chunk_skips_private_functions(chunker):
    """Test that private functions (starting with _) are skipped except __init__."""
    source = '''
def public_function():
    pass

def _private_function():
    pass

class MyClass:
    def __init__(self):
        pass
    
    def _private_method(self):
        pass
'''
    file_path = Path("test_private.py")
    file_hash = "test_hash_jkl"

    chunks = chunker.chunk(file_path, source, file_hash)

    chunk_names = [c.name for c in chunks]
    
    # Should include public_function and __init__
    assert "public_function" in chunk_names
    assert "__init__" in chunk_names
    
    # Should NOT include _private_function or _private_method
    assert "_private_function" not in chunk_names
    assert "_private_method" not in chunk_names


def test_chunk_identifies_methods_vs_functions(chunker):
    """Test that methods are correctly identified as 'method' type."""
    source = '''
def standalone_function():
    pass

class MyClass:
    def my_method(self):
        pass
'''
    file_path = Path("test_types.py")
    file_hash = "test_hash_mno"

    chunks = chunker.chunk(file_path, source, file_hash)

    function_chunk = next(c for c in chunks if c.name == "standalone_function")
    method_chunk = next(c for c in chunks if c.name == "my_method")

    assert function_chunk.chunk_type == "function"
    assert method_chunk.chunk_type == "method"


def test_chunk_extracts_calls(chunker):
    """Test that function calls are extracted from chunks."""
    source = '''
def helper():
    pass

def main():
    helper()
    print("test")
    len([1, 2, 3])
'''
    file_path = Path("test_calls.py")
    file_hash = "test_hash_pqr"

    chunks = chunker.chunk(file_path, source, file_hash)

    main_chunk = next(c for c in chunks if c.name == "main")
    
    # Should have captured the function calls
    assert "helper" in main_chunk.calls
    assert "print" in main_chunk.calls
    assert "len" in main_chunk.calls

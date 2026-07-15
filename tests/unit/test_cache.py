"""Unit tests for file hash cache."""

import json
from pathlib import Path

import pytest

from code_reviewer.indexer.cache import FileHashCache, compute_file_hash


@pytest.fixture
def temp_project_root(tmp_path):
    """Provide a temporary project root directory."""
    return tmp_path / "test_project"


@pytest.fixture
def cache(temp_project_root):
    """Provide a FileHashCache instance."""
    temp_project_root.mkdir(parents=True, exist_ok=True)
    return FileHashCache(temp_project_root)


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "sample.py"
    file_path.write_text("def example():\n    pass\n")
    return file_path


def test_compute_file_hash_returns_consistent_hash(sample_file):
    """Test that compute_file_hash returns consistent SHA256 for same content."""
    hash1 = compute_file_hash(sample_file)
    hash2 = compute_file_hash(sample_file)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex characters
    assert hash1 != ""


def test_compute_file_hash_error_returns_empty_string():
    """Test that compute_file_hash returns empty string on error."""
    nonexistent_file = Path("/nonexistent/path/to/file.py")
    
    hash_result = compute_file_hash(nonexistent_file)
    
    assert hash_result == ""


def test_is_changed_returns_true_for_new_file(cache, sample_file):
    """Test that is_changed returns True for a file not in cache."""
    changed, file_hash = cache.is_changed(sample_file)

    assert changed is True
    assert file_hash != ""
    assert len(file_hash) == 64


def test_is_changed_returns_false_after_update_and_save(cache, sample_file):
    """Test that is_changed returns False for unchanged file after update+save."""
    # First check - file is new
    changed1, hash1 = cache.is_changed(sample_file)
    assert changed1 is True

    # Update cache and save
    cache.update(str(sample_file), hash1)
    cache.save()

    # Second check - file unchanged
    changed2, hash2 = cache.is_changed(sample_file)
    assert changed2 is False
    assert hash2 == hash1


def test_is_changed_returns_true_after_file_content_changes(cache, sample_file):
    """Test that is_changed returns True after file content changes."""
    # Initial state
    changed1, hash1 = cache.is_changed(sample_file)
    cache.update(str(sample_file), hash1)
    cache.save()

    # Modify the file
    sample_file.write_text("def example():\n    return 42\n")

    # Check again
    changed2, hash2 = cache.is_changed(sample_file)
    assert changed2 is True
    assert hash2 != hash1


def test_save_writes_valid_json(cache, sample_file, temp_project_root):
    """Test that save() writes valid JSON that can be reloaded."""
    # Add some entries
    cache.update(str(sample_file), "abc123")
    cache.update("/path/to/another.py", "def456")
    cache.save()

    # Check that cache file exists
    cache_file = temp_project_root / ".code-reviewer" / "cache" / "hashes.json"
    assert cache_file.exists()

    # Load and verify JSON
    with open(cache_file, "r") as f:
        data = json.load(f)

    assert str(sample_file) in data
    assert data[str(sample_file)] == "abc123"
    assert "/path/to/another.py" in data
    assert data["/path/to/another.py"] == "def456"


def test_get_returns_none_for_unknown_file(cache):
    """Test that get() returns None for files not in cache."""
    result = cache.get("/nonexistent/file.py")
    
    assert result is None


def test_get_returns_cached_hash(cache):
    """Test that get() returns the cached hash for known files."""
    file_path = "/path/to/file.py"
    file_hash = "test_hash_123"

    cache.update(file_path, file_hash)

    result = cache.get(file_path)
    assert result == file_hash


def test_remove_deletes_entry(cache):
    """Test that remove() deletes an entry from the cache."""
    file_path = "/path/to/file.py"
    file_hash = "test_hash_456"

    cache.update(file_path, file_hash)
    assert cache.get(file_path) == file_hash

    cache.remove(file_path)
    assert cache.get(file_path) is None


def test_cache_persists_across_instances(temp_project_root, sample_file):
    """Test that cache persists when creating new cache instances."""
    # Create first cache and add entry
    cache1 = FileHashCache(temp_project_root)
    cache1.update(str(sample_file), "persist_test_hash")
    cache1.save()

    # Create second cache - should load existing data
    cache2 = FileHashCache(temp_project_root)
    
    assert cache2.get(str(sample_file)) == "persist_test_hash"


def test_cache_handles_missing_cache_file(temp_project_root):
    """Test that cache starts empty when cache file doesn't exist."""
    # Don't create any cache file
    cache = FileHashCache(temp_project_root)
    
    result = cache.get("/any/file.py")
    assert result is None


def test_atomic_write_creates_cache_file(cache, temp_project_root):
    """Test that atomic write creates the cache file correctly."""
    cache.update("/test/file.py", "atomic_hash")
    cache.save()

    cache_file = temp_project_root / ".code-reviewer" / "cache" / "hashes.json"
    assert cache_file.exists()
    
    # Verify no temp file left behind
    temp_file = cache_file.with_suffix(".json.tmp")
    assert not temp_file.exists()

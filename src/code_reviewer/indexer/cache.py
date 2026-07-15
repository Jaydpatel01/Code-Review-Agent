"""File hash cache for incremental indexing.

Tracks file hashes to avoid re-indexing unchanged files. The cache is stored
as JSON at {project_root}/.code-reviewer/cache/hashes.json and uses atomic
writes to prevent corruption during updates.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's contents.

    Args:
        file_path: Path to the file to hash

    Returns:
        SHA256 hexdigest string, or empty string on error
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash {file_path}: {e}")
        return ""


class FileHashCache:
    """In-memory cache of file hashes with JSON persistence.

    Cache location: {project_root}/.code-reviewer/cache/hashes.json

    The cache tracks SHA256 hashes of indexed files to enable incremental
    indexing. Files are only re-indexed when their hash changes.
    """

    def __init__(self, project_root: Path):
        """Initialize the hash cache.

        Args:
            project_root: Root directory of the project being indexed.
                         Cache will be created at {project_root}/.code-reviewer/cache/

        Note:
            Loads existing cache from disk if present. Creates empty cache
            if the file doesn't exist yet.
        """
        self.project_root = project_root
        self.cache_dir = project_root / ".code-reviewer" / "cache"
        self.cache_path = self.cache_dir / "hashes.json"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache or start with empty dict
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded cache with {len(self._cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache, starting fresh: {e}")
                self._cache = {}
        else:
            logger.info("No existing cache found, starting fresh")

    def get(self, file_path: str) -> str | None:
        """Retrieve cached hash for a file.

        Args:
            file_path: Path to the file (as string)

        Returns:
            SHA256 hash string if cached, None otherwise
        """
        return self._cache.get(file_path)

    def update(self, file_path: str, file_hash: str) -> None:
        """Update the in-memory cache with a new hash.

        Args:
            file_path: Path to the file (as string)
            file_hash: SHA256 hash of the file contents

        Note:
            This does NOT write to disk immediately. Call save() to persist.
        """
        self._cache[file_path] = file_hash

    def remove(self, file_path: str) -> None:
        """Remove a file from the cache.

        Args:
            file_path: Path to the file (as string)

        Note:
            This does NOT write to disk immediately. Call save() to persist.
        """
        self._cache.pop(file_path, None)

    def save(self) -> None:
        """Write the current cache to disk using atomic write.

        Uses a temporary file and rename to prevent corruption if the
        process is interrupted during the write.
        """
        try:
            # Write to temporary file first
            temp_path = self.cache_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)

            # Atomic rename
            temp_path.replace(self.cache_path)
            logger.info(f"Saved cache with {len(self._cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def is_changed(self, file_path: Path) -> tuple[bool, str]:
        """Check if a file has changed since last indexing.

        Args:
            file_path: Path to the file to check

        Returns:
            Tuple of (changed, hash) where:
            - changed is True if file is new or modified, False if unchanged
            - hash is the current SHA256 hash of the file

        Note:
            A file is considered changed if:
            1. It's not in the cache (new file)
            2. Its current hash differs from the cached hash (modified file)
        """
        current_hash = compute_file_hash(file_path)
        if not current_hash:
            # Error computing hash, treat as unchanged to avoid re-processing
            return False, ""

        file_path_str = str(file_path)
        cached_hash = self.get(file_path_str)

        if cached_hash is None:
            # New file, not in cache
            return True, current_hash

        if cached_hash != current_hash:
            # File modified since last index
            return True, current_hash

        # File unchanged
        return False, cached_hash

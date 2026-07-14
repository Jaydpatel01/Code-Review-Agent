"""File system walker for repo-level code review.

Discovers source files under a root directory, filters by include/exclude
patterns, skips binary files and files over the size limit, and respects
an optional maximum-file cap.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Generator

import typer

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE: list[str] = [
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".git",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "*.egg-info",
]

_MAX_FILE_SIZE_BYTES: int = 512_000   # 500 KiB
_BINARY_PROBE_BYTES: int = 8_192      # bytes to read for binary detection


class FileWalker:
    """Recursively walk a directory tree and yield reviewable source files.

    Parameters
    ----------
    root:
        Directory to search.
    include:
        Glob patterns matched against the *filename* (not the full path).
        Defaults to ``["*.py"]``.
    exclude:
        Directory/file name patterns to skip.  Merged with
        ``_DEFAULT_EXCLUDE`` only when *exclude* is ``None``; an explicit
        empty list disables all default exclusions.
    max_files:
        Maximum number of files to yield.  ``0`` means unlimited.
    """

    def __init__(
        self,
        root: str | Path,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_files: int = 0,
    ) -> None:
        """Initialise the walker with root directory and filter settings."""
        self.root = Path(root)
        self.include: list[str] = include if include is not None else ["*.py"]
        self.exclude: list[str] = exclude if exclude is not None else list(_DEFAULT_EXCLUDE)
        self.max_files: int = max_files

        # FIX 5: Pre-compute exact vs glob exclude sets for O(1) common-case lookup
        self._exclude_exact: set[str] = {
            p for p in self.exclude if "*" not in p and "?" not in p
        }
        self._exclude_glob: list[str] = [
            p for p in self.exclude if "*" in p or "?" in p
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, path: Path) -> bool:
        """Return True if any path component matches an exclude pattern.

        Uses an exact-name set for O(1) lookup on the common case (e.g.
        ``.venv``, ``__pycache__``), then falls back to fnmatch only for
        wildcard patterns.
        """
        for part in path.parts:
            # O(1) set lookup for exact patterns (the common case)
            if part in self._exclude_exact:
                return True
            # O(m) fnmatch only for wildcard patterns
            for pattern in self._exclude_glob:
                if fnmatch(part, pattern):
                    return True
        return False

    def _matches_include(self, path: Path) -> bool:
        """Return True if the filename matches any include pattern."""
        name = path.name
        return any(fnmatch(name, pat) for pat in self.include)

    def _is_binary(self, path: Path) -> bool:
        """Return True if the file appears to be binary (non-UTF-8)."""
        try:
            with path.open("rb") as fh:
                chunk = fh.read(_BINARY_PROBE_BYTES)
            chunk.decode("utf-8")
            return False
        except (UnicodeDecodeError, OSError):
            return True

    def _is_oversized(self, path: Path) -> bool:
        """Return True if the file exceeds the maximum reviewable size."""
        try:
            return path.stat().st_size > _MAX_FILE_SIZE_BYTES
        except OSError:
            return True

    def _candidate_files(self) -> Generator[Path, None, None]:
        """Yield all files that pass include/exclude/binary/size checks."""
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if self._is_excluded(path):
                continue
            if not self._matches_include(path):
                continue
            if self._is_oversized(path):
                continue
            if self._is_binary(path):
                continue
            yield path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def walk(self) -> Generator[Path, None, None]:
        """Yield matching file Paths up to the ``max_files`` limit.

        FIX 4: The file that triggers the limit is YIELDED first, then
        iteration stops.  The previous implementation checked the limit
        BEFORE yielding, causing one file to be silently skipped.

        Set ``max_files=0`` for unlimited traversal.
        """
        count = 0
        for path in self._candidate_files():
            yield path
            count += 1
            if self.max_files > 0 and count >= self.max_files:
                remaining = self.count() - count
                if remaining > 0:
                    typer.echo(
                        f"[WARN] Stopped at {self.max_files} files. "
                        f"{remaining} more files exist. "
                        f"Use --max-files 0 to review all."
                    )
                break

    def count(self) -> int:
        """Return the total number of candidate files (ignores max_files)."""
        return sum(1 for _ in self._candidate_files())

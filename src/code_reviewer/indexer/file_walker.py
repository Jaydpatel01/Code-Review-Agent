"""File walker for recursively discovering source files in a repository.

Supports include/exclude glob patterns, binary-file detection, size limits,
and a max-files safety cap to prevent runaway reviews on huge codebases.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Generator


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

_MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB
_BINARY_PROBE_BYTES = 512


class FileWalker:
    """Recursively walks a directory tree yielding source files.

    Filters are applied in this order:
      1. Exclude: any path component matching an exclude pattern is skipped.
      2. Include: only files matching at least one include glob are yielded.
      3. Size:    files larger than 500 KB are skipped.
      4. Binary:  files whose first 512 bytes cannot be decoded as UTF-8 are skipped.
      5. Limit:   at most ``max_files`` files are yielded (0 = unlimited).

    Args:
        root:      Root directory to walk.
        include:   List of glob patterns that file *names* must match (e.g. ``["*.py"]``).
        exclude:   List of path-component patterns to skip (e.g. ``[".venv", "__pycache__"]``).
        max_files: Maximum number of files to yield. 0 disables the limit.
    """

    def __init__(
        self,
        root: Path,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_files: int = 50,
    ) -> None:
        self.root = Path(root)
        self.include: list[str] = include if include is not None else ["*.py"]
        self.exclude: list[str] = exclude if exclude is not None else list(_DEFAULT_EXCLUDE)
        self.max_files = max_files

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, path: Path) -> bool:
        """Return True if any path component matches an exclude pattern."""
        for part in path.parts:
            for pattern in self.exclude:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    def _matches_include(self, path: Path) -> bool:
        """Return True if the file name matches at least one include pattern."""
        name = path.name
        return any(fnmatch.fnmatch(name, pat) for pat in self.include)

    def _is_binary(self, path: Path) -> bool:
        """Return True if the file cannot be decoded as UTF-8 (binary heuristic)."""
        try:
            with path.open("rb") as fh:
                chunk = fh.read(_BINARY_PROBE_BYTES)
            chunk.decode("utf-8")
            return False
        except (UnicodeDecodeError, OSError):
            return True

    def _is_oversized(self, path: Path) -> bool:
        """Return True if the file exceeds the 500 KB size limit."""
        try:
            return path.stat().st_size > _MAX_FILE_SIZE_BYTES
        except OSError:
            return True

    def _candidate_files(self) -> Generator[Path, None, None]:
        """Yield every file that passes all filters (no max_files cap applied)."""
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

        If the limit is reached, a warning is printed and iteration stops.
        Set ``max_files=0`` for unlimited traversal.
        """
        count = 0
        for path in self._candidate_files():
            if self.max_files > 0 and count >= self.max_files:
                print(
                    f"[WARN] max_files limit ({self.max_files}) reached. "
                    "Use --max-files 0 for unlimited."
                )
                return
            yield path
            count += 1

    def count(self) -> int:
        """Return the total number of matching files without applying the max_files cap.

        Use this to show "X files found" before walking.
        """
        return sum(1 for _ in self._candidate_files())

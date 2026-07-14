"""Unit tests for FileWalker.

Tests cover:
  - walk() max_files boundary: the file that triggers the limit is included
  - _is_excluded() exact-match patterns (.venv, __pycache__)
  - _is_excluded() wildcard patterns (*.egg-info)
  - _is_excluded() returns False for valid paths
  - walk() returns all files when max_files=0 (unlimited)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_reviewer.indexer.file_walker import FileWalker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_py_tree(tmp_path: Path) -> Path:
    """Create a small temp directory tree with Python source files.

    Layout:
        tmp_path/
            a.py
            b.py
            c.py
            d.py
            .venv/
                lib.py          <- should be excluded
            __pycache__/
                x.pyc           <- extension won't match *.py; also excluded dir
            valid_module/
                main.py
    """
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("x = 3\n", encoding="utf-8")
    (tmp_path / "d.py").write_text("x = 4\n", encoding="utf-8")

    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("# venv file\n", encoding="utf-8")

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    # .pyc won't match *.py pattern but the directory is also excluded
    (cache / "x.pyc").write_text("", encoding="utf-8")

    mod = tmp_path / "valid_module"
    mod.mkdir()
    (mod / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: walk() max_files boundary (FIX 4)
# ---------------------------------------------------------------------------

class TestWalkMaxFiles:
    """The file that triggers the max_files limit must be yielded (not skipped)."""

    def test_walk_max_files_3_yields_exactly_3(self, tmp_py_tree: Path) -> None:
        """walk(max_files=3) must yield exactly 3 files, not 2."""
        walker = FileWalker(tmp_py_tree, include=["*.py"], max_files=3)
        result = list(walker.walk())
        assert len(result) == 3, (
            f"Expected exactly 3 files, got {len(result)}: {result}"
        )

    def test_walk_max_files_1_yields_exactly_1(self, tmp_py_tree: Path) -> None:
        """Edge case: max_files=1 must yield exactly 1 file."""
        walker = FileWalker(tmp_py_tree, include=["*.py"], max_files=1)
        result = list(walker.walk())
        assert len(result) == 1

    def test_walk_unlimited_yields_all_non_excluded_py_files(
        self, tmp_py_tree: Path
    ) -> None:
        """max_files=0 means unlimited; should yield all *.py not in excluded dirs."""
        walker = FileWalker(tmp_py_tree, include=["*.py"], max_files=0)
        result = list(walker.walk())
        names = {p.name for p in result}
        # a.py, b.py, c.py, d.py, valid_module/main.py — NOT .venv/lib.py
        assert "a.py" in names
        assert "b.py" in names
        assert "c.py" in names
        assert "d.py" in names
        assert "main.py" in names
        assert "lib.py" not in names, ".venv/lib.py must be excluded"


# ---------------------------------------------------------------------------
# Tests: _is_excluded (FIX 5)
# ---------------------------------------------------------------------------

class TestIsExcluded:
    """_is_excluded must use the pre-computed exact set for common names."""

    def _make_walker(self, root: Path) -> FileWalker:
        return FileWalker(root, include=["*.py"])

    def test_venv_directory_is_excluded(self, tmp_py_tree: Path) -> None:
        """Paths inside .venv must be excluded via exact match."""
        walker = self._make_walker(tmp_py_tree)
        excluded_path = tmp_py_tree / ".venv" / "lib.py"
        assert walker._is_excluded(excluded_path) is True

    def test_pycache_directory_is_excluded(self, tmp_py_tree: Path) -> None:
        """Paths inside __pycache__ must be excluded via exact match."""
        walker = self._make_walker(tmp_py_tree)
        excluded_path = tmp_py_tree / "__pycache__" / "x.pyc"
        assert walker._is_excluded(excluded_path) is True

    def test_valid_module_not_excluded(self, tmp_py_tree: Path) -> None:
        """Paths inside a normal module directory must NOT be excluded."""
        walker = self._make_walker(tmp_py_tree)
        valid_path = tmp_py_tree / "valid_module" / "main.py"
        assert walker._is_excluded(valid_path) is False

    def test_wildcard_egg_info_excluded(self, tmp_py_tree: Path) -> None:
        """*.egg-info is a wildcard pattern and must still be excluded."""
        walker = self._make_walker(tmp_py_tree)
        egg_path = tmp_py_tree / "mypackage.egg-info" / "PKG-INFO"
        assert walker._is_excluded(egg_path) is True

    def test_exact_set_populated_correctly(self, tmp_py_tree: Path) -> None:
        """Pre-computed _exclude_exact set must contain all non-wildcard patterns."""
        walker = self._make_walker(tmp_py_tree)
        assert ".venv" in walker._exclude_exact
        assert "__pycache__" in walker._exclude_exact
        assert ".git" in walker._exclude_exact
        # wildcard patterns must NOT be in the exact set
        assert "*.egg-info" not in walker._exclude_exact

    def test_glob_list_populated_correctly(self, tmp_py_tree: Path) -> None:
        """Pre-computed _exclude_glob must contain only wildcard patterns."""
        walker = self._make_walker(tmp_py_tree)
        assert "*.egg-info" in walker._exclude_glob
        # exact patterns must NOT be in the glob list
        assert ".venv" not in walker._exclude_glob

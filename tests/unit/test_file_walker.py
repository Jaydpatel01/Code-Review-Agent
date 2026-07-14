"""Unit tests for FileWalker (src/code_reviewer/indexer/file_walker.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_reviewer.indexer.file_walker import FileWalker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_tree(base: Path, structure: dict) -> None:
    """Recursively create a directory tree from a dict spec.

    Keys are path components; values are either dicts (directories) or str
    (file content).  Example::

        {"src": {"main.py": "x=1", "__pycache__": {"cache.pyc": ""}}}
    """
    for name, content in structure.items():
        target = base / name
        if isinstance(content, dict):
            target.mkdir(parents=True, exist_ok=True)
            _create_tree(target, content)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileWalkerInclude:
    def test_yields_only_py_files(self, tmp_path):
        """walk() only yields .py files when include=['*.py']."""
        _create_tree(tmp_path, {
            "a.py": "x = 1",
            "b.txt": "text",
            "c.js": "var x = 1;",
            "sub": {"d.py": "y = 2"},
        })
        walker = FileWalker(tmp_path, include=["*.py"])
        results = list(walker.walk())
        names = {p.name for p in results}
        assert "a.py" in names
        assert "d.py" in names
        assert "b.txt" not in names
        assert "c.js" not in names

    def test_multiple_include_patterns(self, tmp_path):
        """walk() yields files matching any of multiple include patterns."""
        _create_tree(tmp_path, {
            "a.py": "x = 1",
            "b.js": "var x;",
            "c.txt": "text",
        })
        walker = FileWalker(tmp_path, include=["*.py", "*.js"])
        names = {p.name for p in walker.walk()}
        assert "a.py" in names
        assert "b.js" in names
        assert "c.txt" not in names


class TestFileWalkerExclude:
    def test_skips_pycache(self, tmp_path):
        """walk() skips files inside __pycache__."""
        _create_tree(tmp_path, {
            "ok.py": "x = 1",
            "__pycache__": {"compiled.py": "# compiled"},
        })
        walker = FileWalker(tmp_path)
        names = {p.name for p in walker.walk()}
        assert "ok.py" in names
        assert "compiled.py" not in names

    def test_skips_venv_directory(self, tmp_path):
        """walk() skips files inside .venv."""
        _create_tree(tmp_path, {
            "app.py": "x = 1",
            ".venv": {"lib.py": "# venv lib"},
        })
        walker = FileWalker(tmp_path)
        names = {p.name for p in walker.walk()}
        assert "app.py" in names
        assert "lib.py" not in names

    def test_custom_exclude_pattern(self, tmp_path):
        """walk() respects custom exclude patterns."""
        _create_tree(tmp_path, {
            "src": {"main.py": "x = 1"},
            "vendor": {"third_party.py": "# vendor"},
        })
        walker = FileWalker(tmp_path, exclude=["vendor"])
        names = {p.name for p in walker.walk()}
        assert "main.py" in names
        assert "third_party.py" not in names

    def test_skips_node_modules(self, tmp_path):
        """walk() skips files inside node_modules."""
        _create_tree(tmp_path, {
            "index.py": "x = 1",
            "node_modules": {"pkg.py": "# npm"},
        })
        walker = FileWalker(tmp_path)
        names = {p.name for p in walker.walk()}
        assert "index.py" in names
        assert "pkg.py" not in names


class TestFileWalkerMaxFiles:
    def test_stops_at_max_files(self, tmp_path, capsys):
        """walk() stops after max_files and prints a warning."""
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}", encoding="utf-8")

        walker = FileWalker(tmp_path, max_files=3)
        results = list(walker.walk())

        assert len(results) == 3
        captured = capsys.readouterr()
        assert "max_files limit (3) reached" in captured.out

    def test_max_files_zero_returns_all(self, tmp_path):
        """walk() with max_files=0 returns all matching files."""
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}", encoding="utf-8")

        walker = FileWalker(tmp_path, max_files=0)
        results = list(walker.walk())
        assert len(results) == 10

    def test_no_warning_when_under_limit(self, tmp_path, capsys):
        """walk() does not print a warning when file count is within limit."""
        (tmp_path / "only.py").write_text("x = 1", encoding="utf-8")
        walker = FileWalker(tmp_path, max_files=10)
        list(walker.walk())
        captured = capsys.readouterr()
        assert "max_files" not in captured.out


class TestFileWalkerCount:
    def test_count_returns_correct_total(self, tmp_path):
        """count() returns the total number of matching files ignoring max_files."""
        for i in range(7):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}", encoding="utf-8")

        walker = FileWalker(tmp_path, max_files=3)
        assert walker.count() == 7

    def test_count_excludes_non_matching(self, tmp_path):
        """count() only counts files matching the include patterns."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.txt").write_text("text", encoding="utf-8")
        (tmp_path / "c.js").write_text("var x;", encoding="utf-8")

        walker = FileWalker(tmp_path, include=["*.py"])
        assert walker.count() == 1


class TestFileWalkerEdgeCases:
    def test_skips_oversized_file(self, tmp_path):
        """walk() skips files larger than 500 KB."""
        big = tmp_path / "big.py"
        big.write_bytes(b"x = 1\n" * 100_000)  # ~600 KB
        small = tmp_path / "small.py"
        small.write_text("x = 1", encoding="utf-8")

        walker = FileWalker(tmp_path)
        names = {p.name for p in walker.walk()}
        assert "small.py" in names
        assert "big.py" not in names

    def test_skips_binary_file(self, tmp_path):
        """walk() skips files with non-UTF-8 content."""
        binary = tmp_path / "binary.py"
        binary.write_bytes(b"\x00\xFF\xFE invalid utf8 \x80\x81")
        readable = tmp_path / "readable.py"
        readable.write_text("x = 1", encoding="utf-8")

        walker = FileWalker(tmp_path)
        names = {p.name for p in walker.walk()}
        assert "readable.py" in names
        assert "binary.py" not in names

    def test_empty_directory_yields_nothing(self, tmp_path):
        """walk() on an empty directory yields no files."""
        walker = FileWalker(tmp_path)
        assert list(walker.walk()) == []

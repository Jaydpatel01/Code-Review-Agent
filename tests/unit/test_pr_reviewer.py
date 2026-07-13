"""Unit tests for PRReviewer._build_diff_position_map."""

import pytest
from unittest.mock import MagicMock

from code_reviewer.core.pr_reviewer import PRReviewer


def _make_reviewer() -> PRReviewer:
    """Return a PRReviewer with mocked dependencies."""
    return PRReviewer(
        github_client=MagicMock(),
        llm_client=MagicMock(),
        settings=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Canonical diff fixture
# ---------------------------------------------------------------------------
#
# Diff breakdown for single-file case (file: src/foo.py):
#
#   @@ -1,3 +1,4 @@          → position 1  (hunk header)
#    def foo():               → position 2  (context, new_line=1)
#   +    x = 1               → position 3  (added, new_line=2)  ← mapped
#   -    y = 2               → position 4  (removed)            ← NOT mapped
#    return x                → position 5  (context, new_line=3)
#
SINGLE_FILE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def foo():
+    x = 1
-    y = 2
 return x
"""

# Two-file diff to verify position counter resets per file.
#
# File 1 (main.py):
#   @@ -1,1 +1,2 @@          → pos 1
#   +def init():             → pos 2  (new_line=1) ← mapped
#   +    pass                → pos 3  (new_line=2) ← mapped
#
# File 2 (utils.py)  — position MUST reset to 0:
#   @@ -5,2 +5,3 @@          → pos 1
#    a = 1                   → pos 2  (context, new_line=5)
#   +    b = 2               → pos 3  (new_line=6) ← mapped
#
MULTI_FILE_DIFF = """\
diff --git a/main.py b/main.py
--- a/main.py
+++ b/main.py
@@ -1,1 +1,2 @@
+def init():
+    pass
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -5,2 +5,3 @@
 a = 1
+    b = 2
"""


class TestBuildDiffPositionMap:
    """Unit tests for the _build_diff_position_map private method."""

    def setup_method(self):
        self.reviewer = _make_reviewer()

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_hunk_header_counts_as_position_1(self):
        """The @@ hunk header must be position 1 for every file."""
        result = self.reviewer._build_diff_position_map(SINGLE_FILE_DIFF)
        # The only added line is at new file line 2 → position 3
        assert result["src/foo.py"][2] == 3

    def test_added_lines_are_mapped(self):
        """Every + line must appear in the output map."""
        result = self.reviewer._build_diff_position_map(SINGLE_FILE_DIFF)
        assert 2 in result["src/foo.py"]

    def test_removed_lines_are_not_mapped(self):
        """- lines must NOT appear in the output map (but they do increment position)."""
        result = self.reviewer._build_diff_position_map(SINGLE_FILE_DIFF)
        # The removed line (y = 2) has no entry; only new-file line numbers are keys
        # The context line at new_line=1 is also not a + so it won't be in map
        # Only new_line=2 (the added x=1) should be mapped
        assert list(result["src/foo.py"].keys()) == [2]

    def test_context_lines_not_mapped(self):
        """Context lines (space-prefixed) must not appear in the output map."""
        result = self.reviewer._build_diff_position_map(SINGLE_FILE_DIFF)
        # new_line 1 = context 'def foo():' — should NOT be in map
        assert 1 not in result["src/foo.py"]
        # new_line 3 = context 'return x' — should NOT be in map
        assert 3 not in result["src/foo.py"]

    # ------------------------------------------------------------------
    # Position counter behaviour
    # ------------------------------------------------------------------

    def test_position_increments_through_context_and_removed(self):
        """
        The position counter must increment for ALL line types (context, +, -).
        Verified by the exact position of the added line.

        Layout:
          pos 1 → @@ header
          pos 2 → context  (def foo():)
          pos 3 → added    (x = 1)       ← 3, not 2
          pos 4 → removed  (y = 2)
          pos 5 → context  (return x)
        """
        result = self.reviewer._build_diff_position_map(SINGLE_FILE_DIFF)
        assert result["src/foo.py"][2] == 3  # position 3, not 2

    # ------------------------------------------------------------------
    # Multi-file: position resets per file
    # ------------------------------------------------------------------

    def test_position_resets_per_file(self):
        """
        Position counter must reset to 0 at each 'diff --git' boundary.
        Both files should independently start at position 1 for their @@ header.
        """
        result = self.reviewer._build_diff_position_map(MULTI_FILE_DIFF)

        # main.py: @@ = pos 1, first + = pos 2, second + = pos 3
        assert result["main.py"][1] == 2
        assert result["main.py"][2] == 3

        # utils.py: @@ = pos 1, context = pos 2, added b=2 = pos 3
        assert result["utils.py"][6] == 3

    def test_both_files_present_in_result(self):
        """Both files from the multi-file diff must appear as keys."""
        result = self.reviewer._build_diff_position_map(MULTI_FILE_DIFF)
        assert "main.py" in result
        assert "utils.py" in result

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_diff_returns_empty_map(self):
        """An empty diff string must produce an empty dict."""
        result = self.reviewer._build_diff_position_map("")
        assert result == {}

    def test_deleted_file_not_in_map(self):
        """Files deleted in the diff ('+++ /dev/null') must not appear."""
        deleted_diff = """\
diff --git a/old.py b/old.py
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old():
-    pass
"""
        result = self.reviewer._build_diff_position_map(deleted_diff)
        assert "old.py" not in result
        assert result == {}

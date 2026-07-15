"""Parser for Git unified diffs."""

import re
from typing import List, Optional
from code_reviewer.core.models import DiffHunk

# Compiled once at import time — matches @@ -old,len +new,len @@ headers.
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _flush_hunk(
    current_hunk: Optional[DiffHunk],
    raw_hunk_lines: List[str],
    hunks: List[DiffHunk],
) -> None:
    """Finalise a hunk and append it to the result list.

    Args:
        current_hunk: The hunk being built, or None if there is none.
        raw_hunk_lines: Lines accumulated for the current hunk's raw text.
        hunks: The list to which the completed hunk is appended.
    """
    if current_hunk:
        current_hunk.raw_hunk = "\n".join(raw_hunk_lines)
        hunks.append(current_hunk)


def _parse_file_path(line: str) -> Optional[str]:
    """Extract the file path from a ``+++ `` header line.

    Returns None for deleted files (``+++ /dev/null``).

    Args:
        line: A ``+++ `` prefixed diff header line.

    Returns:
        File path string, or None if the file was deleted.
    """
    if line.startswith("+++ b/"):
        return line[6:]
    if line.startswith("+++ /dev/null"):
        return None          # Deleted file — skip
    return line[4:]          # Non-standard prefix (e.g. bare path)


def _open_hunk(line: str, current_file: str) -> Optional[tuple]:
    """Parse a hunk header and return the new DiffHunk plus tracking state.

    Args:
        line: The ``@@ … @@`` hunk header line.
        current_file: The file path this hunk belongs to.

    Returns:
        (DiffHunk, new_start, old_start, raw_lines) tuple, or None if the
        header does not match the expected regex.
    """
    match = _HUNK_HEADER_RE.search(line)
    if not match:
        return None

    old_start = int(match.group(1))
    new_start = int(match.group(3))
    new_len = int(match.group(4)) if match.group(4) is not None else 1

    hunk = DiffHunk(
        file_path=current_file,
        start_line=new_start,
        end_line=new_start + new_len - 1,
        added_lines=[],
        removed_lines=[],
        context_lines=[],
        raw_hunk="",
    )
    return hunk, new_start, old_start, [line]


def parse_diff(diff_text: str) -> List[DiffHunk]:
    """
    Parse a unified git diff into a list of structured DiffHunks.

    Args:
        diff_text: Raw unified diff output from git

    Returns:
        List[DiffHunk]: Extracted hunks with mapped line numbers.
    """
    hunks: List[DiffHunk] = []

    current_file: Optional[str] = None
    current_hunk: Optional[DiffHunk] = None
    raw_hunk_lines: List[str] = []
    current_new_line = 0
    current_old_line = 0

    for line in diff_text.splitlines():

        # --- New file boundary ---
        if line.startswith("diff --git"):
            _flush_hunk(current_hunk, raw_hunk_lines, hunks)
            current_hunk = None
            current_file = None
            continue

        # --- File header lines ---
        if line.startswith("+++ "):
            _flush_hunk(current_hunk, raw_hunk_lines, hunks)
            current_hunk = None
            current_file = _parse_file_path(line)
            continue

        if line.startswith("--- "):
            continue

        # --- Hunk header ---
        if line.startswith("@@ ") and current_file:
            _flush_hunk(current_hunk, raw_hunk_lines, hunks)
            result = _open_hunk(line, current_file)
            if result is None:
                current_hunk = None
                continue
            current_hunk, current_new_line, current_old_line, raw_hunk_lines = result
            continue

        # --- Diff content lines ---
        if current_hunk:
            raw_hunk_lines.append(line)

            if line.startswith("+"):
                current_hunk.added_lines.append((current_new_line, line[1:]))
                current_new_line += 1
            elif line.startswith("-"):
                current_hunk.removed_lines.append((current_old_line, line[1:]))
                current_old_line += 1
            elif line.startswith(" "):
                current_hunk.context_lines.append((current_new_line, line[1:]))
                current_new_line += 1
                current_old_line += 1
            # Lines like `\ No newline at end of file` are naturally ignored.

    _flush_hunk(current_hunk, raw_hunk_lines, hunks)
    return hunks

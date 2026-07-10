"""Parser for Git unified diffs."""

import re
from typing import List
from code_reviewer.core.models import DiffHunk


def parse_diff(diff_text: str) -> List[DiffHunk]:
    """
    Parse a unified git diff into a list of structured DiffHunks.
    
    Args:
        diff_text: Raw unified diff output from git
        
    Returns:
        List[DiffHunk]: Extracted hunks with mapped line numbers.
    """
    hunks: List[DiffHunk] = []
    
    current_file = None
    current_hunk = None
    raw_hunk_lines = []
    
    current_new_line = 0
    current_old_line = 0
    
    # Regex to match hunk headers like @@ -old,len +new,len @@
    hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    
    lines = diff_text.splitlines()
    
    for line in lines:
        # Stop processing current file/hunk if a new file starts
        if line.startswith("diff --git"):
            if current_hunk:
                current_hunk.raw_hunk = "\n".join(raw_hunk_lines)
                hunks.append(current_hunk)
                current_hunk = None
            current_file = None
            continue
            
        if line.startswith("+++ "):
            # Close previous hunk if we hit another file marker unexpectedly
            if current_hunk:
                current_hunk.raw_hunk = "\n".join(raw_hunk_lines)
                hunks.append(current_hunk)
                current_hunk = None
                
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("+++ /dev/null"):
                current_file = None  # Deleted file; we don't review these
            else:
                current_file = line[4:]  # Handle non-standard prefixes
            continue
            
        if line.startswith("--- "):
            continue
            
        if line.startswith("@@ ") and current_file:
            if current_hunk:
                current_hunk.raw_hunk = "\n".join(raw_hunk_lines)
                hunks.append(current_hunk)
            
            match = hunk_header_re.search(line)
            if not match:
                continue
                
            old_start = int(match.group(1))
            new_start = int(match.group(3))
            
            # If length is omitted in diff (e.g., @@ -1 +1 @@), it means a length of 1
            new_len = int(match.group(4)) if match.group(4) is not None else 1
            
            current_old_line = old_start
            current_new_line = new_start
            
            raw_hunk_lines = [line]
            
            current_hunk = DiffHunk(
                file_path=current_file,
                start_line=new_start,
                end_line=new_start + new_len - 1 if new_len > 0 else new_start,
                added_lines=[],
                removed_lines=[],
                context_lines=[],
                raw_hunk=""
            )
            continue
            
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
            # We naturally ignore `\ No newline at end of file` and 
            # other non-code additions since they don't start with space, +, or -.
            
    if current_hunk:
        current_hunk.raw_hunk = "\n".join(raw_hunk_lines)
        hunks.append(current_hunk)
        
    return hunks

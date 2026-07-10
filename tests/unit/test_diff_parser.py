import os
from code_reviewer.analyzers.diff_parser import parse_diff

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_diffs")

def read_fixture(filename: str) -> str:
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def test_parse_simple_change():
    diff_text = read_fixture("simple_change.diff")
    hunks = parse_diff(diff_text)
    
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.file_path == "app.py"
    assert hunk.start_line == 10
    assert hunk.end_line == 13
    
    assert len(hunk.added_lines) == 2
    assert hunk.added_lines[0] == (11, '    print("World")')
    assert hunk.added_lines[1] == (12, '    return 1')
    
    assert len(hunk.removed_lines) == 1
    assert hunk.removed_lines[0] == (11, '    return 0')
    
    assert len(hunk.context_lines) == 1
    assert hunk.context_lines[0] == (10, '    print("Hello")')

def test_parse_multi_file():
    diff_text = read_fixture("multi_file.diff")
    hunks = parse_diff(diff_text)
    
    assert len(hunks) == 2
    
    hunk1 = hunks[0]
    assert hunk1.file_path == "main.py"
    assert hunk1.start_line == 1
    assert len(hunk1.added_lines) == 2
    assert hunk1.added_lines[0] == (1, 'def init():')
    assert hunk1.added_lines[1] == (2, '    pass')
    assert len(hunk1.removed_lines) == 1
    
    hunk2 = hunks[1]
    assert hunk2.file_path == "utils.py"
    assert hunk2.start_line == 5
    assert len(hunk2.added_lines) == 1
    assert hunk2.added_lines[0] == (6, '    b = 2')

def test_parse_new_file():
    diff_text = read_fixture("new_file.diff")
    hunks = parse_diff(diff_text)
    
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.file_path == "new_feature.py"
    assert hunk.start_line == 1
    assert hunk.end_line == 2
    assert len(hunk.added_lines) == 2
    assert hunk.added_lines[0] == (1, 'def setup():')
    assert hunk.added_lines[1] == (2, '    return True')
    assert len(hunk.removed_lines) == 0

"""Unit tests for DependencyGraph."""

import pytest
from code_reviewer.indexer.dependency_graph import DependencyGraph
from code_reviewer.indexer.chunker import CodeChunk


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunk_a = CodeChunk(
        chunk_id="a_func_a",
        file_path="a.py",
        name="func_a",
        chunk_type="function",
        start_line=1,
        end_line=5,
        source_code="def func_a():\n    func_b()\n",
        docstring=None,
        calls=["func_b"],
        complexity=1,
        file_hash="hash_a",
    )
    
    chunk_b = CodeChunk(
        chunk_id="b_func_b",
        file_path="b.py",
        name="func_b",
        chunk_type="function",
        start_line=10,
        end_line=15,
        source_code="def func_b():\n    func_c()\n",
        docstring=None,
        calls=["func_c"],
        complexity=1,
        file_hash="hash_b",
    )
    
    chunk_c = CodeChunk(
        chunk_id="c_func_c",
        file_path="c.py",
        name="func_c",
        chunk_type="function",
        start_line=20,
        end_line=25,
        source_code="def func_c():\n    pass\n",
        docstring=None,
        calls=[],
        complexity=1,
        file_hash="hash_c",
    )
    
    return [chunk_a, chunk_b, chunk_c]


def test_graph_build(sample_chunks):
    """Test building a graph from chunks."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_edges() == 2  # A→B, B→C


def test_get_callees(sample_chunks):
    """Test getting callees (successors)."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    callees_a = graph.get_callees("a.py::func_a")
    assert callees_a == ["b.py::func_b"]
    
    callees_b = graph.get_callees("b.py::func_b")
    assert callees_b == ["c.py::func_c"]
    
    callees_c = graph.get_callees("c.py::func_c")
    assert callees_c == []


def test_get_callers(sample_chunks):
    """Test getting callers (predecessors)."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    callers_a = graph.get_callers("a.py::func_a")
    assert callers_a == []
    
    callers_b = graph.get_callers("b.py::func_b")
    assert callers_b == ["a.py::func_a"]
    
    callers_c = graph.get_callers("c.py::func_c")
    assert callers_c == ["b.py::func_b"]


def test_get_call_chain(sample_chunks):
    """Test getting call chains via BFS."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    chains = graph.get_call_chain("a.py::func_a", max_depth=3)
    
    # Should find the chain A→B→C
    assert len(chains) == 1
    assert chains[0] == ["a.py::func_a", "b.py::func_b", "c.py::func_c"]


def test_get_centrality_scores(sample_chunks):
    """Test PageRank centrality scoring."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    scores = graph.get_centrality_scores()
    
    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores.values())
    
    # All nodes should have some centrality score
    assert scores["a.py::func_a"] > 0
    assert scores["b.py::func_b"] > 0
    assert scores["c.py::func_c"] > 0


def test_get_centrality_score_single(sample_chunks):
    """Test getting centrality for a single node."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    score_b = graph.get_centrality_score("b.py::func_b")
    assert 0 <= score_b <= 1
    
    score_nonexistent = graph.get_centrality_score("nonexistent.py::func")
    assert score_nonexistent == 0.0


def test_find_node_key(sample_chunks):
    """Test finding node key by file and function name."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    key = graph.find_node_key("a.py", "func_a")
    assert key == "a.py::func_a"
    
    key_none = graph.find_node_key("nonexistent.py", "func")
    assert key_none is None


def test_external_calls():
    """Test handling of external calls (stdlib/third-party)."""
    chunk = CodeChunk(
        chunk_id="test_my_func",
        file_path="test.py",
        name="my_func",
        chunk_type="function",
        start_line=1,
        end_line=5,
        source_code="def my_func():\n    print('hello')\n",
        docstring=None,
        calls=["print", "len"],  # External calls
        complexity=1,
        file_hash="hash",
    )
    
    graph = DependencyGraph()
    graph.build([chunk])
    
    # Should have 3 nodes: my_func + 2 external nodes
    assert graph.graph.number_of_nodes() == 3
    
    # Should have 2 edges: my_func→external::print, my_func→external::len
    assert graph.graph.number_of_edges() == 2
    
    callees = graph.get_callees("test.py::my_func")
    assert "external::print" in callees
    assert "external::len" in callees


def test_empty_graph():
    """Test operations on an empty graph."""
    graph = DependencyGraph()
    graph.build([])
    
    assert graph.graph.number_of_nodes() == 0
    assert graph.graph.number_of_edges() == 0
    assert graph.get_centrality_scores() == {}
    assert graph.get_callees("any::node") == []
    assert graph.get_callers("any::node") == []


def test_call_chain_with_cycle():
    """Test call chain detection with cycles (should avoid infinite loops)."""
    # Create a cycle: A→B→A
    chunk_a = CodeChunk(
        chunk_id="a_func_a",
        file_path="a.py",
        name="func_a",
        chunk_type="function",
        start_line=1,
        end_line=5,
        source_code="def func_a():\n    func_b()\n",
        docstring=None,
        calls=["func_b"],
        complexity=1,
        file_hash="hash_a",
    )
    
    chunk_b = CodeChunk(
        chunk_id="b_func_b",
        file_path="b.py",
        name="func_b",
        chunk_type="function",
        start_line=10,
        end_line=15,
        source_code="def func_b():\n    func_a()\n",  # Cycle back to A
        docstring=None,
        calls=["func_a"],
        complexity=1,
        file_hash="hash_b",
    )
    
    graph = DependencyGraph()
    graph.build([chunk_a, chunk_b])
    
    # Should not infinite loop - cycle detection prevents revisiting
    chains = graph.get_call_chain("a.py::func_a", max_depth=3)
    
    # With a cycle, BFS will find paths but stop when it encounters
    # nodes already in the current path
    # The implementation should handle this gracefully
    assert isinstance(chains, list)


def test_centrality_caching(sample_chunks):
    """Test that centrality scores are cached."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    # First call computes
    scores1 = graph.get_centrality_scores()
    
    # Second call should return cached result
    scores2 = graph.get_centrality_scores()
    
    assert scores1 is scores2  # Same object reference


def test_rebuild_clears_cache(sample_chunks):
    """Test that rebuild() clears the centrality cache."""
    graph = DependencyGraph()
    graph.build(sample_chunks)
    
    scores1 = graph.get_centrality_scores()
    
    # Rebuild with same chunks
    graph.build(sample_chunks)
    
    scores2 = graph.get_centrality_scores()
    
    # Should be different objects (cache was cleared)
    assert scores1 is not scores2

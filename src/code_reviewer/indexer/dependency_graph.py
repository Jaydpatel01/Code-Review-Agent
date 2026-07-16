"""Dependency graph builder for call chains and import relationships.

Uses NetworkX to build a directed graph of function calls and dependencies
across the codebase. Enables centrality scoring, call chain analysis, and
cross-file root cause tracking.
"""

import logging
from collections import deque
from typing import Optional

import networkx as nx

from code_reviewer.indexer.chunker import CodeChunk

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Builds and queries a directed graph of code dependencies.

    Nodes represent functions/classes/methods in the codebase.
    Edges represent call relationships (A calls B → edge A→B).
    
    Node key format: "file_path::function_name"
    Example: "src/code_reviewer/core/reviewer.py::combine_findings"
    
    Node attributes:
        - file_path: Path to the source file
        - name: Function/class/method name
        - chunk_type: "function", "class", or "method"
        - complexity: Cyclomatic complexity score
        - start_line: Starting line number in the file
    
    External calls (to stdlib or third-party libs) are tracked as
    "external::{function_name}" nodes for completeness.
    """

    def __init__(self):
        """Initialize an empty directed graph."""
        self.graph = nx.DiGraph()
        self._centrality_cache: Optional[dict[str, float]] = None
        self._lookup: dict[str, str] = {}  # function_name -> node_key

    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a code chunk as a node to the graph.

        Creates a node with the key format "file_path::function_name"
        and stores chunk metadata as node attributes. Adds unresolved
        edges for all calls made by this chunk (resolved later in build()).

        Args:
            chunk: The code chunk to add as a graph node.
        """
        node_key = f"{chunk.file_path}::{chunk.name}"
        
        # Add node with attributes
        self.graph.add_node(
            node_key,
            file_path=chunk.file_path,
            name=chunk.name,
            chunk_type=chunk.chunk_type,
            complexity=chunk.complexity,
            start_line=chunk.start_line,
        )
        
        # Add unresolved edges for calls (will be resolved in build())
        for called_name in chunk.calls:
            # Temporary edge - will be updated in build() with resolved target
            self.graph.add_edge(node_key, called_name)

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build the complete dependency graph from a list of chunks.

        Two-pass algorithm:
        1. First pass: Add all chunks as nodes and build lookup table
        2. Second pass: Resolve call edges to actual node keys
        
        Args:
            chunks: List of all code chunks to include in the graph.
        """
        # Clear existing graph and cache
        self.graph.clear()
        self._centrality_cache = None
        self._lookup.clear()
        
        # First pass: Add all chunks as nodes
        for chunk in chunks:
            node_key = f"{chunk.file_path}::{chunk.name}"
            
            self.graph.add_node(
                node_key,
                file_path=chunk.file_path,
                name=chunk.name,
                chunk_type=chunk.chunk_type,
                complexity=chunk.complexity,
                start_line=chunk.start_line,
            )
            
            # Build lookup: function_name -> node_key
            # If duplicate names exist, last one wins (limitation)
            self._lookup[chunk.name] = node_key
        
        # Second pass: Resolve edges
        for chunk in chunks:
            source_key = f"{chunk.file_path}::{chunk.name}"
            
            for called_name in chunk.calls:
                # Try to resolve the called name to a node key
                if called_name in self._lookup:
                    target_key = self._lookup[called_name]
                    self.graph.add_edge(source_key, target_key)
                else:
                    # Mark as external call (stdlib or third-party)
                    external_key = f"external::{called_name}"
                    
                    # Add external node if it doesn't exist
                    if external_key not in self.graph:
                        self.graph.add_node(
                            external_key,
                            file_path="<external>",
                            name=called_name,
                            chunk_type="external",
                            complexity=0,
                            start_line=0,
                        )
                    
                    self.graph.add_edge(source_key, external_key)
        
        logger.info(
            "Dependency graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def get_callers(self, node_key: str) -> list[str]:
        """Get all nodes that call this node (predecessors).

        Args:
            node_key: The node to find callers for.

        Returns:
            List of node keys that have edges pointing TO this node.
            Empty list if node doesn't exist or has no callers.
        """
        if node_key not in self.graph:
            return []
        return list(self.graph.predecessors(node_key))

    def get_callees(self, node_key: str) -> list[str]:
        """Get all nodes that this node calls (successors).

        Args:
            node_key: The node to find callees for.

        Returns:
            List of node keys this node has edges pointing TO.
            Empty list if node doesn't exist or makes no calls.
        """
        if node_key not in self.graph:
            return []
        return list(self.graph.successors(node_key))

    def get_centrality_scores(self) -> dict[str, float]:
        """Calculate PageRank centrality scores for all nodes.

        Uses NetworkX PageRank algorithm to measure importance based on
        the structure of incoming edges. Nodes with many incoming edges
        from other important nodes score higher.

        Returns:
            Dictionary mapping node_key → centrality score (0.0 to 1.0).
            Higher score means more things depend on this node.

        Note:
            Result is cached after first computation. Call build() to
            clear the cache.
        """
        if self._centrality_cache is not None:
            return self._centrality_cache
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            scores = nx.pagerank(self.graph)
            self._centrality_cache = scores
            logger.debug("Centrality scores computed for %d nodes", len(scores))
            return scores
        except Exception as e:
            logger.warning("Failed to compute PageRank centrality: %s", e)
            return {}

    def get_centrality_score(self, node_key: str) -> float:
        """Get the centrality score for a single node.

        Args:
            node_key: The node to get the score for.

        Returns:
            Centrality score (0.0 to 1.0), or 0.0 if node doesn't exist.
        """
        scores = self.get_centrality_scores()
        return scores.get(node_key, 0.0)

    def find_node_key(self, file_path: str, function_name: str) -> Optional[str]:
        """Look up the full node key for a given file and function.

        Args:
            file_path: Path to the source file.
            function_name: Name of the function/class/method.

        Returns:
            Full node key in format "file_path::function_name",
            or None if not found in the graph.
        """
        node_key = f"{file_path}::{function_name}"
        if node_key in self.graph:
            return node_key
        return None

    def get_call_chain(self, start_node: str, max_depth: int = 4) -> list[list[str]]:
        """Find all call chains starting from a given node using BFS.

        Performs breadth-first search following outgoing edges to find
        all paths from start_node up to max_depth hops. Used by the
        correlator to understand issue propagation paths.

        Args:
            start_node: The node key to start the search from.
            max_depth: Maximum depth to search (default: 4 hops).

        Returns:
            List of paths, where each path is a list of node keys.
            Empty list if start_node doesn't exist.

        Example:
            [
                ["A", "B", "C"],
                ["A", "B", "D"],
                ["A", "E"],
            ]
        """
        if start_node not in self.graph:
            return []
        
        paths: list[list[str]] = []
        queue: deque[tuple[str, list[str]]] = deque([(start_node, [start_node])])
        
        while queue:
            current, path = queue.popleft()
            
            # Stop if we've reached max depth
            if len(path) > max_depth:
                continue
            
            # Get all callees
            callees = self.get_callees(current)
            
            if not callees:
                # Leaf node - add path to results
                if len(path) > 1:  # Don't include single-node paths
                    paths.append(path)
            else:
                # Continue BFS
                for callee in callees:
                    # Avoid cycles
                    if callee not in path:
                        new_path = path + [callee]
                        queue.append((callee, new_path))
        
        return paths

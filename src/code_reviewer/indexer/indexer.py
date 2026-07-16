"""Codebase indexer orchestrator.

Coordinates chunking, embedding, and storage of code for semantic search.
Supports incremental indexing by tracking file hashes and only re-indexing
changed files.
"""

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from code_reviewer.indexer.cache import FileHashCache
from code_reviewer.indexer.chunker import PythonChunker, CodeChunk
from code_reviewer.indexer.dependency_graph import DependencyGraph
from code_reviewer.indexer.embedder import ChunkEmbedder
from code_reviewer.indexer.file_walker import FileWalker
from code_reviewer.indexer.store import CodebaseStore

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of an indexing operation.

    Attributes:
        indexed_files: Number of files that were indexed
        skipped_files: Number of files skipped (unchanged)
        total_chunks: Total number of code chunks created
        total_edges: Total number of call edges in dependency graph
        elapsed_seconds: Time taken to complete indexing
    """

    indexed_files: int
    skipped_files: int
    total_chunks: int
    total_edges: int
    elapsed_seconds: float


class CodebaseIndexer:
    """Orchestrates indexing of a codebase into the vector store.

    The indexer:
    1. Walks the codebase to find Python files
    2. Checks which files have changed since last index
    3. Chunks changed files into semantic units
    4. Generates embeddings for all chunks
    5. Stores chunks and embeddings in ChromaDB
    6. Updates the hash cache

    Supports incremental indexing by default (only re-indexes changed files).
    Use force=True to re-index all files.
    """

    def __init__(self, project_root: Path):
        """Initialize the indexer.

        Args:
            project_root: Root directory of the project to index
        """
        self.root = project_root
        self.walker = FileWalker(project_root, include=["*.py"], max_files=0)
        self.chunker = PythonChunker()
        self.embedder = ChunkEmbedder()
        self.store = CodebaseStore(project_root)
        self.cache = FileHashCache(project_root)
        self.graph = DependencyGraph()
        
        # Path to persist the dependency graph
        graph_path = project_root / ".code-reviewer" / "cache" / "graph.pkl"
        self.graph_path = graph_path

    def index(self, force: bool = False) -> IndexResult:
        """Index the codebase into the vector store.

        Args:
            force: If True, re-index all files even if unchanged.
                  If False (default), only index files that have changed.

        Returns:
            IndexResult with statistics about the indexing operation

        Process:
            1. Walk all Python files in the codebase
            2. For each file:
               - Check if it has changed (via hash cache)
               - Skip if unchanged (unless force=True)
               - Chunk the file into semantic units
               - Generate embeddings for all chunks
               - Store chunks in the vector database
               - Update file metadata and hash cache
            3. Build dependency graph from all chunks
            4. Save graph to disk
            5. Save cache to disk
            6. Return statistics
        """
        start_time = time.perf_counter()

        indexed_count = 0
        skipped_count = 0
        total_chunks = 0
        all_chunks: list[CodeChunk] = []  # Collect all chunks for graph building

        logger.info(f"Starting index of {self.root}")

        for file_path in self.walker.walk():
            # Check if file has changed
            changed, file_hash = self.cache.is_changed(file_path)

            if not changed and not force:
                skipped_count += 1
                logger.debug(f"Skipping unchanged file: {file_path}")
                continue

            # Read file content
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

            # Chunk the file
            chunks = self.chunker.chunk(file_path, source, file_hash)
            if not chunks:
                logger.debug(f"No chunks extracted from {file_path}")
                continue

            # Collect chunks for graph building
            all_chunks.extend(chunks)

            # Delete stale chunks for this file
            self.store.delete_by_file(str(file_path))

            # Generate embeddings
            try:
                embeddings = self.embedder.embed_chunks(chunks)
            except Exception as e:
                logger.warning(f"Failed to embed chunks for {file_path}: {e}")
                continue

            # Store chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                self.store.upsert_chunk(chunk, embedding)

            # Update file metadata
            self.store.upsert_file_record(
                str(file_path), file_hash, len(chunks)
            )

            # Update cache
            self.cache.update(str(file_path), file_hash)

            indexed_count += 1
            total_chunks += len(chunks)
            logger.info(
                f"Indexed {file_path}: {len(chunks)} chunks "
                f"({indexed_count} files so far)"
            )

        # Build dependency graph from all chunks
        logger.info("Building dependency graph...")
        self.graph.build(all_chunks)
        
        # Save graph to disk
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info(f"Dependency graph saved to {self.graph_path}")
        except Exception as e:
            logger.warning(f"Failed to save dependency graph: {e}")

        # Save cache to disk
        self.cache.save()

        elapsed_seconds = time.perf_counter() - start_time
        total_edges = self.graph.graph.number_of_edges()

        logger.info(
            f"Indexing complete: {indexed_count} files indexed, "
            f"{skipped_count} skipped, {total_chunks} chunks, "
            f"{total_edges} edges, {elapsed_seconds:.1f}s"
        )

        return IndexResult(
            indexed_files=indexed_count,
            skipped_files=skipped_count,
            total_chunks=total_chunks,
            total_edges=total_edges,
            elapsed_seconds=elapsed_seconds,
        )

    @classmethod
    def load_graph(cls, project_root: Path) -> DependencyGraph | None:
        """Load the dependency graph from disk.

        Args:
            project_root: Root directory of the project

        Returns:
            DependencyGraph if the graph file exists and can be loaded,
            None otherwise.
        """
        graph_path = project_root / ".code-reviewer" / "cache" / "graph.pkl"
        
        if not graph_path.exists():
            logger.debug(f"Dependency graph not found at {graph_path}")
            return None
        
        try:
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)
            logger.info(f"Loaded dependency graph from {graph_path}")
            return graph
        except Exception as e:
            logger.warning(f"Failed to load dependency graph: {e}")
            return None

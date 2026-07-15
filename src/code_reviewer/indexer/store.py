"""Persistent vector store for code chunks using ChromaDB.

Stores code embeddings and metadata for semantic search across the codebase.
The store maintains three collections: chunks (code embeddings), files (per-file
metadata), and findings (reserved for Layer 4).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings

from code_reviewer.indexer.chunker import CodeChunk

logger = logging.getLogger(__name__)


class CodebaseStore:
    """Persistent ChromaDB store for code chunk embeddings and metadata.

    Store location: {project_root}/.code-reviewer/index/
    
    Collections:
        chunks: Stores chunk embeddings and metadata for semantic search
        files: Stores per-file metadata (hash, chunk count, indexed timestamp)
        findings: Reserved for Layer 4 (review findings cache)
    """

    def __init__(self, project_root: Path):
        """Initialize the ChromaDB store.

        Args:
            project_root: Root directory of the project being indexed.
                         Store will be created at {project_root}/.code-reviewer/index/
        """
        self.project_root = project_root
        self.store_path = project_root / ".code-reviewer" / "index"
        
        # Create directory if it doesn't exist
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.store_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collections
            self.chunks_collection = self.client.get_or_create_collection(
                name="chunks",
                metadata={"description": "Code chunk embeddings for semantic search"}
            )
            self.files_collection = self.client.get_or_create_collection(
                name="files",
                metadata={"description": "Per-file metadata and indexing status"}
            )
            self.findings_collection = self.client.get_or_create_collection(
                name="findings",
                metadata={"description": "Review findings cache (Layer 4)"}
            )
            
            logger.info(f"Initialized CodebaseStore at {self.store_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def upsert_chunk(self, chunk: CodeChunk, embedding: list[float]) -> None:
        """Add or update a chunk in the store.

        Args:
            chunk: CodeChunk object with metadata
            embedding: Vector embedding for the chunk

        Note:
            All metadata values must be str, int, float, or bool for ChromaDB.
            The chunk.source_code is stored as the document text.
        """
        try:
            self.chunks_collection.upsert(
                ids=[chunk.chunk_id],
                embeddings=[embedding],
                documents=[chunk.source_code],
                metadatas=[{
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "complexity": chunk.complexity,
                    "has_docstring": chunk.docstring is not None,
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to upsert chunk {chunk.chunk_id}: {e}")

    def get_chunks_by_file(self, file_path: str) -> list[dict]:
        """Retrieve all chunks for a specific file.

        Args:
            file_path: Path to the source file

        Returns:
            List of metadata dictionaries for all chunks in the file.
            Returns empty list if no chunks found or on error.
        """
        try:
            results = self.chunks_collection.get(
                where={"file_path": file_path},
                include=["metadatas", "documents"]
            )
            
            # Combine metadata with document text
            chunks = []
            if results and results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results.get("metadatas") else {}
                    document = results["documents"][i] if results.get("documents") else ""
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_code": document,
                        **metadata
                    })
            return chunks
        except Exception as e:
            logger.warning(f"Failed to get chunks for {file_path}: {e}")
            return []

    def search_similar(self, embedding: list[float], n: int = 5) -> list[dict]:
        """Search for similar code chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            n: Number of results to return (default: 5)

        Returns:
            List of metadata dictionaries for the top n most similar chunks.
            Returns empty list on error.
        """
        try:
            results = self.chunks_collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            chunks = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    document = results["documents"][0][i] if results.get("documents") else ""
                    distance = results["distances"][0][i] if results.get("distances") else None
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_code": document,
                        "distance": distance,
                        **metadata
                    })
            return chunks
        except Exception as e:
            logger.warning(f"Failed to search similar chunks: {e}")
            return []

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path to the source file

        Note:
            Used when a file is deleted or needs to be re-indexed.
        """
        try:
            # Get all chunk IDs for this file
            results = self.chunks_collection.get(
                where={"file_path": file_path},
                include=[]  # Only need IDs
            )
            
            if results and results["ids"]:
                self.chunks_collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete chunks for {file_path}: {e}")

    def get_file_hash(self, file_path: str) -> str | None:
        """Retrieve the stored hash for a file.

        Args:
            file_path: Path to the source file

        Returns:
            The stored sha256 hash string, or None if file not found or on error.
        """
        try:
            results = self.files_collection.get(
                ids=[file_path],
                include=["metadatas"]
            )
            
            if results and results["ids"] and results["metadatas"]:
                return results["metadatas"][0].get("file_hash")
            return None
        except Exception as e:
            logger.warning(f"Failed to get file hash for {file_path}: {e}")
            return None

    def upsert_file_record(
        self, file_path: str, file_hash: str, chunk_count: int
    ) -> None:
        """Store or update per-file metadata.

        Args:
            file_path: Path to the source file (used as document ID)
            file_hash: sha256 hash of the file contents
            chunk_count: Number of chunks extracted from this file
        """
        try:
            indexed_at = datetime.now(timezone.utc).isoformat()
            
            self.files_collection.upsert(
                ids=[file_path],
                documents=[file_path],  # Store file path as document
                metadatas=[{
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "chunk_count": chunk_count,
                    "indexed_at": indexed_at,
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to upsert file record for {file_path}: {e}")

    def get_stats(self) -> dict:
        """Get store statistics.

        Returns:
            Dictionary with:
                total_files: Number of indexed files
                total_chunks: Number of stored chunks
                last_indexed: ISO timestamp of most recent indexing, or None
        """
        try:
            files_count = self.files_collection.count()
            chunks_count = self.chunks_collection.count()
            
            # Get the most recent indexed_at timestamp
            last_indexed = None
            if files_count > 0:
                # Get all files and find the most recent timestamp
                results = self.files_collection.get(
                    include=["metadatas"]
                )
                if results and results["metadatas"]:
                    timestamps = [
                        meta.get("indexed_at")
                        for meta in results["metadatas"]
                        if meta.get("indexed_at")
                    ]
                    if timestamps:
                        last_indexed = max(timestamps)
            
            return {
                "total_files": files_count,
                "total_chunks": chunks_count,
                "last_indexed": last_indexed,
            }
        except Exception as e:
            logger.warning(f"Failed to get store stats: {e}")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "last_indexed": None,
            }

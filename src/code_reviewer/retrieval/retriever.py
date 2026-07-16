"""Context retriever using ChromaDB semantic search.

Provides semantic search capabilities to find similar code chunks across
the codebase. Used to augment code reviews with relevant context from
other parts of the project.
"""

import logging
from pathlib import Path

from code_reviewer.indexer.embedder import ChunkEmbedder
from code_reviewer.indexer.store import CodebaseStore

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieves semantically similar code chunks for context.

    Uses ChromaDB vector search to find functions, classes, and methods
    that are similar to a given piece of code. This context helps LLMs
    understand patterns and conventions across the codebase.
    """

    def __init__(self, project_root: Path):
        """Initialize the context retriever.

        Args:
            project_root: Root directory of the project (where index is stored)
        """
        self.project_root = project_root
        self.store = CodebaseStore(project_root)
        self.embedder = ChunkEmbedder()
        self._model_loaded = False

    def index_exists(self) -> bool:
        """Check if a valid index exists for this project.

        Returns:
            True if .code-reviewer/index/ exists and contains chunks,
            False otherwise.

        Note:
            This is a guard method used before attempting retrieval.
            If False, callers should skip context retrieval gracefully.
        """
        index_dir = self.project_root / ".code-reviewer" / "index"

        if not index_dir.exists():
            return False

        # Check if ChromaDB has any chunks
        try:
            stats = self.store.get_stats()
            return stats["total_chunks"] > 0
        except Exception as e:
            logger.warning(f"Failed to check index stats: {e}")
            return False

    def get_context_for_function(
        self,
        file_path: str,
        function_name: str,
        n_similar: int = 3,
    ) -> str:
        """Get context by finding similar functions to a specific function.

        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function to find similar code for
            n_similar: Number of similar functions to retrieve (default: 3)

        Returns:
            Formatted context string with similar functions, or empty string
            if the function is not found or no similar functions exist.

        Process:
            1. Find the chunk for this specific function in the store
            2. Embed the function's source code
            3. Search for similar chunks
            4. Filter out the function itself
            5. Format and return the top n_similar results
        """
        if not self.index_exists():
            return ""

        # Find the specific function chunk
        try:
            chunks = self.store.get_chunks_by_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to get chunks for {file_path}: {e}")
            return ""

        # Find the target function
        target_chunk = None
        for chunk in chunks:
            if chunk.get("name") == function_name:
                target_chunk = chunk
                break

        if not target_chunk:
            logger.debug(f"Function {function_name} not found in {file_path}")
            return ""

        # Embed the function's source code
        try:
            source_code = target_chunk.get("source_code", "")
            if not source_code:
                return ""

            embedding = self.embedder.embed_text(source_code)
        except Exception as e:
            logger.warning(f"Failed to embed function {function_name}: {e}")
            return ""

        # Search for similar chunks
        try:
            # Request n_similar + 1 to account for filtering out the function itself
            similar_chunks = self.store.search_similar(
                embedding, n=n_similar + 1
            )
        except Exception as e:
            logger.warning(f"Failed to search similar chunks: {e}")
            return ""

        # Filter out the function itself and limit results
        filtered_chunks = []
        for chunk in similar_chunks:
            # Skip if this is the exact same function
            if (
                chunk.get("file_path") == file_path
                and chunk.get("name") == function_name
            ):
                continue

            filtered_chunks.append(chunk)

            if len(filtered_chunks) >= n_similar:
                break

        if not filtered_chunks:
            return ""

        # Format the context
        return self._format_similar_functions(filtered_chunks)

    def get_context_for_file(
        self,
        file_path: str,
        n_similar: int = 3,
    ) -> str:
        """Get context by finding similar functions for all chunks in a file.

        Args:
            file_path: Path to the file to get context for
            n_similar: Number of similar functions to retrieve per chunk

        Returns:
            Formatted context string with similar functions from the codebase,
            or empty string if no index exists or no similar functions found.

        Process:
            1. Get all chunks for the file
            2. For each chunk, find similar functions
            3. Deduplicate results (same function may be similar to multiple chunks)
            4. Format and return combined context
        """
        if not self.index_exists():
            return ""

        # Get all chunks for this file
        try:
            chunks = self.store.get_chunks_by_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to get chunks for {file_path}: {e}")
            return ""

        if not chunks:
            logger.debug(f"No chunks found for {file_path}")
            return ""

        # Collect similar functions for all chunks in the file
        all_similar = []
        seen_functions = set()  # Track (file_path, name) to deduplicate

        for chunk in chunks:
            source_code = chunk.get("source_code", "")
            if not source_code:
                continue

            # Embed this chunk
            try:
                embedding = self.embedder.embed_text(source_code)
            except Exception as e:
                logger.warning(f"Failed to embed chunk {chunk.get('name')}: {e}")
                continue

            # Search for similar chunks
            try:
                similar_chunks = self.store.search_similar(
                    embedding, n=n_similar + 1
                )
            except Exception as e:
                logger.warning(f"Failed to search similar chunks: {e}")
                continue

            # Add unique similar functions
            for similar in similar_chunks:
                similar_file = similar.get("file_path")
                similar_name = similar.get("name")

                # Skip if this is from the same file
                if similar_file == file_path:
                    continue

                # Skip if we've already seen this function
                func_key = (similar_file, similar_name)
                if func_key in seen_functions:
                    continue

                seen_functions.add(func_key)
                all_similar.append(similar)

                # Stop if we have enough
                if len(all_similar) >= n_similar:
                    break

            if len(all_similar) >= n_similar:
                break

        if not all_similar:
            return ""

        # Format the context with a header
        header = (
            "## Codebase Context\n\n"
            "The following functions from the codebase are semantically "
            "similar to code in this file. Use them for context only.\n\n"
        )

        return header + self._format_similar_functions(all_similar)

    def _format_similar_functions(self, chunks: list[dict]) -> str:
        """Format a list of chunks as a readable context string.

        Args:
            chunks: List of chunk metadata dictionaries

        Returns:
            Formatted markdown string with function details and source code
        """
        if not chunks:
            return ""

        sections = [
            "## Similar Functions in Codebase\n",
            "(Provided for context only — do not review these)\n",
        ]

        for chunk in chunks:
            name = chunk.get("name", "unknown")
            file_path = chunk.get("file_path", "unknown")
            start_line = chunk.get("start_line", 0)
            end_line = chunk.get("end_line", 0)
            source_code = chunk.get("source_code", "")

            # Truncate source code to 30 lines max
            lines = source_code.split("\n")
            if len(lines) > 30:
                lines = lines[:30]
                source_code = "\n".join(lines) + "\n... (truncated)"
            else:
                source_code = "\n".join(lines)

            section = (
                f"\n### {name} ({file_path}:{start_line}–{end_line})\n"
                f"```python\n{source_code}\n```\n"
            )
            sections.append(section)

        return "".join(sections)

"""Code embedding using sentence-transformers.

Provides efficient batch embedding for code chunks using the all-MiniLM-L6-v2
model. The model is ~90MB and downloads automatically on first use (may take
30-60 seconds on first run).

The embedder converts code chunks into dense vectors suitable for semantic
search in the ChromaDB vector store.
"""

import logging

from sentence_transformers import SentenceTransformer

from code_reviewer.indexer.chunker import CodeChunk

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """Embeds code chunks using sentence-transformers for semantic search.

    Uses the all-MiniLM-L6-v2 model which provides:
    - Fast inference (suitable for batch processing)
    - Good semantic understanding for code
    - 384-dimensional embeddings
    - ~90MB model size (downloads on first use)
    """

    def __init__(self):
        """Initialize the embedder and load the model.

        Note:
            On first run, the model will be downloaded (~90MB).
            This may take 30-60 seconds depending on network speed.
        """
        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        print("Loading embedding model...")
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("Embedding model loaded successfully")

    def embed_chunk(self, chunk: CodeChunk) -> list[float]:
        """Embed a single code chunk.

        Args:
            chunk: CodeChunk object to embed

        Returns:
            List of floats representing the embedding vector (384 dimensions)

        Note:
            The embedding is computed from the chunk's name, docstring, and
            source code concatenated together. This provides semantic context
            for search.
        """
        # Build text: name + docstring + source code
        text = f"{chunk.name}\n{chunk.docstring or ''}\n{chunk.source_code}"
        
        # Encode and convert to list
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed multiple code chunks efficiently in batch.

        Args:
            chunks: List of CodeChunk objects to embed

        Returns:
            List of embedding vectors, one per chunk. Empty list if input is empty.

        Note:
            Batch processing is significantly faster than calling embed_chunk()
            in a loop. Use this method when embedding multiple chunks.
        """
        if not chunks:
            return []
        
        # Build texts for all chunks
        texts = [
            f"{chunk.name}\n{chunk.docstring or ''}\n{chunk.source_code}"
            for chunk in chunks
        ]
        
        # Batch encode
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )
        
        # Convert to list of lists
        return [embedding.tolist() for embedding in embeddings]

    def embed_text(self, text: str) -> list[float]:
        """Embed a raw text string.

        Args:
            text: Raw text to embed (typically a search query)

        Returns:
            List of floats representing the embedding vector (384 dimensions)

        Note:
            Used in Layer 3 for embedding user search queries to find
            semantically similar code chunks.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

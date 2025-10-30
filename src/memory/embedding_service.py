"""Embedding Service for RAG Memory System.

This service generates vector embeddings using Ollama's nomic-embed-text model.
Embeddings are used for semantic search in the memory system.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import OrderedDict

from src.services.ollama_service import OllamaService, OllamaConfig


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embedding: List[float]
    model: str
    dimensions: int


class EmbeddingService:
    """Service for generating text embeddings using Ollama.

    Features:
    - Generates embeddings using nomic-embed-text model
    - Caches embeddings to avoid redundant API calls
    - Supports batch embedding generation
    - Optional L2 normalization
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 8192,
        normalize: bool = False,
        max_cache_size: int = 1000,
    ):
        """Initialize the embedding service.

        Args:
            model: Ollama embedding model to use
            base_url: Ollama server URL
            max_tokens: Maximum tokens per text
            normalize: Whether to normalize embeddings to unit length
            max_cache_size: Maximum number of cached embeddings
        """
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.max_cache_size = max_cache_size

        # Initialize Ollama service
        config = OllamaConfig(host=base_url, model=model)
        self.ollama_service = OllamaService(config=config)

        # LRU cache for embeddings
        self.embedding_cache: OrderedDict[str, EmbeddingResponse] = OrderedDict()

    async def generate_embedding(self, text: str) -> EmbeddingResponse:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResponse with embedding vector

        Raises:
            ValueError: If text is empty or whitespace-only
        """
        # Validate text
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Truncate if needed
        text = self._truncate_text(text)

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            # Move to end (LRU)
            self.embedding_cache.move_to_end(cache_key)
            return self.embedding_cache[cache_key]

        # Generate embedding via Ollama
        response = await self.ollama_service.generate_embedding(
            text=text,
            model=self.model
        )

        embedding = response.embedding

        # Normalize if requested
        if self.normalize:
            embedding = self._normalize_embedding(embedding)

        # Create response
        embed_response = EmbeddingResponse(
            embedding=embedding,
            model=self.model,
            dimensions=len(embedding),
        )

        # Cache the result
        self._add_to_cache(cache_key, embed_response)

        return embed_response

    async def generate_batch_embeddings(
        self,
        texts: List[str]
    ) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResponse objects
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]

        # Generate embeddings for each text
        results = []
        for text in valid_texts:
            embedding = await self.generate_embedding(text)
            results.append(embedding)

        return results

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max tokens.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        # Rough estimate: 1 token ~= 4 characters
        max_chars = self.max_tokens * 4

        if len(text) <= max_chars:
            return text

        return text[:max_chars]

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length (L2 normalization).

        Args:
            embedding: Vector to normalize

        Returns:
            Normalized vector
        """
        # Calculate L2 norm
        norm = math.sqrt(sum(x**2 for x in embedding))

        if norm == 0:
            return embedding

        # Normalize
        return [x / norm for x in embedding]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Text to hash

        Returns:
            Cache key (hash)
        """
        # Use SHA256 hash of text as cache key
        return hashlib.sha256(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, response: EmbeddingResponse):
        """Add embedding to cache with LRU eviction.

        Args:
            key: Cache key
            response: Embedding response to cache
        """
        # Add to cache
        self.embedding_cache[key] = response

        # Evict oldest if cache is full
        while len(self.embedding_cache) > self.max_cache_size:
            self.embedding_cache.popitem(last=False)

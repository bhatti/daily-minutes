"""Memory Retrieval Service for RAG System.

This service provides high-level retrieval functions that combine
embedding generation and vector search for contextual memory access.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from src.memory.embedding_service import EmbeddingService
from src.memory.vector_store import MemoryStore
from src.memory.models import (
    Memory,
    MemoryType,
    ActionItemStatus,
    RetrievalResult,
)


class MemoryRetriever:
    """High-level service for retrieving relevant memories.

    Combines embedding generation and vector search to provide
    contextual memory retrieval for the ReAct agent.

    Features:
    - Automatic embedding generation
    - Text-based semantic search
    - Filtered retrieval by type and metadata
    - Recent memory access
    - Action item tracking
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        persist_directory: str = "./chroma_data",
        collection_name: str = "memories",
    ):
        """Initialize the memory retriever.

        Args:
            embedding_model: Ollama embedding model to use
            ollama_base_url: Ollama server URL
            persist_directory: ChromaDB persistence directory
            collection_name: Name of the ChromaDB collection
        """
        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model=embedding_model,
            base_url=ollama_base_url,
        )

        # Initialize vector store
        self.memory_store = MemoryStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    async def store_memory(self, memory: Memory) -> None:
        """Store a memory with automatic embedding generation.

        If the memory doesn't have an embedding, one will be generated
        automatically from the memory's content.

        Args:
            memory: Memory object to store
        """
        # Generate embedding if not present
        if memory.embedding is None:
            embedding_response = await self.embedding_service.generate_embedding(
                memory.content
            )
            memory.embedding = embedding_response.embedding

        # Store in vector database
        await self.memory_store.store_memory(memory)

    async def retrieve_by_text(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories using text-based semantic search.

        Args:
            query: Text query to search for
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        # Generate embedding for query
        embedding_response = await self.embedding_service.generate_embedding(query)

        # Search vector store
        results = await self.memory_store.retrieve_similar(
            query_embedding=embedding_response.embedding,
            k=k,
            filters=filters,
        )

        return results

    async def get_relevant_context(
        self,
        query: str,
        context_window: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Get relevant contextual memories for a query.

        This is the primary method for retrieving context to inject
        into the ReAct agent's reasoning process.

        Args:
            query: Question or task description
            context_window: Number of memories to retrieve
            filters: Optional metadata filters

        Returns:
            List of relevant memories with relevance scores
        """
        return await self.retrieve_by_text(
            query=query,
            k=context_window,
            filters=filters,
        )

    async def get_recent_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Retrieve recent memories, optionally filtered by type.

        Args:
            memory_type: Optional memory type filter
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects sorted by recency
        """
        if memory_type is None:
            # Get all recent memories
            count = await self.memory_store.count_memories()
            if count == 0:
                return []

            # Query with a generic embedding to get recent memories
            # (In a real implementation, you'd want a time-based index)
            results = await self.memory_store.retrieve_similar(
                query_embedding=[0.0] * 768,  # Placeholder
                k=min(limit, count),
            )
            memories = [r.memory for r in results]
        else:
            # Filter by type
            memories = await self.memory_store.filter_by_type(memory_type)

        # Sort by timestamp (most recent first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)

        return memories[:limit]

    async def get_pending_action_items(self) -> List[Memory]:
        """Retrieve all pending action items.

        Returns:
            List of pending ActionItemMemory objects
        """
        # Filter by action item type and pending status
        action_items = await self.memory_store.filter_by_type(
            MemoryType.ACTION_ITEM
        )

        # Filter for pending status
        pending_items = [
            item for item in action_items
            if item.metadata.get("status") == ActionItemStatus.PENDING.value
        ]

        return pending_items

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory from the store.

        Args:
            memory_id: ID of the memory to delete
        """
        await self.memory_store.delete_memory(memory_id)

    async def count_memories(self) -> int:
        """Count total memories in the store.

        Returns:
            Number of stored memories
        """
        return await self.memory_store.count_memories()

    async def clear_all_memories(self) -> None:
        """Clear all memories from the store.

        WARNING: This operation cannot be undone.
        """
        await self.memory_store.clear_collection()

"""Vector Store for RAG Memory System.

This service provides vector storage and retrieval using ChromaDB.
Stores memory embeddings for semantic search and contextual retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from src.memory.models import Memory, MemoryType, RetrievalResult


class MemoryStore:
    """ChromaDB-backed vector store for memories.

    Features:
    - Persistent vector storage with ChromaDB
    - Semantic search via embeddings
    - Metadata filtering
    - CRUD operations for memories
    """

    def __init__(
        self,
        collection_name: str = "memories",
        persist_directory: str = "./chroma_data",
    ):
        """Initialize the memory store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    async def store_memory(self, memory: Memory) -> None:
        """Store a memory in the vector store.

        Args:
            memory: Memory object to store

        Raises:
            ValueError: If memory doesn't have an embedding
        """
        if memory.embedding is None:
            raise ValueError("Memory must have an embedding")

        # Prepare metadata for ChromaDB
        metadata = self._prepare_metadata(memory)

        # Store in ChromaDB (upsert - adds or updates)
        self.collection.upsert(
            ids=[memory.id],
            embeddings=[memory.embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: ID of the memory to retrieve

        Returns:
            Memory object if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if not result["ids"]:
                return None

            # Reconstruct Memory object
            memory = self._reconstruct_memory(
                id=result["ids"][0],
                embedding=result["embeddings"][0],
                document=result["documents"][0],
                metadata=result["metadatas"][0],
            )

            return memory

        except Exception:
            return None

    async def retrieve_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve similar memories using semantic search.

        Args:
            query_embedding: Query vector for similarity search
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        # Check if collection is empty
        count = self.collection.count()
        if count == 0:
            return []

        # Query ChromaDB
        where = filters if filters else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),  # Don't request more than available
            where=where,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Convert to RetrievalResult objects
        retrieval_results = []
        for i in range(len(results["ids"][0])):
            memory = self._reconstruct_memory(
                id=results["ids"][0][i],
                embedding=results["embeddings"][0][i],
                document=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )

            # ChromaDB returns L2 distance, convert to similarity score
            distance = results["distances"][0][i]
            relevance_score = 1 / (1 + distance)  # Convert distance to score

            retrieval_results.append(
                RetrievalResult(
                    memory=memory,
                    relevance_score=relevance_score,
                    distance=distance,
                )
            )

        return retrieval_results

    async def filter_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Filter memories by type.

        Args:
            memory_type: Type of memories to retrieve

        Returns:
            List of Memory objects of the specified type
        """
        # Check if collection is empty
        count = self.collection.count()
        if count == 0:
            return []

        try:
            results = self.collection.get(
                where={"type": memory_type.value},
                include=["embeddings", "documents", "metadatas"],
            )

            if not results["ids"]:
                return []

            # Reconstruct Memory objects
            memories = []
            for i in range(len(results["ids"])):
                memory = self._reconstruct_memory(
                    id=results["ids"][i],
                    embedding=results["embeddings"][i],
                    document=results["documents"][i],
                    metadata=results["metadatas"][i],
                )
                memories.append(memory)

            return memories

        except Exception:
            return []

    async def filter_by_metadata(
        self, filters: Dict[str, Any]
    ) -> List[Memory]:
        """Filter memories by metadata.

        Args:
            filters: Dictionary of metadata filters

        Returns:
            List of Memory objects matching the filters
        """
        # Check if collection is empty
        count = self.collection.count()
        if count == 0:
            return []

        try:
            results = self.collection.get(
                where=filters,
                include=["embeddings", "documents", "metadatas"],
            )

            if not results["ids"]:
                return []

            # Reconstruct Memory objects
            memories = []
            for i in range(len(results["ids"])):
                memory = self._reconstruct_memory(
                    id=results["ids"][i],
                    embedding=results["embeddings"][i],
                    document=results["documents"][i],
                    metadata=results["metadatas"][i],
                )
                memories.append(memory)

            return memories

        except Exception:
            return []

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory from the store.

        Args:
            memory_id: ID of the memory to delete
        """
        try:
            self.collection.delete(ids=[memory_id])
        except Exception:
            # Silently ignore if ID doesn't exist
            pass

    async def count_memories(self) -> int:
        """Count total memories in the store.

        Returns:
            Number of memories stored
        """
        return self.collection.count()

    async def clear_collection(self) -> None:
        """Clear all memories from the collection."""
        # Delete the collection
        self.client.delete_collection(name=self.collection_name)

        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _prepare_metadata(self, memory: Memory) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage.

        ChromaDB has restrictions on metadata types (str, int, float, bool).
        Convert complex types to JSON strings.

        Args:
            memory: Memory object

        Returns:
            ChromaDB-compatible metadata dictionary
        """
        metadata = {
            "type": memory.type.value,
            "timestamp": memory.timestamp.isoformat(),
        }

        # Add user metadata, converting complex types
        for key, value in memory.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif isinstance(value, list):
                # Convert lists to JSON strings
                metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                metadata[key] = json.dumps(value)
            else:
                # Convert other types to strings
                metadata[key] = str(value)

        return metadata

    def _reconstruct_memory(
        self,
        id: str,
        embedding: List[float],
        document: str,
        metadata: Dict[str, Any],
    ) -> Memory:
        """Reconstruct Memory object from ChromaDB data.

        Args:
            id: Memory ID
            embedding: Vector embedding
            document: Content text
            metadata: Metadata dictionary

        Returns:
            Reconstructed Memory object
        """
        # Extract standard fields
        memory_type = MemoryType(metadata.pop("type"))
        timestamp_str = metadata.pop("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str)

        # Parse JSON strings back to objects
        parsed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Try to parse as JSON
                try:
                    parsed_value = json.loads(value)
                    parsed_metadata[key] = parsed_value
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, keep as string
                    parsed_metadata[key] = value
            else:
                parsed_metadata[key] = value

        # Create Memory object
        memory = Memory(
            id=id,
            type=memory_type,
            content=document,
            metadata=parsed_metadata,
            timestamp=timestamp,
            embedding=embedding,
        )

        return memory

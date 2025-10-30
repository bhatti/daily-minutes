"""Unit tests for Vector Store (ChromaDB).

Following TDD - write tests first, then implement.
Tests the ChromaDB-backed vector storage for memories.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List
import tempfile
import shutil
import os

from src.memory.vector_store import MemoryStore
from src.memory.models import (
    Memory,
    MemoryType,
    DailyBriefMemory,
    ActionItemMemory,
    ActionItemStatus,
    RetrievalResult,
)


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB, clean up after test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def memory_store(temp_chroma_dir):
    """Create a MemoryStore instance with temporary directory."""
    return MemoryStore(persist_directory=temp_chroma_dir)


class TestMemoryStoreInitialization:
    """Test MemoryStore initialization."""

    def test_store_initialization_with_defaults(self, temp_chroma_dir):
        """Test creating store with default parameters."""
        store = MemoryStore(persist_directory=temp_chroma_dir)

        assert store.collection_name == "memories"
        assert store.client is not None

    def test_store_initialization_with_custom_params(self, temp_chroma_dir):
        """Test creating store with custom parameters."""
        store = MemoryStore(
            collection_name="test_memories",
            persist_directory=temp_chroma_dir,
        )

        assert store.collection_name == "test_memories"
        assert store.persist_directory == temp_chroma_dir

    def test_store_has_collection(self, memory_store):
        """Test that store initializes with a collection."""
        assert hasattr(memory_store, 'collection')
        assert memory_store.collection is not None


class TestMemoryStoreStorage:
    """Test storing memories in vector store."""

    @pytest.mark.asyncio
    async def test_store_memory_success(self, memory_store):
        """Test successfully storing a memory."""
        memory = Memory(
            id="test-001",
            type=MemoryType.CONVERSATION,
            content="Test conversation content",
            metadata={"user": "Alice"},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        # Should not raise any exceptions
        await memory_store.store_memory(memory)

        # Verify memory was stored
        result = await memory_store.get_memory_by_id("test-001")
        assert result is not None
        assert result.id == "test-001"
        assert result.content == "Test conversation content"

    @pytest.mark.asyncio
    async def test_store_memory_without_embedding(self, memory_store):
        """Test storing a memory without pre-computed embedding."""
        memory = Memory(
            id="test-002",
            type=MemoryType.DAILY_BRIEF,
            content="Daily brief content",
            metadata={"date": "2025-01-15"},
            embedding=None,
        )

        # Should raise error if embedding is required
        with pytest.raises(ValueError, match="Memory must have an embedding"):
            await memory_store.store_memory(memory)

    @pytest.mark.asyncio
    async def test_store_multiple_memories(self, memory_store):
        """Test storing multiple memories."""
        memories = [
            Memory(
                id=f"mem-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Content {i}",
                metadata={"index": i},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
            )
            for i in range(5)
        ]

        for memory in memories:
            await memory_store.store_memory(memory)

        # Verify all were stored
        for i in range(5):
            result = await memory_store.get_memory_by_id(f"mem-{i}")
            assert result is not None
            assert result.metadata["index"] == i

    @pytest.mark.asyncio
    async def test_store_memory_overwrites_existing(self, memory_store):
        """Test that storing with same ID overwrites existing memory."""
        # Store original
        memory1 = Memory(
            id="test-003",
            type=MemoryType.ACTION_ITEM,
            content="Original content",
            metadata={"version": 1},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        await memory_store.store_memory(memory1)

        # Store updated version
        memory2 = Memory(
            id="test-003",
            type=MemoryType.ACTION_ITEM,
            content="Updated content",
            metadata={"version": 2},
            embedding=[0.4, 0.5, 0.6, 0.7, 0.8],
        )
        await memory_store.store_memory(memory2)

        # Verify updated content
        result = await memory_store.get_memory_by_id("test-003")
        assert result.content == "Updated content"
        assert result.metadata["version"] == 2


class TestMemoryRetrieval:
    """Test retrieving memories from vector store."""

    @pytest.mark.asyncio
    async def test_get_memory_by_id_success(self, memory_store):
        """Test retrieving a memory by ID."""
        # Store memory
        memory = Memory(
            id="test-004",
            type=MemoryType.MEETING_CONTEXT,
            content="Meeting notes",
            metadata={"meeting": "Sprint Planning"},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        await memory_store.store_memory(memory)

        # Retrieve by ID
        result = await memory_store.get_memory_by_id("test-004")
        assert result is not None
        assert result.id == "test-004"
        assert result.content == "Meeting notes"

    @pytest.mark.asyncio
    async def test_get_memory_by_id_not_found(self, memory_store):
        """Test retrieving non-existent memory."""
        result = await memory_store.get_memory_by_id("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_similar_memories(self, memory_store):
        """Test semantic search for similar memories."""
        # Store some memories
        memories = [
            Memory(
                id="similar-1",
                type=MemoryType.CONVERSATION,
                content="Discussion about Python programming",
                metadata={},
                embedding=[0.9, 0.1, 0.0, 0.0, 0.0],  # Close to query
            ),
            Memory(
                id="similar-2",
                type=MemoryType.CONVERSATION,
                content="Talk about weather today",
                metadata={},
                embedding=[0.0, 0.0, 1.0, 0.0, 0.0],  # Far from query
            ),
            Memory(
                id="similar-3",
                type=MemoryType.CONVERSATION,
                content="Python coding best practices",
                metadata={},
                embedding=[0.85, 0.15, 0.0, 0.0, 0.0],  # Close to query
            ),
        ]

        for memory in memories:
            await memory_store.store_memory(memory)

        # Query with embedding similar to Python-related content
        query_embedding = [0.95, 0.05, 0.0, 0.0, 0.0]
        results = await memory_store.retrieve_similar(
            query_embedding=query_embedding, k=2
        )

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Should return Python-related memories first
        assert results[0].memory.id in ["similar-1", "similar-3"]

    @pytest.mark.asyncio
    async def test_retrieve_similar_with_limit(self, memory_store):
        """Test limiting number of results."""
        # Store 10 memories
        for i in range(10):
            memory = Memory(
                id=f"limit-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Content {i}",
                metadata={},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
            )
            await memory_store.store_memory(memory)

        # Request only 3 results
        results = await memory_store.retrieve_similar(
            query_embedding=[0.5, 0.5, 0.5, 0.5, 0.5], k=3
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_similar_empty_store(self, temp_chroma_dir):
        """Test retrieving from empty store."""
        store = MemoryStore(persist_directory=temp_chroma_dir)

        results = await store.retrieve_similar(
            query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5], k=5
        )

        assert results == []


class TestMemoryFiltering:
    """Test filtering memories by type and metadata."""

    @pytest.mark.asyncio
    async def test_filter_by_memory_type(self, memory_store):
        """Test filtering memories by type."""
        # Store different types
        brief = DailyBriefMemory(
            id="brief-001",
            summary="Daily summary",
            key_points=["Point 1"],
            action_items=["Action 1"],
            emails_count=10,
            calendar_events_count=5,
            news_items_count=20,
        )
        brief.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        action = ActionItemMemory(
            id="action-001",
            description="Complete task",
            status=ActionItemStatus.PENDING,
            source="email",
        )
        action.embedding = [0.4, 0.5, 0.6, 0.7, 0.8]

        await memory_store.store_memory(brief)
        await memory_store.store_memory(action)

        # Filter by type
        action_items = await memory_store.filter_by_type(MemoryType.ACTION_ITEM)
        assert len(action_items) >= 1
        assert all(m.type == MemoryType.ACTION_ITEM for m in action_items)

    @pytest.mark.asyncio
    async def test_filter_by_metadata(self, memory_store):
        """Test filtering memories by metadata."""
        # Store memories with different metadata
        memories = [
            Memory(
                id="meta-1",
                type=MemoryType.CONVERSATION,
                content="Content 1",
                metadata={"category": "work", "priority": "high"},
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            Memory(
                id="meta-2",
                type=MemoryType.CONVERSATION,
                content="Content 2",
                metadata={"category": "personal", "priority": "low"},
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            ),
            Memory(
                id="meta-3",
                type=MemoryType.CONVERSATION,
                content="Content 3",
                metadata={"category": "work", "priority": "medium"},
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
            ),
        ]

        for memory in memories:
            await memory_store.store_memory(memory)

        # Filter by metadata
        work_memories = await memory_store.filter_by_metadata({"category": "work"})
        assert len(work_memories) >= 2
        assert all(m.metadata.get("category") == "work" for m in work_memories)

    @pytest.mark.asyncio
    async def test_filter_by_type_no_results(self, memory_store):
        """Test filtering with no matching results."""
        # Query for type that doesn't exist
        results = await memory_store.filter_by_type(MemoryType.EMAIL_SUMMARY)
        assert results == []


class TestMemoryDeletion:
    """Test deleting memories from vector store."""

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, memory_store):
        """Test successfully deleting a memory."""
        # Store memory
        memory = Memory(
            id="delete-001",
            type=MemoryType.CONVERSATION,
            content="To be deleted",
            metadata={},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        await memory_store.store_memory(memory)

        # Verify it exists
        result = await memory_store.get_memory_by_id("delete-001")
        assert result is not None

        # Delete it
        await memory_store.delete_memory("delete-001")

        # Verify it's gone
        result = await memory_store.get_memory_by_id("delete-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_non_existent_memory(self, memory_store):
        """Test deleting a memory that doesn't exist."""
        # Should not raise error
        await memory_store.delete_memory("non-existent-id")

    @pytest.mark.asyncio
    async def test_delete_multiple_memories(self, memory_store):
        """Test deleting multiple memories."""
        # Store memories
        ids = [f"delete-{i}" for i in range(5)]
        for id in ids:
            memory = Memory(
                id=id,
                type=MemoryType.CONVERSATION,
                content=f"Content for {id}",
                metadata={},
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            )
            await memory_store.store_memory(memory)

        # Delete all
        for id in ids:
            await memory_store.delete_memory(id)

        # Verify all are gone
        for id in ids:
            result = await memory_store.get_memory_by_id(id)
            assert result is None


class TestMemoryStoreUtilities:
    """Test utility functions of memory store."""

    @pytest.mark.asyncio
    async def test_count_memories(self, memory_store):
        """Test counting total memories in store."""
        # Store some memories
        for i in range(5):
            memory = Memory(
                id=f"count-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Content {i}",
                metadata={},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
            )
            await memory_store.store_memory(memory)

        count = await memory_store.count_memories()
        assert count >= 5

    @pytest.mark.asyncio
    async def test_clear_collection(self, temp_chroma_dir):
        """Test clearing all memories from collection."""
        store = MemoryStore(
            collection_name="test_clear", persist_directory=temp_chroma_dir
        )

        # Store memories
        for i in range(3):
            memory = Memory(
                id=f"clear-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Content {i}",
                metadata={},
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            )
            await store.store_memory(memory)

        # Clear collection
        await store.clear_collection()

        # Verify empty
        count = await store.count_memories()
        assert count == 0

"""Unit tests for Memory Retriever Service.

Following TDD - write tests first, then implement.
Tests the high-level retrieval service that combines embedding + vector search.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List
import tempfile
import shutil
import os

from src.memory.retrieval import MemoryRetriever
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
async def memory_retriever(temp_chroma_dir):
    """Create a MemoryRetriever instance with temporary directory."""
    retriever = MemoryRetriever(persist_directory=temp_chroma_dir)
    return retriever


class TestMemoryRetrieverInitialization:
    """Test MemoryRetriever initialization."""

    def test_retriever_initialization(self, temp_chroma_dir):
        """Test creating retriever with default parameters."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        assert retriever.embedding_service is not None
        assert retriever.memory_store is not None

    def test_retriever_with_custom_embedding_model(self, temp_chroma_dir):
        """Test creating retriever with custom embedding model."""
        retriever = MemoryRetriever(
            persist_directory=temp_chroma_dir,
            embedding_model="custom-model",
        )

        assert retriever.embedding_service.model == "custom-model"


class TestMemoryStorage:
    """Test storing memories with automatic embedding."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_memory_with_auto_embedding(self, MockOllama, temp_chroma_dir):
        """Test storing a memory with automatic embedding generation."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        memory = Memory(
            id="test-001",
            type=MemoryType.CONVERSATION,
            content="Test conversation",
            metadata={"user": "Alice"},
        )

        # Should automatically generate embedding and store
        await retriever.store_memory(memory)

        # Verify embedding was generated
        assert memory.embedding is not None
        assert len(memory.embedding) > 0

        # Verify memory can be retrieved
        result = await retriever.memory_store.get_memory_by_id("test-001")
        assert result is not None
        assert result.content == "Test conversation"

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_memory_with_existing_embedding(self, MockOllama, temp_chroma_dir):
        """Test storing a memory that already has an embedding."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        memory = Memory(
            id="test-002",
            type=MemoryType.DAILY_BRIEF,
            content="Brief content",
            metadata={},
            embedding=[0.9, 0.8, 0.7, 0.6, 0.5],
        )

        await retriever.store_memory(memory)

        # Should not generate new embedding
        result = await retriever.memory_store.get_memory_by_id("test-002")
        # Convert to list if it's a numpy array
        embedding_list = list(result.embedding) if hasattr(result.embedding, '__iter__') else result.embedding
        # Use approximate comparison for floating point precision
        assert embedding_list == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5], rel=1e-4)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_daily_brief(self, MockOllama, temp_chroma_dir):
        """Test storing a daily brief memory."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        brief = DailyBriefMemory(
            id="brief-001",
            summary="Daily summary",
            key_points=["Point 1", "Point 2"],
            action_items=["Task 1"],
            emails_count=10,
            calendar_events_count=5,
            news_items_count=20,
        )

        await retriever.store_memory(brief)

        # Verify stored
        result = await retriever.memory_store.get_memory_by_id("brief-001")
        assert result is not None
        assert result.type == MemoryType.DAILY_BRIEF


class TestTextRetrieval:
    """Test retrieving memories by text query."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_retrieve_by_text(self, MockOllama, temp_chroma_dir):
        """Test retrieving memories using text query."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.9, 0.1, 0.0, 0.0, 0.0]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store some memories
        memories = [
            Memory(
                id="mem-1",
                type=MemoryType.CONVERSATION,
                content="Python programming discussion",
                metadata={},
                embedding=[0.95, 0.05, 0.0, 0.0, 0.0],
            ),
            Memory(
                id="mem-2",
                type=MemoryType.CONVERSATION,
                content="Weather forecast",
                metadata={},
                embedding=[0.0, 0.0, 0.9, 0.1, 0.0],
            ),
        ]

        for memory in memories:
            await retriever.memory_store.store_memory(memory)

        # Retrieve by text query
        results = await retriever.retrieve_by_text("Python coding", k=2)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Should return Python-related memory first
        assert "Python" in results[0].memory.content

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_retrieve_by_text_with_filters(self, MockOllama, temp_chroma_dir):
        """Test retrieving memories with metadata filters."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.5, 0.5, 0.0, 0.0, 0.0]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store memories with different categories
        memories = [
            Memory(
                id="work-1",
                type=MemoryType.CONVERSATION,
                content="Work project discussion",
                metadata={"category": "work"},
                embedding=[0.6, 0.4, 0.0, 0.0, 0.0],
            ),
            Memory(
                id="personal-1",
                type=MemoryType.CONVERSATION,
                content="Personal notes",
                metadata={"category": "personal"},
                embedding=[0.55, 0.45, 0.0, 0.0, 0.0],
            ),
        ]

        for memory in memories:
            await retriever.memory_store.store_memory(memory)

        # Retrieve only work-related memories
        results = await retriever.retrieve_by_text(
            "project discussion",
            k=5,
            filters={"category": "work"}
        )

        assert len(results) >= 1
        assert all(r.memory.metadata.get("category") == "work" for r in results)


class TestContextRetrieval:
    """Test retrieving contextual memories for agent."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_relevant_context(self, MockOllama, temp_chroma_dir):
        """Test getting relevant context for a query."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.7, 0.3, 0.0, 0.0, 0.0]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store memories
        memories = [
            Memory(
                id="ctx-1",
                type=MemoryType.DAILY_BRIEF,
                content="Morning brief with action items",
                metadata={"date": "2025-01-15"},
                embedding=[0.8, 0.2, 0.0, 0.0, 0.0],
            ),
            Memory(
                id="ctx-2",
                type=MemoryType.ACTION_ITEM,
                content="Complete project report",
                metadata={"status": "pending"},
                embedding=[0.75, 0.25, 0.0, 0.0, 0.0],
            ),
        ]

        for memory in memories:
            await retriever.memory_store.store_memory(memory)

        # Get relevant context
        context = await retriever.get_relevant_context(
            query="What are my pending tasks?",
            context_window=5
        )

        assert len(context) > 0
        assert all(isinstance(r, RetrievalResult) for r in context)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_recent_memories(self, MockOllama, temp_chroma_dir):
        """Test retrieving recent memories by type."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store memories with different timestamps
        now = datetime.now()
        memories = [
            Memory(
                id="recent-1",
                type=MemoryType.DAILY_BRIEF,
                content="Today's brief",
                metadata={},
                timestamp=now,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            Memory(
                id="recent-2",
                type=MemoryType.DAILY_BRIEF,
                content="Yesterday's brief",
                metadata={},
                timestamp=now - timedelta(days=1),
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            ),
            Memory(
                id="recent-3",
                type=MemoryType.ACTION_ITEM,
                content="Recent task",
                metadata={},
                timestamp=now,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
            ),
        ]

        for memory in memories:
            await retriever.memory_store.store_memory(memory)

        # Get recent daily briefs
        recent = await retriever.get_recent_memories(
            memory_type=MemoryType.DAILY_BRIEF,
            limit=5
        )

        assert len(recent) >= 2
        assert all(m.type == MemoryType.DAILY_BRIEF for m in recent)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_pending_action_items(self, MockOllama, temp_chroma_dir):
        """Test retrieving pending action items."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store action items with different statuses
        pending = ActionItemMemory(
            id="action-pending",
            description="Pending task",
            status=ActionItemStatus.PENDING,
            source="email",
        )
        pending.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        completed = ActionItemMemory(
            id="action-completed",
            description="Completed task",
            status=ActionItemStatus.COMPLETED,
            source="meeting",
        )
        completed.embedding = [0.2, 0.3, 0.4, 0.5, 0.6]

        await retriever.memory_store.store_memory(pending)
        await retriever.memory_store.store_memory(completed)

        # Get only pending action items
        pending_items = await retriever.get_pending_action_items()

        assert len(pending_items) >= 1
        assert all(m.metadata.get("status") == "pending" for m in pending_items)


class TestMemoryManagement:
    """Test memory management utilities."""

    @pytest.mark.asyncio
    async def test_delete_memory(self, temp_chroma_dir):
        """Test deleting a memory."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        memory = Memory(
            id="delete-001",
            type=MemoryType.CONVERSATION,
            content="To delete",
            metadata={},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        await retriever.memory_store.store_memory(memory)

        # Delete
        await retriever.delete_memory("delete-001")

        # Verify deleted
        result = await retriever.memory_store.get_memory_by_id("delete-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_count_memories(self, temp_chroma_dir):
        """Test counting memories."""
        retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store some memories
        for i in range(3):
            memory = Memory(
                id=f"count-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Content {i}",
                metadata={},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
            )
            await retriever.memory_store.store_memory(memory)

        count = await retriever.count_memories()
        assert count >= 3

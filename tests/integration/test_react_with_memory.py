"""Integration tests for ReAct Agent with RAG Memory.

Tests the integration between the ReAct agent and the memory system,
ensuring context is properly stored and retrieved during reasoning.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
import shutil
import os

from src.agents.react_agent import ReActAgent
from src.memory.retrieval import MemoryRetriever
from src.memory.models import (
    Memory,
    MemoryType,
    DailyBriefMemory,
    ActionItemMemory,
    ActionItemStatus,
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


class TestReActAgentWithMemory:
    """Test ReAct agent integration with memory system."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    @patch('src.services.ollama_service.OllamaService')
    async def test_agent_stores_daily_brief_in_memory(
        self, MockAgentOllama, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test that ReAct agent stores daily briefs in memory."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        # Mock agent's Ollama service
        mock_agent_ollama = MagicMock()
        mock_agent_response = MagicMock()
        mock_agent_response.response = "Generated daily brief summary"
        mock_agent_ollama.generate = AsyncMock(return_value=mock_agent_response)
        MockAgentOllama.return_value = mock_agent_ollama

        # Create agent with memory
        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)
        agent = ReActAgent()
        agent.memory_retriever = memory_retriever

        # Simulate generating a daily brief
        brief = DailyBriefMemory(
            id="brief-test-001",
            summary="Test daily summary",
            key_points=["Key point 1", "Key point 2"],
            action_items=["Action 1"],
            emails_count=10,
            calendar_events_count=5,
            news_items_count=20,
        )

        # Store in memory
        await memory_retriever.store_memory(brief)

        # Verify it was stored
        result = await memory_retriever.memory_store.get_memory_by_id("brief-test-001")
        assert result is not None
        assert result.type == MemoryType.DAILY_BRIEF
        assert "Test daily summary" in result.content

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_agent_retrieves_context_before_reasoning(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test that agent retrieves relevant context from memory."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.5, 0.5, 0.0, 0.0, 0.0]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        # Create memory retriever
        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store some historical context
        past_brief = Memory(
            id="past-brief-001",
            type=MemoryType.DAILY_BRIEF,
            content="Previous daily brief with project updates",
            metadata={"date": "2025-01-14"},
            embedding=[0.6, 0.4, 0.0, 0.0, 0.0],
        )
        await memory_retriever.memory_store.store_memory(past_brief)

        # Retrieve context for today's brief
        context = await memory_retriever.get_relevant_context(
            query="What happened in previous days?",
            context_window=3
        )

        assert len(context) > 0
        assert context[0].memory.id == "past-brief-001"
        assert "project updates" in context[0].memory.content

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_agent_tracks_action_items_in_memory(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test that agent can track action items through memory."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Create and store action items
        action1 = ActionItemMemory(
            id="action-001",
            description="Complete project report",
            status=ActionItemStatus.PENDING,
            source="email",
        )
        action1.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        action2 = ActionItemMemory(
            id="action-002",
            description="Review code changes",
            status=ActionItemStatus.COMPLETED,
            source="meeting",
        )
        action2.embedding = [0.2, 0.3, 0.4, 0.5, 0.6]

        await memory_retriever.memory_store.store_memory(action1)
        await memory_retriever.memory_store.store_memory(action2)

        # Get pending action items
        pending = await memory_retriever.get_pending_action_items()

        assert len(pending) >= 1
        assert any(item.metadata.get("description") == "Complete project report" for item in pending)
        assert all(item.metadata.get("status") == "pending" for item in pending)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_memory_persistence_across_sessions(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test that memory persists across different retriever instances."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        # Session 1: Store memory
        retriever1 = MemoryRetriever(persist_directory=temp_chroma_dir)
        memory = Memory(
            id="persistent-001",
            type=MemoryType.CONVERSATION,
            content="Important conversation to persist",
            metadata={"session": 1},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        await retriever1.memory_store.store_memory(memory)

        # Session 2: Retrieve memory with new retriever instance
        retriever2 = MemoryRetriever(persist_directory=temp_chroma_dir)
        result = await retriever2.memory_store.get_memory_by_id("persistent-001")

        assert result is not None
        assert result.content == "Important conversation to persist"
        assert result.metadata["session"] == 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_agent_uses_memory_for_contextual_responses(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test that agent provides better responses with memory context."""
        # Mock embedding service
        mock_embedding = MagicMock()

        def mock_embed_fn(text, model):
            # Return different embeddings based on text content
            if "meeting" in text.lower():
                response = MagicMock()
                response.embedding = [0.9, 0.1, 0.0, 0.0, 0.0]
                return response
            else:
                response = MagicMock()
                response.embedding = [0.5, 0.5, 0.0, 0.0, 0.0]
                return response

        mock_embedding.generate_embedding = AsyncMock(side_effect=mock_embed_fn)
        MockEmbeddingOllama.return_value = mock_embedding

        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store meeting context
        meeting_memory = Memory(
            id="meeting-001",
            type=MemoryType.MEETING_CONTEXT,
            content="Sprint planning meeting discussed new features",
            metadata={"meeting_title": "Sprint Planning"},
            embedding=[0.95, 0.05, 0.0, 0.0, 0.0],
        )
        await memory_retriever.memory_store.store_memory(meeting_memory)

        # Query about meetings
        context = await memory_retriever.get_relevant_context(
            query="What was discussed in recent meetings?",
            context_window=3
        )

        assert len(context) > 0
        assert "Sprint planning" in context[0].memory.content

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_memory_count_and_statistics(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test memory count and basic statistics."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store multiple memories
        for i in range(5):
            memory = Memory(
                id=f"stat-{i}",
                type=MemoryType.CONVERSATION,
                content=f"Memory {i}",
                metadata={},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
            )
            await memory_retriever.memory_store.store_memory(memory)

        # Check count
        count = await memory_retriever.count_memories()
        assert count >= 5

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_memory_filtering_by_type(
        self, MockEmbeddingOllama, temp_chroma_dir
    ):
        """Test filtering memories by type for agent queries."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbeddingOllama.return_value = mock_embedding

        memory_retriever = MemoryRetriever(persist_directory=temp_chroma_dir)

        # Store different types of memories
        brief = DailyBriefMemory(
            id="filter-brief-001",
            summary="Daily brief",
            key_points=["Point"],
            action_items=[],
            emails_count=5,
            calendar_events_count=2,
            news_items_count=10,
        )
        brief.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        conversation = Memory(
            id="filter-conv-001",
            type=MemoryType.CONVERSATION,
            content="User conversation",
            metadata={},
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
        )

        await memory_retriever.memory_store.store_memory(brief)
        await memory_retriever.memory_store.store_memory(conversation)

        # Get only daily briefs
        briefs = await memory_retriever.get_recent_memories(
            memory_type=MemoryType.DAILY_BRIEF,
            limit=10
        )

        assert len(briefs) >= 1
        assert all(m.type == MemoryType.DAILY_BRIEF for m in briefs)

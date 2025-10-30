"""Unit tests for LangGraph Workflow.

Following TDD - write tests first, then implement.
Tests the complete workflow orchestration using LangGraph.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
import shutil
import os

from src.workflows.daily_brief_workflow import (
    DailyBriefWorkflow,
    WorkflowState,
    WorkflowStep,
)


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB, clean up after test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
async def workflow(temp_chroma_dir):
    """Create a DailyBriefWorkflow instance."""
    wf = DailyBriefWorkflow(persist_directory=temp_chroma_dir)
    return wf


class TestWorkflowInitialization:
    """Test workflow initialization."""

    def test_workflow_initialization(self, temp_chroma_dir):
        """Test creating workflow with default parameters."""
        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        assert workflow.memory_retriever is not None
        assert workflow.react_agent is not None
        assert workflow.preference_tracker is not None
        assert workflow.feedback_collector is not None

    def test_workflow_graph_structure(self, temp_chroma_dir):
        """Test that workflow graph is properly constructed."""
        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Workflow should have a compiled graph
        assert workflow.graph is not None


class TestWorkflowState:
    """Test workflow state management."""

    def test_workflow_state_creation(self):
        """Test creating workflow state."""
        state = WorkflowState(
            user_query="Generate my daily brief",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.START,
        )

        assert state["user_query"] == "Generate my daily brief"
        assert state["step"] == WorkflowStep.START

    def test_workflow_state_progression(self):
        """Test state transitions through workflow."""
        state = WorkflowState(
            user_query="Test",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.START,
        )

        # Progress through states
        state["step"] = WorkflowStep.RETRIEVE_CONTEXT
        assert state["step"] == WorkflowStep.RETRIEVE_CONTEXT

        state["step"] = WorkflowStep.FETCH_DATA
        assert state["step"] == WorkflowStep.FETCH_DATA


class TestContextRetrievalNode:
    """Test context retrieval node."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_retrieve_context_from_memory(
        self, MockOllama, temp_chroma_dir
    ):
        """Test retrieving relevant context from memory."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Create initial state
        state = WorkflowState(
            user_query="What happened yesterday?",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.RETRIEVE_CONTEXT,
        )

        # Execute context retrieval
        result = await workflow._retrieve_context(state)

        assert "context" in result
        assert result["step"] == WorkflowStep.FETCH_DATA


class TestDataFetchingNode:
    """Test data fetching node."""

    @pytest.mark.asyncio
    async def test_fetch_emails_calendar_news(self, temp_chroma_dir):
        """Test fetching data from MCP connectors."""
        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Mock MCP connectors
        workflow.email_client = MagicMock()
        workflow.email_client.get_recent_emails = AsyncMock(return_value=[
            {"subject": "Test Email", "from": "test@example.com"}
        ])

        workflow.calendar_client = MagicMock()
        workflow.calendar_client.get_today_events = AsyncMock(return_value=[
            {"title": "Meeting", "time": "10:00 AM"}
        ])

        workflow.news_client = MagicMock()
        workflow.news_client.get_top_stories = AsyncMock(return_value=[
            {"title": "AI News", "source": "TechCrunch"}
        ])

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.FETCH_DATA,
        )

        result = await workflow._fetch_data(state)

        assert len(result["emails"]) > 0
        assert len(result["calendar_events"]) > 0
        assert len(result["news_items"]) > 0
        assert result["step"] == WorkflowStep.LOAD_PREFERENCES


class TestPreferenceLoadingNode:
    """Test preference loading node."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_load_user_preferences(self, MockOllama, temp_chroma_dir):
        """Test loading user preferences."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Store some preferences first
        from src.services.preference_tracker import PreferenceCategory
        await workflow.preference_tracker.store_preference(
            category=PreferenceCategory.CONTENT,
            key="topics",
            value=["AI", "Tech"],
            confidence=0.9
        )

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[{"subject": "Test"}],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.LOAD_PREFERENCES,
        )

        result = await workflow._load_preferences(state)

        assert "preferences" in result
        assert result["step"] == WorkflowStep.AGENT_REASONING


class TestAgentReasoningNode:
    """Test ReAct agent reasoning node."""

    @pytest.mark.asyncio
    @patch('src.services.ollama_service.OllamaService')
    async def test_agent_generates_brief(self, MockOllama, temp_chroma_dir):
        """Test agent reasoning to generate daily brief."""
        # Mock Ollama service for agent
        mock_ollama = MagicMock()
        mock_agent_response = MagicMock()
        mock_agent_response.response = """Thought: I need to create a daily brief.
Action: generate_summary
Observation: Brief generated successfully.
Final Answer: Here is your daily brief for today."""
        mock_ollama.generate = AsyncMock(return_value=mock_agent_response)
        MockOllama.return_value = mock_ollama

        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[{"subject": "Important Email"}],
            calendar_events=[{"title": "Team Meeting"}],
            news_items=[{"title": "AI Breakthrough"}],
            preferences={"topics": ["AI", "Tech"]},
            agent_response="",
            step=WorkflowStep.AGENT_REASONING,
        )

        result = await workflow._agent_reasoning(state)

        assert "agent_response" in result
        assert len(result["agent_response"]) > 0
        assert result["step"] == WorkflowStep.STORE_MEMORY


class TestMemoryStorageNode:
    """Test memory storage node."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_brief_in_memory(self, MockOllama, temp_chroma_dir):
        """Test storing generated brief in memory."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[{"subject": "Test"}],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="Your daily brief: Important updates today.",
            step=WorkflowStep.STORE_MEMORY,
        )

        result = await workflow._store_memory(state)

        assert result["step"] == WorkflowStep.END


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow execution."""

    @pytest.mark.asyncio
    @patch('src.services.ollama_service.OllamaService')
    @patch('src.memory.embedding_service.OllamaService')
    async def test_complete_workflow_execution(
        self, MockEmbedding, MockAgent, temp_chroma_dir
    ):
        """Test running the complete workflow from start to end."""
        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.generate_embedding = AsyncMock(return_value=mock_embed_response)
        MockEmbedding.return_value = mock_embedding

        # Mock agent service
        mock_agent = MagicMock()
        mock_agent_response = MagicMock()
        mock_agent_response.response = "Final Answer: Daily brief generated."
        mock_agent.generate = AsyncMock(return_value=mock_agent_response)
        MockAgent.return_value = mock_agent

        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Mock MCP clients
        workflow.email_client = MagicMock()
        workflow.email_client.get_recent_emails = AsyncMock(return_value=[])
        workflow.calendar_client = MagicMock()
        workflow.calendar_client.get_today_events = AsyncMock(return_value=[])
        workflow.news_client = MagicMock()
        workflow.news_client.get_top_stories = AsyncMock(return_value=[])

        # Execute workflow
        result = await workflow.run("Generate my daily brief")

        assert result is not None
        assert "response" in result


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    @pytest.mark.asyncio
    async def test_workflow_handles_mcp_errors(self, temp_chroma_dir):
        """Test workflow gracefully handles MCP connector errors."""
        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Mock failing MCP client
        workflow.email_client = MagicMock()
        workflow.email_client.get_recent_emails = AsyncMock(
            side_effect=Exception("Email fetch failed")
        )

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.FETCH_DATA,
        )

        # Should handle error gracefully
        result = await workflow._fetch_data(state)
        assert "emails" in result
        # Emails should be empty due to error, but workflow continues
        assert isinstance(result["emails"], list)

    @pytest.mark.asyncio
    async def test_workflow_handles_agent_errors(self, temp_chroma_dir):
        """Test workflow handles agent reasoning errors."""
        workflow = DailyBriefWorkflow(persist_directory=temp_chroma_dir)

        # Mock failing agent
        workflow.react_agent = MagicMock()
        workflow.react_agent.run = AsyncMock(
            side_effect=Exception("Agent failed")
        )

        state = WorkflowState(
            user_query="Generate daily brief",
            context=[],
            emails=[],
            calendar_events=[],
            news_items=[],
            preferences={},
            agent_response="",
            step=WorkflowStep.AGENT_REASONING,
        )

        # Should provide fallback response
        result = await workflow._agent_reasoning(state)
        assert "agent_response" in result

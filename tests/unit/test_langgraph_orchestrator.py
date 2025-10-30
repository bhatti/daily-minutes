"""Unit tests for LangGraph Orchestrator.

Tests the orchestrator workflow method aliases and functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.langgraph_orchestrator import LangGraphOrchestrator


class TestOrchestratorMethodAlias:
    """Test orchestrator method aliases (TDD for UI compatibility)."""

    @pytest.mark.asyncio
    async def test_run_orchestrated_workflow_method_exists(self):
        """Test that run_orchestrated_workflow method exists."""

        with patch('src.services.langgraph_orchestrator.get_ollama_service'), \
             patch('src.services.langgraph_orchestrator.get_mcp_server'), \
             patch('src.services.langgraph_orchestrator.get_rag_service'):

            orchestrator = LangGraphOrchestrator()

            # Verify method exists
            assert hasattr(orchestrator, 'run_orchestrated_workflow')
            assert callable(orchestrator.run_orchestrated_workflow)

    @pytest.mark.asyncio
    async def test_run_orchestrated_workflow_calls_run(self):
        """Test that run_orchestrated_workflow delegates to run method."""

        with patch('src.services.langgraph_orchestrator.get_ollama_service'), \
             patch('src.services.langgraph_orchestrator.get_mcp_server'), \
             patch('src.services.langgraph_orchestrator.get_rag_service'):

            orchestrator = LangGraphOrchestrator()

            # Mock the run method
            orchestrator.run = AsyncMock(return_value={
                "success": True,
                "output": "Test output",
                "execution_time_ms": 100
            })

            # Call the alias method
            result = await orchestrator.run_orchestrated_workflow(
                "Test request",
                {"model": "test-model"}
            )

            # Verify run was called with correct arguments
            orchestrator.run.assert_called_once_with(
                "Test request",
                {"model": "test-model"}
            )

            # Verify result is returned correctly
            assert result["success"] is True
            assert result["output"] == "Test output"

    @pytest.mark.asyncio
    async def test_run_orchestrated_workflow_with_positional_args(self):
        """Test that run_orchestrated_workflow works with positional arguments."""

        with patch('src.services.langgraph_orchestrator.get_ollama_service'), \
             patch('src.services.langgraph_orchestrator.get_mcp_server'), \
             patch('src.services.langgraph_orchestrator.get_rag_service'):

            orchestrator = LangGraphOrchestrator()

            # Mock the run method
            orchestrator.run = AsyncMock(return_value={"success": True})

            # Call with positional args (as UI does)
            await orchestrator.run_orchestrated_workflow("request", {"pref": "value"})

            # Verify run was called
            orchestrator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_orchestrated_workflow_without_preferences(self):
        """Test that run_orchestrated_workflow works without preferences."""

        with patch('src.services.langgraph_orchestrator.get_ollama_service'), \
             patch('src.services.langgraph_orchestrator.get_mcp_server'), \
             patch('src.services.langgraph_orchestrator.get_rag_service'):

            orchestrator = LangGraphOrchestrator()

            # Mock the run method
            orchestrator.run = AsyncMock(return_value={"success": True})

            # Call without preferences
            await orchestrator.run_orchestrated_workflow("Test request")

            # Verify run was called with None preferences
            orchestrator.run.assert_called_once_with("Test request", None)

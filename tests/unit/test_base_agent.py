"""Tests for base agent class."""

import time
from typing import Optional

import pytest

from src.agents.base_agent import AgentResult, AgentState, BaseAgent
from src.core.config import reset_settings
from src.core.logging import setup_logging


class MockAgent(BaseAgent[str]):
    """Mock agent for testing."""
    
    def __init__(self, name: Optional[str] = None, should_fail: bool = False):
        super().__init__(name)
        self.should_fail = should_fail
        self.execute_count = 0
    
    def _execute(self) -> str:
        """Mock execution."""
        self.execute_count += 1
        if self.should_fail:
            raise ValueError("Mock failure")
        return "mock_result"


class TestAgentResult:
    """Tests for AgentResult dataclass."""
    
    def test_successful_result(self):
        """Test creating successful result."""
        result = AgentResult(success=True, data="test_data")
        assert result.success is True
        assert result.data == "test_data"
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.timestamp is not None
    
    def test_failed_result(self):
        """Test creating failed result."""
        result = AgentResult(success=False, error="test_error")
        assert result.success is False
        assert result.data is None
        assert result.error == "test_error"
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        metadata = {"agent": "test", "version": "1.0"}
        result = AgentResult(
            success=True,
            data="test",
            metadata=metadata
        )
        assert result.metadata == metadata
    
    def test_successful_result_without_data_fails(self):
        """Test that successful result requires data."""
        with pytest.raises(ValueError, match="Successful result must have data"):
            AgentResult(success=True)
    
    def test_failed_result_without_error_fails(self):
        """Test that failed result requires error message."""
        with pytest.raises(ValueError, match="Failed result must have error"):
            AgentResult(success=False)


class TestAgentState:
    """Tests for AgentState enum."""
    
    def test_state_values(self):
        """Test state enum values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.SUCCESS.value == "success"
        assert AgentState.FAILED.value == "failed"
        assert AgentState.RETRY.value == "retry"


class TestBaseAgentInitialization:
    """Tests for base agent initialization."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_agent_initialization_with_name(self):
        """Test agent initialization with custom name."""
        agent = MockAgent(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert agent.metrics.total_executions == 0
        assert agent.metrics.last_execution_time is None
    
    def test_agent_initialization_without_name(self):
        """Test agent uses class name by default."""
        agent = MockAgent()
        assert agent.name == "MockAgent"
    
    def test_agent_has_logger(self):
        """Test agent has configured logger."""
        agent = MockAgent()
        assert agent.logger is not None
    
    def test_agent_has_settings(self):
        """Test agent has access to settings."""
        agent = MockAgent()
        assert agent.settings is not None


class TestBaseAgentValidation:
    """Tests for agent validation methods."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_default_validate_inputs(self):
        """Test default input validation returns True."""
        agent = MockAgent()
        assert agent.validate_inputs() is True
    
    def test_default_validate_outputs(self):
        """Test default output validation."""
        agent = MockAgent()
        assert agent.validate_outputs("result") is True
        assert agent.validate_outputs(None) is False


class TestAgentExecution:
    """Tests for agent execution."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_successful_execution(self):
        """Test successful agent execution."""
        agent = MockAgent()
        result = agent.run()
        
        assert result.success is True
        assert result.data == "mock_result"
        assert result.error is None
        assert result.execution_time > 0
        assert agent.state == AgentState.SUCCESS
        assert agent.metrics.total_executions == 1
    
    def test_failed_execution(self):
        """Test failed agent execution."""
        agent = MockAgent(should_fail=True)
        result = agent.run()
        
        assert result.success is False
        assert result.data is None
        assert "Mock failure" in result.error
        assert result.execution_time > 0
        assert agent.state == AgentState.FAILED
    
    def test_execution_tracks_count(self):
        """Test that execution count is tracked."""
        agent = MockAgent()
        
        agent.run()
        assert agent.metrics.total_executions == 1
        
        agent.run()
        assert agent.metrics.total_executions == 2
        
        agent.run()
        assert agent.metrics.total_executions == 3
    
    def test_execution_tracks_time(self):
        """Test that last execution time is tracked."""
        agent = MockAgent()
        assert agent.metrics.last_execution_time is None
        
        agent.run()
        assert agent.metrics.last_execution_time is not None
        first_time = agent.metrics.last_execution_time
        
        time.sleep(0.01)
        agent.run()
        assert agent.metrics.last_execution_time > first_time
    
    def test_result_metadata(self):
        """Test result includes metadata."""
        agent = MockAgent(name="TestAgent")
        result = agent.run()
        
        assert result.metadata["agent_name"] == "TestAgent"
        assert result.metadata["execution_count"] == 1


class TestAgentRetryLogic:
    """Tests for agent retry logic."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_retry_on_failure(self, monkeypatch):
        """Test agent retries on failure."""
        # Set max retries to 2 for faster testing
        monkeypatch.setenv("LLM_MAX_RETRIES", "2")
        reset_settings()

        agent = MockAgent(should_fail=True)
        result = agent.run()

        # Should have retried (initial + 2 retries = 3 executions)
        assert agent.execute_count == 3
        assert result.success is False
    
    def test_retry_with_eventual_success(self):
        """Test agent succeeds after retry."""
        class FlakeyAgent(BaseAgent[str]):
            def __init__(self):
                super().__init__()
                self.attempts = 0
            
            def _execute(self) -> str:
                self.attempts += 1
                if self.attempts < 2:
                    raise ValueError("Temporary failure")
                return "success"
        
        agent = FlakeyAgent()
        result = agent.run()
        
        assert result.success is True
        assert result.data == "success"
        assert agent.attempts == 2


class TestAgentStats:
    """Tests for agent statistics."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_get_stats_initial(self):
        """Test stats for unexecuted agent."""
        agent = MockAgent(name="TestAgent")
        stats = agent.get_stats()
        
        assert stats["name"] == "TestAgent"
        assert stats["state"] == "idle"
        assert stats["execution_count"] == 0
        assert stats["last_execution"] is None
    
    def test_get_stats_after_execution(self):
        """Test stats after execution."""
        agent = MockAgent(name="TestAgent")
        agent.run()
        stats = agent.get_stats()
        
        assert stats["name"] == "TestAgent"
        assert stats["state"] == "success"
        assert stats["execution_count"] == 1
        assert stats["last_execution"] is not None


class TestAgentReset:
    """Tests for agent reset functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_reset_agent(self):
        """Test resetting agent state."""
        agent = MockAgent()
        
        # Execute a few times
        agent.run()
        agent.run()
        
        assert agent.state == AgentState.SUCCESS
        assert agent.metrics.total_executions == 2
        assert agent.metrics.last_execution_time is not None
        
        # Reset
        agent.reset()
        
        assert agent.state == AgentState.IDLE
        assert agent.metrics.total_executions == 0
        assert agent.metrics.last_execution_time is None


class TestAgentCustomValidation:
    """Tests for custom validation."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_custom_input_validation(self):
        """Test agent with custom input validation."""
        class ValidatingAgent(BaseAgent[str]):
            def __init__(self, valid_input: bool = True):
                super().__init__()
                self.valid_input = valid_input
            
            def validate_inputs(self) -> bool:
                return self.valid_input
            
            def _execute(self) -> str:
                return "result"
        
        # Valid input
        agent = ValidatingAgent(valid_input=True)
        result = agent.run()
        assert result.success is True
        
        # Invalid input
        agent = ValidatingAgent(valid_input=False)
        result = agent.run()
        assert result.success is False
        assert "Input validation failed" in result.error
    
    def test_custom_output_validation(self):
        """Test agent with custom output validation."""
        class OutputValidatingAgent(BaseAgent[str]):
            def __init__(self, return_valid: bool = True):
                super().__init__()
                self.return_valid = return_valid
            
            def validate_outputs(self, result: str) -> bool:
                return self.return_valid
            
            def _execute(self) -> str:
                return "result"
        
        # Valid output
        agent = OutputValidatingAgent(return_valid=True)
        result = agent.run()
        assert result.success is True
        
        # Invalid output
        agent = OutputValidatingAgent(return_valid=False)
        result = agent.run()
        assert result.success is False
        assert "Output validation failed" in result.error


class TestAgentIntegration:
    """Integration tests for agent."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_logging(log_level="WARNING", json_format=False)
        reset_settings()
    
    def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle."""
        agent = MockAgent(name="LifecycleTest")
        
        # Initial state
        assert agent.state == AgentState.IDLE
        assert agent.metrics.total_executions == 0
        
        # First execution
        result1 = agent.run()
        assert result1.success is True
        assert agent.state == AgentState.SUCCESS
        assert agent.metrics.total_executions == 1
        
        # Second execution
        result2 = agent.run()
        assert result2.success is True
        assert agent.metrics.total_executions == 2
        
        # Get stats
        stats = agent.get_stats()
        assert stats["execution_count"] == 2
        assert stats["state"] == "success"
        
        # Reset
        agent.reset()
        assert agent.state == AgentState.IDLE
        assert agent.metrics.total_executions == 0

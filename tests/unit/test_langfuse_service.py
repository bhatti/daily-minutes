"""Tests for Langfuse Observability Service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.observability.langfuse_service import (
    LangfuseService,
    get_langfuse_service,
    trace_operation,
    LANGFUSE_AVAILABLE,
)


class TestLangfuseServiceInitialization:
    """Test Langfuse service initialization."""

    @patch.dict("os.environ", {"LANGFUSE_ENABLED": "false"})
    def test_service_disabled_by_default(self):
        """Test service is disabled by default when env var is false."""
        service = LangfuseService()
        assert service.enabled is False
        assert service.client is None

    @patch.dict("os.environ", {"LANGFUSE_ENABLED": "true", "LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"})
    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_service_enabled_from_env(self, mock_langfuse):
        """Test service is enabled when env var is true."""
        service = LangfuseService()
        assert service.enabled is True
        mock_langfuse.assert_called_once()

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_service_enabled_explicitly(self, mock_langfuse):
        """Test service can be enabled explicitly."""
        service = LangfuseService(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
        )
        assert service.enabled is True
        mock_langfuse.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_service_with_custom_host(self, mock_langfuse):
        """Test service with custom Langfuse host."""
        service = LangfuseService(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
            host="https://custom.langfuse.com",
        )
        assert service.enabled is True
        mock_langfuse.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://custom.langfuse.com",
        )

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", False)
    def test_service_disabled_when_langfuse_not_available(self):
        """Test service is disabled when Langfuse library not available."""
        service = LangfuseService(enabled=True)
        assert service.enabled is False
        assert service.client is None


class TestLangfuseTraceWorkflow:
    """Test workflow tracing functionality."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_trace_workflow_when_enabled(self, mock_langfuse):
        """Test workflow tracing when service is enabled."""
        mock_trace = Mock()
        mock_client = Mock()
        mock_client.trace.return_value = mock_trace
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        with service.trace_workflow("test_workflow", user_id="user123", metadata={"key": "value"}) as trace:
            assert trace == mock_trace
            mock_client.trace.assert_called_once_with(
                name="test_workflow",
                user_id="user123",
                metadata={"key": "value"},
            )

        mock_trace.update.assert_called_once_with(status="completed")

    def test_trace_workflow_when_disabled(self):
        """Test workflow tracing when service is disabled."""
        service = LangfuseService(enabled=False)

        with service.trace_workflow("test_workflow") as trace:
            assert trace is None

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_trace_workflow_with_exception(self, mock_langfuse):
        """Test workflow tracing still completes trace even with exception."""
        mock_trace = Mock()
        mock_client = Mock()
        mock_client.trace.return_value = mock_trace
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        with pytest.raises(ValueError):
            with service.trace_workflow("test_workflow") as trace:
                raise ValueError("Test error")

        mock_trace.update.assert_called_once_with(status="completed")


class TestLangfuseLLMTracking:
    """Test LLM call tracking."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_track_llm_call_when_enabled(self, mock_langfuse):
        """Test LLM call tracking when enabled."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        service.track_llm_call(
            name="generate_summary",
            model="llama3.2:3b",
            input_text="Summarize this",
            output_text="Summary here",
            metadata={"temperature": 0.7},
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        mock_client.generation.assert_called_once_with(
            name="generate_summary",
            model="llama3.2:3b",
            input="Summarize this",
            output="Summary here",
            metadata={"temperature": 0.7},
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    def test_track_llm_call_when_disabled(self):
        """Test LLM call tracking when disabled (no-op)."""
        service = LangfuseService(enabled=False)

        # Should not raise any errors
        service.track_llm_call(
            name="test",
            model="test-model",
            input_text="input",
            output_text="output",
        )


class TestLangfuseAgentTracking:
    """Test ReAct agent step tracking."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_track_agent_step_when_enabled(self, mock_langfuse):
        """Test agent step tracking when enabled."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        service.track_agent_step(
            step_num=1,
            thought="I need to fetch emails",
            action="fetch_emails",
            observation="Found 5 emails",
            metadata={"importance": "high"},
        )

        mock_client.span.assert_called_once_with(
            name="agent_step_1",
            input={"thought": "I need to fetch emails", "action": "fetch_emails"},
            output={"observation": "Found 5 emails"},
            metadata={"importance": "high"},
        )

    def test_track_agent_step_when_disabled(self):
        """Test agent step tracking when disabled (no-op)."""
        service = LangfuseService(enabled=False)

        # Should not raise any errors
        service.track_agent_step(
            step_num=1,
            thought="test",
            action="test",
            observation="test",
        )


class TestLangfuseMemoryTracking:
    """Test RAG memory retrieval tracking."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_track_memory_retrieval_when_enabled(self, mock_langfuse):
        """Test memory retrieval tracking when enabled."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        service.track_memory_retrieval(
            query="previous daily briefs",
            results_count=3,
            top_score=0.95,
            metadata={"collection": "daily_briefs"},
        )

        mock_client.span.assert_called_once_with(
            name="memory_retrieval",
            input={"query": "previous daily briefs"},
            output={"results_count": 3, "top_score": 0.95},
            metadata={"collection": "daily_briefs"},
        )

    def test_track_memory_retrieval_when_disabled(self):
        """Test memory retrieval tracking when disabled (no-op)."""
        service = LangfuseService(enabled=False)

        # Should not raise any errors
        service.track_memory_retrieval(
            query="test",
            results_count=1,
            top_score=0.5,
        )


class TestLangfuseFeedbackTracking:
    """Test user feedback tracking."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_track_feedback_when_enabled(self, mock_langfuse):
        """Test feedback tracking when enabled."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        service.track_feedback(
            trace_id="trace-123",
            score=0.9,
            comment="Great summary!",
            metadata={"rating": "positive"},
        )

        mock_client.score.assert_called_once_with(
            trace_id="trace-123",
            name="user_feedback",
            value=0.9,
            comment="Great summary!",
            metadata={"rating": "positive"},
        )

    def test_track_feedback_when_disabled(self):
        """Test feedback tracking when disabled (no-op)."""
        service = LangfuseService(enabled=False)

        # Should not raise any errors
        service.track_feedback(
            trace_id="test",
            score=0.5,
        )


class TestLangfuseTraceURL:
    """Test trace URL generation."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_get_trace_url_when_enabled(self, mock_langfuse):
        """Test trace URL generation when enabled."""
        mock_client = Mock()
        mock_client.host = "https://cloud.langfuse.com/"
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        url = service.get_trace_url("trace-123")
        assert url == "https://cloud.langfuse.com/trace/trace-123"

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_get_trace_url_custom_host(self, mock_langfuse):
        """Test trace URL with custom host."""
        mock_client = Mock()
        mock_client.host = "https://custom.langfuse.com"
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        url = service.get_trace_url("trace-456")
        assert url == "https://custom.langfuse.com/trace/trace-456"

    def test_get_trace_url_when_disabled(self):
        """Test trace URL returns None when disabled."""
        service = LangfuseService(enabled=False)

        url = service.get_trace_url("trace-123")
        assert url is None


class TestLangfuseFlush:
    """Test flushing pending events."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_flush_when_enabled(self, mock_langfuse):
        """Test flush when enabled."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")
        service.flush()

        mock_client.flush.assert_called_once()

    def test_flush_when_disabled(self):
        """Test flush when disabled (no-op)."""
        service = LangfuseService(enabled=False)

        # Should not raise any errors
        service.flush()


class TestLangfuseSingleton:
    """Test singleton pattern for Langfuse service."""

    def test_get_langfuse_service_returns_singleton(self):
        """Test get_langfuse_service returns same instance."""
        # Reset singleton
        import src.observability.langfuse_service as module
        module._langfuse_service = None

        service1 = get_langfuse_service()
        service2 = get_langfuse_service()

        assert service1 is service2

    def test_get_langfuse_service_creates_instance(self):
        """Test get_langfuse_service creates service if not exists."""
        # Reset singleton
        import src.observability.langfuse_service as module
        module._langfuse_service = None

        service = get_langfuse_service()

        assert service is not None
        assert isinstance(service, LangfuseService)


class TestTraceOperationDecorator:
    """Test trace_operation decorator."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.observe")
    def test_trace_operation_decorator_when_available(self, mock_observe):
        """Test decorator when Langfuse is available."""
        mock_observe.return_value = lambda f: f

        @trace_operation("test_operation")
        def test_func():
            return "result"

        mock_observe.assert_called_once_with(name="test_operation")
        assert test_func() == "result"

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", False)
    def test_trace_operation_decorator_when_not_available(self):
        """Test decorator falls back when Langfuse not available."""
        @trace_operation("test_operation")
        def test_func():
            return "result"

        # Should work as pass-through
        assert test_func() == "result"


class TestLangfuseIntegration:
    """Integration tests for Langfuse service."""

    @patch("src.observability.langfuse_service.LANGFUSE_AVAILABLE", True)
    @patch("src.observability.langfuse_service.Langfuse")
    def test_complete_workflow_tracking(self, mock_langfuse):
        """Test complete workflow with all tracking features."""
        mock_client = Mock()
        mock_trace = Mock()
        mock_client.trace.return_value = mock_trace
        mock_langfuse.return_value = mock_client

        service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

        # Trace workflow
        with service.trace_workflow("daily_brief_generation"):
            # Track memory retrieval
            service.track_memory_retrieval(
                query="previous briefs",
                results_count=2,
                top_score=0.92,
            )

            # Track agent steps
            service.track_agent_step(
                step_num=1,
                thought="Need to fetch data",
                action="fetch_emails",
                observation="5 emails retrieved",
            )

            # Track LLM call
            service.track_llm_call(
                name="generate_brief",
                model="llama3.2:3b",
                input_text="Generate brief",
                output_text="Brief content",
            )

        # Track feedback
        service.track_feedback(
            trace_id="trace-123",
            score=0.95,
            comment="Excellent!",
        )

        # Flush events
        service.flush()

        # Verify all calls made
        mock_client.trace.assert_called_once()
        mock_client.span.assert_called()
        mock_client.generation.assert_called_once()
        mock_client.score.assert_called_once()
        mock_client.flush.assert_called_once()

    def test_graceful_degradation_when_disabled(self):
        """Test service gracefully handles all operations when disabled."""
        service = LangfuseService(enabled=False)

        # All operations should work without errors
        with service.trace_workflow("test"):
            service.track_memory_retrieval("query", 1, 0.5)
            service.track_agent_step(1, "thought", "action", "obs")
            service.track_llm_call("name", "model", "in", "out")

        service.track_feedback("trace", 0.5)
        service.flush()
        url = service.get_trace_url("trace")

        # Should complete without errors
        assert url is None

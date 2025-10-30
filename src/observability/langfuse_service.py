"""Langfuse Observability Service.

This service integrates Langfuse for tracking AI operations including:
- LLM calls and token usage
- Agent reasoning steps
- Memory retrievals
- Workflow execution
- User feedback
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import os
from contextlib import contextmanager

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create dummy decorators for when Langfuse is not available
    def observe(func=None, **kwargs):
        """Dummy observe decorator when Langfuse not available."""
        def decorator(f):
            return f
        return decorator if func is None else decorator(func)

    class DummyContext:
        def update_current_trace(self, **kwargs):
            pass
        def update_current_observation(self, **kwargs):
            pass

    langfuse_context = DummyContext()


class LangfuseService:
    """Service for tracking AI operations with Langfuse.

    Features:
    - Trace workflow executions
    - Monitor LLM calls and token usage
    - Track agent reasoning steps
    - Record memory retrievals
    - Collect user feedback
    """

    def __init__(
        self,
        enabled: bool = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """Initialize Langfuse service.

        Args:
            enabled: Enable/disable Langfuse tracking (default: check env)
            public_key: Langfuse public key (default: from env)
            secret_key: Langfuse secret key (default: from env)
            host: Langfuse host URL (default: from env or cloud)
        """
        # Determine if Langfuse should be enabled
        if enabled is None:
            enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

        self.enabled = enabled and LANGFUSE_AVAILABLE

        if self.enabled:
            # Initialize Langfuse client
            self.client = Langfuse(
                public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        else:
            self.client = None

    @contextmanager
    def trace_workflow(
        self,
        name: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing workflow execution.

        Args:
            name: Workflow name
            user_id: User identifier
            metadata: Additional metadata

        Yields:
            Trace context
        """
        if self.enabled and self.client:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                metadata=metadata or {},
            )
            try:
                yield trace
            finally:
                trace.update(status="completed")
        else:
            yield None

    def track_llm_call(
        self,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
    ):
        """Track an LLM API call.

        Args:
            name: Operation name
            model: Model identifier
            input_text: Input prompt
            output_text: Generated output
            metadata: Additional metadata
            usage: Token usage stats
        """
        if self.enabled and self.client:
            self.client.generation(
                name=name,
                model=model,
                input=input_text,
                output=output_text,
                metadata=metadata or {},
                usage=usage,
            )

    def track_agent_step(
        self,
        step_num: int,
        thought: str,
        action: str,
        observation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a ReAct agent reasoning step.

        Args:
            step_num: Step number in reasoning chain
            thought: Agent's thought
            action: Action taken
            observation: Observation from action
            metadata: Additional metadata
        """
        if self.enabled and self.client:
            self.client.span(
                name=f"agent_step_{step_num}",
                input={"thought": thought, "action": action},
                output={"observation": observation},
                metadata=metadata or {},
            )

    def track_memory_retrieval(
        self,
        query: str,
        results_count: int,
        top_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track RAG memory retrieval.

        Args:
            query: Search query
            results_count: Number of results
            top_score: Highest relevance score
            metadata: Additional metadata
        """
        if self.enabled and self.client:
            self.client.span(
                name="memory_retrieval",
                input={"query": query},
                output={
                    "results_count": results_count,
                    "top_score": top_score
                },
                metadata=metadata or {},
            )

    def track_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track user feedback.

        Args:
            trace_id: Associated trace ID
            score: Feedback score (0.0 to 1.0)
            comment: Optional comment
            metadata: Additional metadata
        """
        if self.enabled and self.client:
            self.client.score(
                trace_id=trace_id,
                name="user_feedback",
                value=score,
                comment=comment,
                metadata=metadata or {},
            )

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """Get URL for viewing a trace in Langfuse.

        Args:
            trace_id: Trace identifier

        Returns:
            URL string or None
        """
        if self.enabled and self.client:
            host = self.client.host.rstrip("/")
            return f"{host}/trace/{trace_id}"
        return None

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self.client:
            self.client.flush()


# Singleton instance
_langfuse_service: Optional[LangfuseService] = None


def get_langfuse_service() -> LangfuseService:
    """Get singleton Langfuse service instance.

    Returns:
        LangfuseService instance
    """
    global _langfuse_service

    if _langfuse_service is None:
        _langfuse_service = LangfuseService()

    return _langfuse_service


# Decorator for automatic tracing
def trace_operation(name: str):
    """Decorator for tracing operations.

    Args:
        name: Operation name

    Returns:
        Decorated function
    """
    if LANGFUSE_AVAILABLE:
        return observe(name=name)
    else:
        def decorator(func):
            return func
        return decorator

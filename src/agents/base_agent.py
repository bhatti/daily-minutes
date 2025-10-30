"""Enhanced base agent class with lifecycle hooks, timeouts, and improved extensibility.

This module provides:
- Abstract base agent class
- Lifecycle hooks for extensibility
- Timeout support
- Agent-specific configuration
- Retry logic with exponential backoff
- State management
- Execution tracking
"""

import signal
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Type variable for agent output
T = TypeVar("T")


class AgentState(Enum):
    """Agent execution states."""
    
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    TIMEOUT = "timeout"


@dataclass
class AgentConfig:
    """Configuration for individual agents.
    
    Allows per-agent configuration overrides while maintaining
    sensible defaults from global settings.
    """
    
    max_retries: int = 3
    timeout: Optional[float] = None  # Timeout in seconds, None = no timeout
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    log_level: Optional[str] = None  # Override global log level
    
    # Advanced options
    retry_on_timeout: bool = True
    exponential_backoff_multiplier: float = 1.0
    exponential_backoff_min: float = 2.0
    exponential_backoff_max: float = 10.0


@dataclass
class AgentResult(Generic[T]):
    """Result from agent execution."""
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate result state."""
        if self.success and self.data is None:
            raise ValueError("Successful result must have data")
        if not self.success and self.error is None:
            raise ValueError("Failed result must have error message")


@dataclass
class AgentMetrics:
    """Detailed agent execution metrics."""
    
    name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_count: int = 0
    retry_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    
    def update_success(self, execution_time: float):
        """Update metrics after successful execution."""
        self.total_executions += 1
        self.successful_executions += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.last_execution_time = datetime.now()
        self.last_success_time = datetime.now()
    
    def update_failure(self, execution_time: float, is_timeout: bool = False):
        """Update metrics after failed execution."""
        self.total_executions += 1
        self.failed_executions += 1
        if is_timeout:
            self.timeout_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.last_execution_time = datetime.now()
        self.last_failure_time = datetime.now()
    
    def update_retry(self):
        """Update retry count."""
        self.retry_count += 1


class TimeoutError(Exception):
    """Exception raised when agent execution times out."""
    pass


class BaseAgent(ABC, Generic[T]):
    """Enhanced base class for all agents.
    
    Provides:
    - Configuration access (global and agent-specific)
    - Structured logging
    - Error handling with lifecycle hooks
    - Retry logic
    - Timeout support
    - State management
    - Detailed execution tracking
    - Dependency injection
    
    Lifecycle Hooks (override to customize):
    - before_execute(): Called before execution starts
    - after_execute(result): Called after successful execution
    - on_error(error): Called when an error occurs
    - on_retry(attempt, error): Called before each retry
    - on_timeout(): Called when execution times out
    
    Examples:
        >>> class MyAgent(BaseAgent[str]):
        ...     def _execute(self) -> str:
        ...         return "Hello, World!"
        ...     
        ...     def before_execute(self):
        ...         self.logger.info("Starting my custom logic")
        ...
        >>> agent = MyAgent(name="test", config=AgentConfig(timeout=5.0))
        >>> result = agent.run()
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        settings: Optional[Any] = None,
    ):
        """Initialize base agent.
        
        Args:
            name: Agent name (defaults to class name)
            config: Agent-specific configuration
            dependencies: Dictionary of injected dependencies
            settings: Global settings (defaults to get_settings())
        """
        self.name = name or self.__class__.__name__
        self.config = config or AgentConfig()
        self.dependencies = dependencies or {}
        
        # Setup logger
        self.logger = structlog.get_logger(self.name)
        if self.config.log_level:
            # TODO: Set logger level if needed
            pass
        
        # Get global settings
        if settings is None:
            try:
                from ..core.config import get_settings
                self.settings = get_settings()
            except ImportError:
                self.settings = None
        else:
            self.settings = settings
        
        # State management
        self.state = AgentState.IDLE
        
        # Metrics
        self.metrics = AgentMetrics(name=self.name)
        
        self.logger.info(
            "agent_initialized",
            agent_name=self.name,
            config=str(self.config),
        )
    
    # ============ Core Abstract Methods ============
    
    @abstractmethod
    def _execute(self) -> T:
        """Execute agent's main logic.
        
        This method must be implemented by subclasses.
        
        Returns:
            Agent-specific result data
            
        Raises:
            Exception: Any error during execution
        """
        pass
    
    # ============ Lifecycle Hooks ============
    
    def before_execute(self) -> None:
        """Hook called before execution starts.
        
        Override this method to add custom pre-execution logic:
        - Initialize resources
        - Validate preconditions
        - Setup temporary state
        - Log custom information
        
        Examples:
            >>> def before_execute(self):
            ...     self.logger.info("Acquiring database connection")
            ...     self.db = self.get_dependency("database")
        """
        pass
    
    def after_execute(self, result: T) -> T:
        """Hook called after successful execution.
        
        Override this method to:
        - Transform the result
        - Log custom metrics
        - Cleanup resources
        - Cache results
        
        Args:
            result: Execution result
            
        Returns:
            Potentially modified result
            
        Examples:
            >>> def after_execute(self, result):
            ...     self.logger.info("Caching result")
            ...     self.cache.set(self.cache_key, result)
            ...     return result
        """
        return result
    
    def on_error(self, error: Exception) -> None:
        """Hook called when an error occurs.
        
        Override this method to:
        - Log detailed error information
        - Send alerts
        - Cleanup failed state
        - Record metrics
        
        Args:
            error: Exception that occurred
            
        Examples:
            >>> def on_error(self, error):
            ...     if isinstance(error, DatabaseError):
            ...         self.send_alert("Database connection failed")
        """
        pass
    
    def on_retry(self, attempt: int, error: Exception) -> None:
        """Hook called before each retry attempt.
        
        Override this method to:
        - Log retry information
        - Adjust retry strategy
        - Update state before retry
        - Send notifications
        
        Args:
            attempt: Current retry attempt number (1-indexed)
            error: Exception that caused the retry
            
        Examples:
            >>> def on_retry(self, attempt, error):
            ...     self.logger.warning(
            ...         "Retrying after error",
            ...         attempt=attempt,
            ...         error_type=type(error).__name__
            ...     )
        """
        self.metrics.update_retry()
    
    def on_timeout(self) -> None:
        """Hook called when execution times out.
        
        Override this method to:
        - Log timeout information
        - Cleanup resources
        - Send alerts
        - Update metrics
        
        Examples:
            >>> def on_timeout(self):
            ...     self.logger.error("Execution timed out, cleaning up")
            ...     self.cleanup_resources()
        """
        pass
    
    # ============ Validation Hooks ============
    
    def validate_inputs(self) -> bool:
        """Validate agent inputs before execution.
        
        Override this method to add custom validation.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        return True
    
    def validate_outputs(self, result: T) -> bool:
        """Validate agent outputs after execution.
        
        Override this method to add custom validation.
        
        Args:
            result: Result to validate
            
        Returns:
            True if result is valid, False otherwise
        """
        return result is not None
    
    # ============ Dependency Management ============
    
    def get_dependency(self, key: str, default: Any = None) -> Any:
        """Get an injected dependency.
        
        Args:
            key: Dependency key
            default: Default value if not found
            
        Returns:
            Dependency value or default
            
        Examples:
            >>> db = self.get_dependency("database")
            >>> cache = self.get_dependency("cache", default=NullCache())
        """
        return self.dependencies.get(key, default)
    
    def set_dependency(self, key: str, value: Any) -> None:
        """Set a dependency at runtime.
        
        Args:
            key: Dependency key
            value: Dependency value
        """
        self.dependencies[key] = value
    
    # ============ Timeout Support ============
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        """Context manager for timeout handling.
        
        Args:
            timeout_seconds: Timeout in seconds
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        def timeout_handler(signum, frame):
            self.on_timeout()
            raise TimeoutError(f"Execution exceeded {timeout_seconds} seconds")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            yield
        finally:
            # Cancel timeout
            signal.alarm(0)
    
    # ============ Retry Logic ============
    
    def _create_retry_decorator(self):
        """Create retry decorator with configured settings."""
        max_retries = self.config.max_retries
        
        def before_sleep_callback(retry_state):
            """Called before each retry sleep."""
            attempt = retry_state.attempt_number
            exception = retry_state.outcome.exception()
            
            self.logger.warning(
                "retrying_execution",
                attempt=attempt,
                max_attempts=max_retries,
                error=str(exception),
            )
            
            # Call lifecycle hook
            self.on_retry(attempt, exception)
        
        return retry(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(
                multiplier=self.config.exponential_backoff_multiplier,
                min=self.config.exponential_backoff_min,
                max=self.config.exponential_backoff_max,
            ),
            reraise=True,
            before_sleep=before_sleep_callback,
        )
    
    # ============ Main Execution ============
    
    def _execute_with_hooks(self) -> T:
        """Execute with lifecycle hooks."""
        # Before hook
        self.before_execute()
        
        # Main execution
        result = self._execute()
        
        # After hook
        result = self.after_execute(result)
        
        return result
    
    def run(self, timeout_override: Optional[float] = None, **kwargs) -> AgentResult[T]:
        """Run the agent with error handling and retry logic.
        
        Args:
            timeout_override: Override configured timeout
            **kwargs: Additional parameters passed to lifecycle hooks
            
        Returns:
            AgentResult with execution status and data
            
        Examples:
            >>> result = agent.run(timeout_override=30.0)
            >>> if result.success:
            ...     print(f"Data: {result.data}")
            ... else:
            ...     print(f"Error: {result.error}")
        """
        start_time = time.time()
        self.state = AgentState.RUNNING
        
        # Determine timeout
        timeout = timeout_override or self.config.timeout
        
        self.logger.info(
            "agent_started",
            execution_count=self.metrics.total_executions + 1,
            timeout=timeout,
        )
        
        try:
            # Validate inputs
            if not self.validate_inputs():
                raise ValueError("Input validation failed")
            
            # Execute with optional timeout
            if timeout:
                with self._timeout_context(timeout):
                    # Execute with retry logic
                    retry_decorator = self._create_retry_decorator()
                    result_data = retry_decorator(self._execute_with_hooks)()
            else:
                # Execute with retry logic (no timeout)
                retry_decorator = self._create_retry_decorator()
                result_data = retry_decorator(self._execute_with_hooks)()
            
            # Validate outputs
            if not self.validate_outputs(result_data):
                raise ValueError("Output validation failed")
            
            # Success
            execution_time = time.time() - start_time
            self.state = AgentState.SUCCESS
            self.metrics.update_success(execution_time)
            
            self.logger.info(
                "agent_completed",
                execution_time=execution_time,
                state=self.state.value,
            )
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_name": self.name,
                    "execution_count": self.metrics.total_executions,
                },
                execution_time=execution_time,
            )
            
        except TimeoutError as e:
            # Timeout
            execution_time = time.time() - start_time
            self.state = AgentState.TIMEOUT
            error_msg = str(e)
            
            self.metrics.update_failure(execution_time, is_timeout=True)
            
            self.logger.error(
                "agent_timeout",
                error=error_msg,
                execution_time=execution_time,
                state=self.state.value,
            )
            
            return AgentResult(
                success=False,
                error=error_msg,
                metadata={
                    "agent_name": self.name,
                    "execution_count": self.metrics.total_executions,
                    "timeout": True,
                },
                execution_time=execution_time,
            )
            
        except Exception as e:
            # Failure
            execution_time = time.time() - start_time
            self.state = AgentState.FAILED
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.metrics.update_failure(execution_time)
            
            # Call error hook
            self.on_error(e)
            
            self.logger.error(
                "agent_failed",
                error=error_msg,
                execution_time=execution_time,
                state=self.state.value,
                exc_info=True,
            )
            
            return AgentResult(
                success=False,
                error=error_msg,
                metadata={
                    "agent_name": self.name,
                    "execution_count": self.metrics.total_executions,
                },
                execution_time=execution_time,
            )
    
    # ============ Metrics and Stats ============
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic agent statistics.
        
        Returns:
            Dictionary with agent stats
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "execution_count": self.metrics.total_executions,
            "last_execution": self.metrics.last_execution_time.isoformat() 
                if self.metrics.last_execution_time else None,
        }
    
    def get_detailed_metrics(self) -> AgentMetrics:
        """Get detailed agent metrics.
        
        Returns:
            AgentMetrics object with detailed statistics
        """
        return self.metrics
    
    # ============ State Management ============
    
    def reset(self):
        """Reset agent state and metrics."""
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics(name=self.name)
        self.logger.info("agent_reset")


# Example agent demonstrating lifecycle hooks
class ExampleAgent(BaseAgent[Dict[str, Any]]):
    """Example agent showing how to use lifecycle hooks."""
    
    def before_execute(self):
        """Custom pre-execution logic."""
        self.logger.info("Starting example agent execution")
        self.start_time = time.time()
    
    def _execute(self) -> Dict[str, Any]:
        """Main execution logic."""
        time.sleep(0.5)  # Simulate work
        return {
            "status": "completed",
            "message": "Example agent executed successfully",
        }
    
    def after_execute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-execution processing."""
        elapsed = time.time() - self.start_time
        result["elapsed_time"] = elapsed
        self.logger.info("Example agent completed", elapsed=elapsed)
        return result
    
    def on_error(self, error: Exception):
        """Custom error handling."""
        self.logger.error(
            "Example agent failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    
    def on_retry(self, attempt: int, error: Exception):
        """Custom retry logic."""
        super().on_retry(attempt, error)
        self.logger.warning(f"Retry attempt {attempt} after {type(error).__name__}")


if __name__ == "__main__":
    # Example usage
    print("Testing Enhanced BaseAgent")
    print("=" * 50)
    
    # Test with default configuration
    agent = ExampleAgent(name="test-agent")
    result = agent.run()
    print(f"Result: {result.success}, Data: {result.data}")
    print(f"Metrics: {agent.get_stats()}")
    
    # Test with timeout
    print("\nTesting with timeout configuration")
    agent2 = ExampleAgent(
        name="timeout-agent",
        config=AgentConfig(timeout=10.0, max_retries=2)
    )
    result2 = agent2.run()
    print(f"Result: {result2.success}")
    
    # Test detailed metrics
    print("\nDetailed Metrics:")
    metrics = agent2.get_detailed_metrics()
    print(f"  Total executions: {metrics.total_executions}")
    print(f"  Successful: {metrics.successful_executions}")
    print(f"  Failed: {metrics.failed_executions}")
    print(f"  Average time: {metrics.average_execution_time:.3f}s")

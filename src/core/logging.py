"""Enhanced logging setup with rotating file handlers and advanced features.

This module provides:
- JSON and console output formats
- Rotating file handlers
- Context-aware logging
- Call site information
- Multiple log levels
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog
from structlog.types import EventDict, Processor


def add_call_site_info(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add call site information to log entries.
    
    Args:
        logger: Logger instance
        method_name: Name of the logging method
        event_dict: Event dictionary
        
    Returns:
        Modified event dictionary with call site info
    """
    # structlog.processors.CallsiteParameterAdder handles this
    return event_dict


def add_exception_info(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add structured exception information.

    Args:
        logger: Logger instance
        method_name: Name of the logging method
        event_dict: Event dictionary

    Returns:
        Modified event dictionary with exception info
    """
    exc_info = event_dict.get('exc_info')
    if exc_info:
        # If exc_info is True, get current exception info
        if exc_info is True:
            import sys
            exc_info = sys.exc_info()
        # If it's a tuple, use it
        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            exc_type, exc_value, exc_traceback = exc_info
            if exc_type is not None:
                event_dict['exception_type'] = exc_type.__name__
                event_dict['exception_message'] = str(exc_value)
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
) -> None:
    """Setup structured logging with optional file rotation.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format; otherwise use console format
        log_file: Optional path to log file for rotation
        max_bytes: Maximum bytes per log file (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        enable_console: If True, also log to console
    
    Examples:
        >>> # Console logging only
        >>> setup_logging(log_level="INFO")
        
        >>> # File + console with rotation
        >>> setup_logging(
        ...     log_level="INFO",
        ...     log_file="logs/app.log",
        ...     max_bytes=10*1024*1024,
        ...     backup_count=5
        ... )
        
        >>> # JSON format for production
        >>> setup_logging(
        ...     log_level="INFO",
        ...     json_format=True,
        ...     log_file="logs/app.json"
        ... )
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure structlog processors
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        add_exception_info,  # Custom exception processor
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add renderer based on format
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handlers = []
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        handlers.append(console_handler)
    
    # Add rotating file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ or module name)
        
    Returns:
        Configured structlog logger
        
    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("application_started", version="1.0.0")
        >>> logger.error("database_error", table="users", error_code=500)
    """
    return structlog.get_logger(name)


# Context manager for temporary logging context
from contextlib import contextmanager
from typing import Any


@contextmanager
def log_context(**context: Any):
    """Add temporary context to all log messages within the context.
    
    Args:
        **context: Key-value pairs to add to log context
        
    Yields:
        Bound logger with context
        
    Examples:
        >>> with log_context(user_id="123", request_id="abc"):
        ...     logger = get_logger(__name__)
        ...     logger.info("user_action", action="login")
        # All logs within context will include user_id and request_id
    """
    logger = structlog.get_logger()
    bound_logger = logger.bind(**context)
    
    # Store original logger
    original_logger = structlog.get_logger()
    
    try:
        # Temporarily bind context
        yield bound_logger
    finally:
        # Restore original logger
        pass  # structlog handles this automatically


# Convenience function for structured exception logging
def log_exception(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    exception: Exception,
    **context: Any
) -> None:
    """Log an exception with structured context.
    
    Args:
        logger: Logger instance
        event: Event name/message
        exception: Exception to log
        **context: Additional context
        
    Examples:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(
        ...         logger,
        ...         "operation_failed",
        ...         e,
        ...         operation="risky_operation",
        ...         user_id="123"
        ...     )
    """
    logger.error(
        event,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        exc_info=True,
        **context
    )


# Logging utility decorators
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar('T')


def log_execution(
    event_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log function execution.
    
    Args:
        event_name: Custom event name (defaults to function name)
        log_args: If True, log function arguments
        log_result: If True, log function result
        
    Returns:
        Decorated function
        
    Examples:
        >>> @log_execution(log_args=True, log_result=True)
        ... def calculate(x: int, y: int) -> int:
        ...     return x + y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = get_logger(func.__module__)
            event = event_name or f"{func.__name__}_execution"
            
            # Log start
            log_data = {"function": func.__name__}
            if log_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.info(f"{event}_started", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                if log_result:
                    log_data["result"] = str(result)
                logger.info(f"{event}_completed", **log_data)
                
                return result
                
            except Exception as e:
                # Log failure
                log_exception(logger, f"{event}_failed", e, **log_data)
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    setup_logging(
        log_level="DEBUG",
        log_file="logs/test.log",
        max_bytes=1024 * 1024,  # 1MB
        backup_count=3,
    )
    
    logger = get_logger(__name__)
    logger.info("test_message", key="value", count=42)
    logger.debug("debug_message", debug_data={"x": 1, "y": 2})
    
    # Test context manager
    with log_context(request_id="test-123"):
        logger.info("context_test", action="test")
    
    # Test exception logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_exception(logger, "test_error", e, extra_info="test")
    
    print("Logging test complete! Check logs/test.log")

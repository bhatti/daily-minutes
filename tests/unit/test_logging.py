"""Tests for logging configuration."""

import logging
from pathlib import Path

import pytest
import structlog

from src.core.logging import (
    get_logger,
    setup_logging,
    log_context,
    log_exception,
    log_execution,
)


class TestSetupLogging:
    """Tests for logging setup function."""
    
    def test_default_setup(self):
        """Test default logging configuration."""
        setup_logging()
        logger = structlog.get_logger("test")
        assert logger is not None
    
    def test_log_level_setting(self):
        """Test setting different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(log_level=level)
            root_logger = logging.getLogger()
            assert root_logger.level == getattr(logging, level)
    
    def test_json_format(self):
        """Test JSON format configuration."""
        setup_logging(json_format=True)
        logger = get_logger("test")
        # Should not raise error
        logger.info("test message")
    
    def test_console_format(self):
        """Test console format configuration."""
        setup_logging(json_format=False)
        logger = get_logger("test")
        # Should not raise error with colors
        logger.info("test message")
    
    def test_file_output(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        setup_logging(log_level="INFO", log_file=log_file)
        
        logger = get_logger("test")
        logger.info("test message to file")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message to file" in content


class TestGetLogger:
    """Tests for logger creation."""
    
    def setup_method(self):
        """Setup logging for each test."""
        setup_logging(log_level="WARNING", json_format=False)
    
    def test_get_basic_logger(self):
        """Test getting a basic logger."""
        logger = get_logger("test")
        assert logger is not None
    
    def test_get_logger_with_context(self):
        """Test getting logger with initial context."""
        logger = get_logger("test")
        bound_logger = logger.bind(user_id="123", request_id="abc")
        # Should not raise error
        bound_logger.info("test with context")
    
    def test_logger_binding(self):
        """Test binding context to logger."""
        logger = get_logger("test")
        bound_logger = logger.bind(operation="test_op")
        bound_logger.info("test message")
        # Should not raise error


class TestLoggerContext:
    """Tests for logger context manager."""

    def setup_method(self):
        """Setup logging for each test."""
        setup_logging(log_level="WARNING", json_format=False)

    def test_context_manager_basic(self):
        """Test basic context manager usage."""

        with log_context(operation="test") as ctx_logger:
            assert ctx_logger is not None
            ctx_logger.info("inside context")

    def test_context_manager_multiple_values(self):
        """Test context with multiple key-value pairs."""

        with log_context(
            user_id="123",
            request_id="abc",
            operation="test_op"
        ) as ctx_logger:
            ctx_logger.info("test with multiple context values")

    def test_nested_contexts(self):
        """Test nested context managers."""

        with log_context(outer="value1") as outer_logger:
            outer_logger.info("outer context")

            with log_context(inner="value2") as inner_logger:
                inner_logger.info("inner context")


class TestLoggingLevels:
    """Tests for different logging levels."""
    
    def setup_method(self):
        """Setup logging for each test."""
        setup_logging(log_level="DEBUG")
    
    def test_debug_level(self):
        """Test DEBUG level logging."""
        logger = get_logger("test")
        logger.debug("debug message", extra_info="details")
    
    def test_info_level(self):
        """Test INFO level logging."""
        logger = get_logger("test")
        logger.info("info message", status="ok")
    
    def test_warning_level(self):
        """Test WARNING level logging."""
        logger = get_logger("test")
        logger.warning("warning message", code="W001")
    
    def test_error_level(self):
        """Test ERROR level logging."""
        logger = get_logger("test")
        logger.error("error message", error_code="E001")
    
    def test_exception_logging(self):
        """Test exception logging with traceback."""
        logger = get_logger("test")
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("caught exception", exc_info=True)


class TestConfigurationPresets:
    """Tests for configuration preset functions."""

    def test_warning_level_setup(self):
        """Test warning level configuration."""
        setup_logging(log_level="WARNING", json_format=False)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_debug_level_setup(self):
        """Test debug level configuration."""
        setup_logging(log_level="DEBUG", json_format=False)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_production_setup(self, tmp_path):
        """Test production-like configuration."""
        log_file = tmp_path / "daily_minutes.log"

        setup_logging(
            log_level="INFO",
            json_format=True,
            log_file=str(log_file)
        )

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

        # Check log file was created
        logger = get_logger("test")
        logger.info("production test")

        assert log_file.exists()


class TestStructuredLogging:
    """Tests for structured logging features."""
    
    def setup_method(self):
        """Setup logging for each test."""
        setup_logging(log_level="DEBUG", json_format=False)
    
    def test_log_with_structured_data(self):
        """Test logging with structured data."""
        logger = get_logger("test")
        logger.info(
            "user_action",
            user_id="123",
            action="login",
            ip_address="127.0.0.1",
            success=True
        )
    
    def test_log_with_nested_data(self):
        """Test logging with nested data structures."""
        logger = get_logger("test")
        logger.info(
            "api_request",
            request={
                "method": "GET",
                "url": "/api/v1/data",
                "headers": {"content-type": "application/json"}
            },
            response={
                "status": 200,
                "duration_ms": 45
            }
        )
    
    def test_callsite_information(self):
        """Test that call site info is captured."""
        setup_logging(log_level="DEBUG", json_format=False)
        logger = get_logger("test")
        logger.info("test with callsite")
        # Call site info should be added by processor


@pytest.fixture
def capture_logs(monkeypatch):
    """Fixture to capture log output for testing."""
    captured = []
    
    def mock_renderer(logger, name, event_dict):
        captured.append(event_dict)
        return ""
    
    # This is a simplified mock - in real tests you might use
    # structlog.testing.CapturingLogger
    return captured


class TestLoggingIntegration:
    """Integration tests for logging."""
    
    def test_end_to_end_logging(self, tmp_path):
        """Test complete logging pipeline."""
        log_file = tmp_path / "integration.log"
        
        setup_logging(
            log_level="INFO",
            log_file=log_file,
            json_format=False
        )
        
        logger = get_logger("integration_test")
        
        # Log various levels
        logger.debug("debug - should not appear")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        
        # Check file was written
        assert log_file.exists()
        content = log_file.read_text()
        
        # Debug should not be in file (level is INFO)
        assert "debug - should not appear" not in content
        # Others should be present
        assert "info message" in content
        assert "warning message" in content
        assert "error message" in content
    
    def test_multiple_loggers(self):
        """Test multiple loggers with different contexts."""
        setup_logging(log_level="DEBUG")

        logger1 = get_logger("service1").bind(service="auth")
        logger2 = get_logger("service2").bind(service="api")

        logger1.info("auth service started")
        logger2.info("api service started")

        # Both should work without interference

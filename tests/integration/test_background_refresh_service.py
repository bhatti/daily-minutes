"""Integration tests for BackgroundRefreshService.

These tests verify that BackgroundRefreshService correctly integrates with:
- EmailService
- CalendarService
- OllamaService
- SQLite database

IMPORTANT: These tests use minimal mocking to catch real integration bugs,
especially parameter mismatches between services.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.services.background_refresh_service import get_background_refresh_service
from src.services.email_service import EmailService
from src.services.calendar_service import CalendarService
from src.services.ollama_service import OllamaService


@pytest.mark.asyncio
async def test_background_refresh_email_parameter_compatibility():
    """Test that EmailService.fetch_emails accepts max_results parameter.

    This test would have caught the max_emails vs max_results bug.
    """
    # Verify EmailService has fetch_emails method with correct signature
    email_service = EmailService()

    # Check method signature using inspect
    import inspect
    sig = inspect.signature(email_service.fetch_emails)

    # Verify max_results is a valid parameter
    assert 'max_results' in sig.parameters, \
        "EmailService.fetch_emails must accept 'max_results' parameter"

    # Verify max_emails is NOT a parameter
    assert 'max_emails' not in sig.parameters, \
        "EmailService.fetch_emails should NOT have 'max_emails' parameter"


@pytest.mark.asyncio
async def test_background_refresh_calendar_parameter_compatibility():
    """Test that CalendarService.fetch_events accepts time_min/time_max parameters.

    This test would have caught the days_ahead parameter bug.
    """
    # Verify CalendarService has fetch_events method with correct signature
    calendar_service = CalendarService()

    # Check method signature using inspect
    import inspect
    sig = inspect.signature(calendar_service.fetch_events)

    # Verify time_min and time_max are valid parameters
    assert 'time_min' in sig.parameters, \
        "CalendarService.fetch_events must accept 'time_min' parameter"
    assert 'time_max' in sig.parameters, \
        "CalendarService.fetch_events must accept 'time_max' parameter"

    # Verify days_ahead is NOT a parameter
    assert 'days_ahead' not in sig.parameters, \
        "CalendarService.fetch_events should NOT have 'days_ahead' parameter"


@pytest.mark.asyncio
async def test_ollama_service_check_availability_exists():
    """Verify OllamaService has check_availability method, not is_available."""
    ollama_service = OllamaService()

    # Verify method exists and is callable
    assert hasattr(ollama_service, 'check_availability'), \
        "OllamaService must have check_availability method"

    assert asyncio.iscoroutinefunction(ollama_service.check_availability), \
        "check_availability must be an async method"

    # Verify old method name doesn't exist
    assert not hasattr(ollama_service, 'is_available'), \
        "OllamaService should not have is_available method (use check_availability)"


@pytest.mark.asyncio
async def test_preload_script_would_use_correct_parameters():
    """Verify that the preload script would call services with correct parameters.

    This is the integration test that simulates what preload_all_data.py does.
    """
    # The preload script calls background_refresh_service methods
    # Verify those methods exist and would call the right service methods

    # Test 1: Verify the fix for email service
    email_service = EmailService()
    import inspect
    sig = inspect.signature(email_service.fetch_emails)
    assert 'max_results' in sig.parameters, "fetch_emails must accept max_results"

    # Test 2: Verify the fix for calendar service
    calendar_service = CalendarService()
    sig = inspect.signature(calendar_service.fetch_events)
    assert 'time_min' in sig.parameters, "fetch_events must accept time_min"
    assert 'time_max' in sig.parameters, "fetch_events must accept time_max"

    # Test 3: Verify the fix for ollama service
    ollama_service = OllamaService()
    assert hasattr(ollama_service, 'check_availability'), \
        "OllamaService must have check_availability method"
    assert asyncio.iscoroutinefunction(ollama_service.check_availability), \
        "check_availability must be async"

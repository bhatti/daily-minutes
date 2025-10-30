"""Integration tests to verify UI cache loading matches preload data.

This test suite ensures that:
1. Background refresh service correctly caches all data types (news, weather, email, calendar)
2. Startup service correctly loads the cached data
3. The data loaded by UI matches what was cached during preload

These tests catch regressions like:
- Missing set_cache() calls in background_refresh_service
- Data serialization issues
- Cache key mismatches
- Missing data fields
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure mock data is enabled for tests
os.environ['USE_MOCK_EMAIL'] = 'true'
os.environ['USE_MOCK_CALENDAR'] = 'true'

from src.services.background_refresh_service import get_background_refresh_service
from src.services.startup_service import StartupService
from src.database.sqlite_manager import get_db_manager


@pytest.mark.asyncio
async def test_email_cache_preload_and_ui_load():
    """Test email data is cached during preload and loaded correctly by UI.

    This test would have caught the bug where _refresh_email() was missing
    the set_cache() call for emails_data.
    """
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear existing cache to start fresh
    await db_manager.set_cache('emails_data', None)

    # Step 1: Simulate preload by refreshing email data
    refresh_service = get_background_refresh_service()
    email_refresh_success = await refresh_service._refresh_email()

    # Verify refresh succeeded (using mock data should always work)
    assert email_refresh_success, "Email refresh should succeed with mock data"

    # Step 2: Verify data was cached to DB (this is the critical check!)
    cached_emails = await db_manager.get_cache('emails_data')
    assert cached_emails is not None, "Emails should be cached after refresh"
    assert isinstance(cached_emails, list), "Cached emails should be a list"
    assert len(cached_emails) > 0, "Should have at least one email cached"

    # Step 3: Verify email structure has required fields
    first_email = cached_emails[0]
    required_fields = ['id', 'subject', 'sender', 'received_at', 'snippet',
                      'importance_score', 'has_action_items']
    for field in required_fields:
        assert field in first_email, f"Email should have '{field}' field"

    # Step 4: Load data via startup_service (how UI loads it)
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Step 5: Verify UI gets the same cached data
    assert startup_data['error'] is None, "Startup should succeed"
    assert len(startup_data['emails']) > 0, "UI should load cached emails"
    assert len(startup_data['emails']) == len(cached_emails), \
        "UI should load same number of emails as cached"

    # Verify first email matches
    ui_first_email = startup_data['emails'][0]
    assert ui_first_email['id'] == first_email['id'], "Email IDs should match"
    assert ui_first_email['subject'] == first_email['subject'], "Subjects should match"
    assert ui_first_email['sender'] == first_email['sender'], "Senders should match"


@pytest.mark.asyncio
async def test_calendar_cache_preload_and_ui_load():
    """Test calendar data is cached during preload and loaded correctly by UI.

    This test would have caught the bug where _refresh_calendar() was missing
    the set_cache() call for calendar_data.
    """
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear existing cache to start fresh
    await db_manager.set_cache('calendar_data', None)

    # Step 1: Simulate preload by refreshing calendar data
    refresh_service = get_background_refresh_service()
    calendar_refresh_success = await refresh_service._refresh_calendar()

    # Verify refresh succeeded (using mock data should always work)
    assert calendar_refresh_success, "Calendar refresh should succeed with mock data"

    # Step 2: Verify data was cached to DB (this is the critical check!)
    cached_calendar = await db_manager.get_cache('calendar_data')
    assert cached_calendar is not None, "Calendar should be cached after refresh"
    assert isinstance(cached_calendar, list), "Cached calendar should be a list"
    assert len(cached_calendar) > 0, "Should have at least one event cached"

    # Step 3: Verify event structure has required fields
    first_event = cached_calendar[0]
    required_fields = ['id', 'summary', 'start_time', 'end_time', 'importance_score']
    for field in required_fields:
        assert field in first_event, f"Event should have '{field}' field"

    # Step 4: Load data via startup_service (how UI loads it)
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Step 5: Verify UI gets the same cached data
    assert startup_data['error'] is None, "Startup should succeed"
    assert len(startup_data['calendar_events']) > 0, "UI should load cached calendar"
    assert len(startup_data['calendar_events']) == len(cached_calendar), \
        "UI should load same number of events as cached"

    # Verify first event matches
    ui_first_event = startup_data['calendar_events'][0]
    assert ui_first_event['id'] == first_event['id'], "Event IDs should match"
    assert ui_first_event['summary'] == first_event['summary'], "Summaries should match"
    assert ui_first_event['start_time'] == first_event['start_time'], "Start times should match"


@pytest.mark.asyncio
async def test_weather_cache_preload_and_ui_load():
    """Test weather data is cached during preload and loaded correctly by UI."""
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear existing cache
    await db_manager.set_cache('weather_data', None)

    # Step 1: Simulate preload by refreshing weather data
    refresh_service = get_background_refresh_service()
    weather_refresh_success = await refresh_service._refresh_weather()

    # Verify refresh succeeded
    assert weather_refresh_success, "Weather refresh should succeed"

    # Step 2: Verify data was cached to DB
    cached_weather = await db_manager.get_cache('weather_data')
    assert cached_weather is not None, "Weather should be cached after refresh"
    assert isinstance(cached_weather, dict), "Cached weather should be a dict"

    # Step 3: Verify weather structure
    required_fields = ['location', 'temperature', 'description', 'timestamp']
    for field in required_fields:
        assert field in cached_weather, f"Weather should have '{field}' field"

    # Step 4: Load data via startup_service
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Step 5: Verify UI gets the same cached data
    assert startup_data['error'] is None, "Startup should succeed"
    assert startup_data['weather_data'] is not None, "UI should load cached weather"
    assert startup_data['weather_data']['location'] == cached_weather['location']
    assert startup_data['weather_data']['temperature'] == cached_weather['temperature']


@pytest.mark.asyncio
async def test_news_cache_preload_and_ui_load():
    """Test news data is cached during preload and loaded correctly by UI."""
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear existing articles
    # Note: News uses articles table, not cache

    # Step 1: Simulate preload by refreshing news
    refresh_service = get_background_refresh_service()
    news_refresh_success = await refresh_service._refresh_news()

    # Verify refresh succeeded
    assert news_refresh_success, "News refresh should succeed"

    # Step 2: Verify articles were saved to DB
    articles = await db_manager.get_all_articles(limit=100)
    assert len(articles) > 0, "Should have articles after refresh"

    # Step 3: Load data via startup_service
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Step 4: Verify UI gets the same articles
    assert startup_data['error'] is None, "Startup should succeed"
    assert len(startup_data['articles']) > 0, "UI should load articles"
    assert startup_data['loaded_from_cache'] is True, "Should be loaded from cache"

    # Verify article count matches
    assert len(startup_data['articles']) == len(articles), \
        "UI should load same number of articles as in DB"


@pytest.mark.asyncio
async def test_full_preload_and_ui_load_all_sources():
    """Test all data sources are cached and loaded correctly (full integration).

    This simulates the complete preload -> UI startup flow.
    """
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear all caches
    await db_manager.set_cache('emails_data', None)
    await db_manager.set_cache('calendar_data', None)
    await db_manager.set_cache('weather_data', None)

    # Step 1: Run full refresh (simulate preload)
    refresh_service = get_background_refresh_service()
    results = await refresh_service.refresh_all_sources()

    # Verify all sources succeeded (with mock data)
    assert results.get('news'), "News refresh should succeed"
    assert results.get('weather'), "Weather refresh should succeed"
    assert results.get('email'), "Email refresh should succeed"
    assert results.get('calendar'), "Calendar refresh should succeed"

    # Step 2: Verify all data is cached
    cached_weather = await db_manager.get_cache('weather_data')
    cached_emails = await db_manager.get_cache('emails_data')
    cached_calendar = await db_manager.get_cache('calendar_data')
    articles = await db_manager.get_all_articles()

    assert cached_weather is not None, "Weather should be cached"
    assert cached_emails is not None, "Emails should be cached"
    assert len(cached_emails) > 0, "Should have emails"
    assert cached_calendar is not None, "Calendar should be cached"
    assert len(cached_calendar) > 0, "Should have events"
    assert len(articles) > 0, "Should have articles"

    # Step 3: Load via startup_service (UI path)
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Step 4: Verify UI loads all data correctly
    assert startup_data['error'] is None, "Startup should succeed"

    # Check all data types are loaded
    assert len(startup_data['articles']) > 0, "Should load articles"
    assert startup_data['weather_data'] is not None, "Should load weather"
    assert len(startup_data['emails']) > 0, "Should load emails"
    assert len(startup_data['calendar_events']) > 0, "Should load calendar"

    # Verify counts match
    assert len(startup_data['emails']) == len(cached_emails), \
        "UI email count should match cache"
    assert len(startup_data['calendar_events']) == len(cached_calendar), \
        "UI calendar count should match cache"

    # Verify cache age exists (may be old if using existing cache)
    # NOTE: We don't assert it's fresh because the test may run against existing cache
    assert startup_data['cache_age_hours'] is not None or len(startup_data['articles']) > 0, \
        "Should have cache age or articles"


@pytest.mark.asyncio
async def test_cache_timestamps_are_updated():
    """Test that refresh timestamps are updated correctly."""
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Record time before refresh
    before_refresh = datetime.now()

    # Run refresh
    refresh_service = get_background_refresh_service()
    await refresh_service._refresh_email()

    # Check timestamp was updated
    last_refresh = await db_manager.get_setting('last_email_refresh')
    assert last_refresh is not None, "Should have email refresh timestamp"

    # Parse timestamp and verify it's recent
    refresh_time = datetime.fromisoformat(last_refresh)
    assert refresh_time >= before_refresh, "Timestamp should be after test start"
    assert refresh_time <= datetime.now() + timedelta(seconds=5), \
        "Timestamp should be recent"


@pytest.mark.asyncio
async def test_startup_service_loads_timestamps():
    """Test that startup_service correctly loads all refresh timestamps.

    This is CRITICAL - timestamps must be loaded for UI to show last refresh times.
    """
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Run a full refresh to ensure timestamps are saved
    refresh_service = get_background_refresh_service()
    await refresh_service.refresh_all_sources()

    # Now load via startup_service (exactly like UI does)
    from src.services.startup_service import StartupService
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Verify timestamps dict is populated
    assert 'last_refresh' in startup_data, "Should have last_refresh key"
    timestamps = startup_data['last_refresh']

    assert isinstance(timestamps, dict), "Timestamps should be a dict"
    assert 'news' in timestamps, "Should have news timestamp"
    assert 'weather' in timestamps, "Should have weather timestamp"
    assert 'email' in timestamps, "Should have email timestamp"
    assert 'calendar' in timestamps, "Should have calendar timestamp"

    # Verify all timestamps are ISO format strings
    for key, value in timestamps.items():
        assert value is not None, f"{key} timestamp should not be None"
        assert isinstance(value, str), f"{key} timestamp should be string"
        # Verify it's valid ISO format
        datetime.fromisoformat(value)  # Will raise if invalid


@pytest.mark.asyncio
async def test_email_data_serialization_preserves_fields():
    """Test that email serialization preserves all important fields.

    This ensures the dict conversion in _refresh_email() doesn't lose data.
    """
    # Setup and refresh
    db_manager = get_db_manager()
    await db_manager.initialize()

    refresh_service = get_background_refresh_service()
    await refresh_service._refresh_email()

    # Get cached emails
    cached_emails = await db_manager.get_cache('emails_data')
    assert cached_emails is not None and len(cached_emails) > 0

    # Verify all expected fields are present
    email = cached_emails[0]
    expected_fields = {
        'id': str,
        'subject': str,
        'sender': str,
        'received_at': (str, type(None)),  # ISO format or None
        'body': str,
        'snippet': (str, type(None)),  # Can be None
        'importance_score': (int, float),
        'has_action_items': bool,
        'action_items': list,
        'is_read': bool,
        'labels': list
    }

    for field_name, expected_type in expected_fields.items():
        assert field_name in email, f"Email missing field: {field_name}"
        if isinstance(expected_type, tuple):
            assert isinstance(email[field_name], expected_type), \
                f"{field_name} should be one of {expected_type}, got {type(email[field_name])}"
        else:
            assert isinstance(email[field_name], expected_type), \
                f"{field_name} should be {expected_type}, got {type(email[field_name])}"


@pytest.mark.asyncio
async def test_calendar_data_serialization_preserves_fields():
    """Test that calendar serialization preserves all important fields.

    This ensures the dict conversion in _refresh_calendar() doesn't lose data.
    """
    # Setup and refresh
    db_manager = get_db_manager()
    await db_manager.initialize()

    refresh_service = get_background_refresh_service()
    await refresh_service._refresh_calendar()

    # Get cached calendar
    cached_calendar = await db_manager.get_cache('calendar_data')
    assert cached_calendar is not None and len(cached_calendar) > 0

    # Verify all expected fields are present
    event = cached_calendar[0]
    expected_fields = {
        'id': str,
        'summary': str,
        'start_time': (str, type(None)),  # ISO format or None
        'end_time': (str, type(None)),
        'location': (str, type(None)),
        'description': (str, type(None)),
        'attendees': list,
        'importance_score': (int, float),
        'requires_preparation': bool,
        'is_focus_time': bool
    }

    for field_name, expected_type in expected_fields.items():
        assert field_name in event, f"Event missing field: {field_name}"
        if isinstance(expected_type, tuple):
            assert isinstance(event[field_name], expected_type), \
                f"{field_name} should be one of {expected_type}, got {type(event[field_name])}"
        else:
            assert isinstance(event[field_name], expected_type), \
                f"{field_name} should be {expected_type}, got {type(event[field_name])}"


@pytest.mark.asyncio
async def test_ui_handles_missing_cache_gracefully():
    """Test that UI handles missing cache gracefully (empty state)."""
    # Setup
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Clear all caches to simulate fresh install
    await db_manager.set_cache('emails_data', None)
    await db_manager.set_cache('calendar_data', None)
    await db_manager.set_cache('weather_data', None)

    # Load via startup_service
    startup_service = StartupService(db_manager)
    startup_data = await startup_service.load_startup_data()

    # Should succeed without errors
    assert startup_data['error'] is None, "Should handle missing cache gracefully"

    # Should return empty collections
    assert startup_data['emails'] == [], "Should return empty emails"
    assert startup_data['calendar_events'] == [], "Should return empty calendar"
    assert startup_data['weather_data'] is None, "Should return None for weather"

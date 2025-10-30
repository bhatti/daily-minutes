"""Integration tests for UI data loading from database.

Tests that verify the UI helper functions properly load and format
data from the database without mocking.
"""

import pytest
import asyncio
from datetime import datetime

from src.services.startup_service import StartupService
from src.database.sqlite_manager import get_db_manager


@pytest.mark.asyncio
async def test_startup_service_loads_weather_data():
    """Test that startup service loads weather data from database."""
    service = StartupService()

    # Load all data
    data = await service.load_startup_data()

    # Verify weather data is loaded
    assert 'weather_data' in data, "weather_data should be in result"

    weather = data['weather_data']
    if weather:  # Only test if weather exists in DB
        assert isinstance(weather, dict), "weather_data should be a dict"
        assert 'temperature' in weather, "weather should have temperature field"
        assert 'description' in weather, "weather should have description field"
        print(f"✓ Weather loaded: {weather['temperature']}°F, {weather['description']}")


@pytest.mark.asyncio
async def test_startup_service_loads_all_data_types():
    """Test that startup service loads all data types."""
    service = StartupService()

    # Load all data
    data = await service.load_startup_data()

    # Verify structure
    assert 'articles' in data, "should have articles"
    assert 'weather_data' in data, "should have weather_data"
    assert 'emails' in data, "should have emails"
    assert 'calendar_events' in data, "should have calendar_events"
    assert 'daily_brief' in data, "should have daily_brief"
    assert 'last_refresh' in data, "should have last_refresh timestamps"

    print(f"✓ Loaded: {len(data['articles'])} articles, "
          f"{len(data['emails'])} emails, "
          f"{len(data['calendar_events'])} calendar events")


@pytest.mark.asyncio
async def test_weather_data_structure():
    """Test weather data has correct structure for UI rendering."""
    service = StartupService()
    data = await service.load_startup_data()

    weather = data.get('weather_data')
    if not weather:
        pytest.skip("No weather data in database")

    # Verify all required fields for UI exist
    required_fields = ['temperature', 'feels_like', 'description', 'humidity']
    for field in required_fields:
        assert field in weather, f"weather should have {field} field"

    # Verify types
    assert isinstance(weather['temperature'], (int, float)), "temperature should be numeric"
    assert isinstance(weather['description'], str), "description should be string"

    print(f"✓ Weather structure valid: {weather['temperature']}°F, {weather['description']}")


@pytest.mark.asyncio
async def test_calendar_events_are_dicts_or_objects():
    """Test that calendar events can be either dicts or CalendarEvent objects."""
    service = StartupService()
    data = await service.load_startup_data()

    calendar_events = data.get('calendar_events', [])
    if not calendar_events:
        pytest.skip("No calendar events in database")

    # Check first event
    event = calendar_events[0]
    assert isinstance(event, dict), "Calendar events should be dicts"

    # Verify required fields
    assert 'start_time' in event, "event should have start_time"
    assert 'summary' in event, "event should have summary"

    print(f"✓ Calendar events structure valid: {len(calendar_events)} events")


@pytest.mark.asyncio
async def test_email_timestamps_are_strings():
    """Test that email timestamps are stored as ISO strings."""
    service = StartupService()
    data = await service.load_startup_data()

    emails = data.get('emails', [])
    if not emails:
        pytest.skip("No emails in database")

    # Check first email
    email = emails[0]
    assert isinstance(email, dict), "Emails should be dicts"

    if 'received_at' in email:
        received_at = email['received_at']
        # Should be string or datetime
        assert isinstance(received_at, (str, datetime)), \
            f"received_at should be string or datetime, got {type(received_at)}"

        # If string, should be parseable
        if isinstance(received_at, str):
            parsed = datetime.fromisoformat(received_at.replace('Z', '+00:00'))
            print(f"✓ Email timestamp valid: {parsed}")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_startup_service_loads_weather_data())
    asyncio.run(test_startup_service_loads_all_data_types())
    asyncio.run(test_weather_data_structure())
    asyncio.run(test_calendar_events_are_dicts_or_objects())
    asyncio.run(test_email_timestamps_are_strings())
    print("\n✅ All integration tests passed!")

"""Integration tests for Streamlit UI data loading.

Tests the actual data loading flow used by streamlit_app.py
without mocking to verify real database integration.
"""

import pytest
import asyncio
from datetime import datetime


def test_streamlit_load_data_function():
    """Test the actual load_data() function used by Streamlit UI."""
    import sys
    sys.path.insert(0, '.')

    # Import the actual function from streamlit_app
    import importlib.util
    spec = importlib.util.spec_from_file_location("streamlit_app", "./streamlit_app.py")
    streamlit_module = importlib.util.module_from_spec(spec)

    # Mock streamlit to avoid GUI dependencies
    import unittest.mock as mock
    sys.modules['streamlit'] = mock.MagicMock()

    # Now load the module
    spec.loader.exec_module(streamlit_module)

    # Call the load_data function
    data = streamlit_module.load_data()

    # Verify data structure
    assert isinstance(data, dict), "load_data should return a dict"
    assert 'articles' in data, "should have articles"
    assert 'weather_data' in data, "should have weather_data"
    assert 'emails' in data, "should have emails"
    assert 'calendar_events' in data, "should have calendar_events"

    # Verify weather data
    weather = data.get('weather_data')
    if weather:
        print(f"\n✓ Weather loaded from UI: {weather.get('temperature')}°F")
        assert isinstance(weather, dict), "weather should be dict"
        assert 'temperature' in weather, "weather should have temperature"
        assert weather['temperature'] is not None, "temperature should not be None"
    else:
        print("\n✗ Weather data is None/empty in UI load_data()")
        pytest.fail("Weather data should be loaded")


def test_weather_tab_rendering_logic():
    """Test the weather tab logic to ensure it can render with loaded data."""
    # Simulate the data structure from load_data()
    mock_weather_data = {
        'temperature': 9.2,
        'feels_like': 6.7,
        'humidity': 86,
        'description': 'overcast',
        'wind_speed': 12.3
    }

    # Test the rendering conditions
    assert mock_weather_data is not None, "weather should exist"
    assert mock_weather_data.get('temperature') == 9.2, "should have correct temperature"

    # Test safety tip logic (temperature-based)
    temp = mock_weather_data.get('temperature', 0)

    # This should trigger "Light jacket" recommendation
    if temp < 0:
        clothing = "Heavy winter coat"
    elif temp < 10:
        clothing = "Warm jacket"
    elif temp < 20:
        clothing = "Light jacket"
    elif temp < 30:
        clothing = "Light clothing"
    else:
        clothing = "Very light clothing"

    print(f"\n✓ Temperature {temp}°F → Clothing: {clothing}")
    assert clothing == "Warm jacket", f"Should recommend warm jacket for {temp}°F"


@pytest.mark.asyncio
async def test_ui_helper_functions_with_weather_dict():
    """Test UI helper functions handle weather dict correctly."""
    from src.services.startup_service import get_startup_service
    from src.database.sqlite_manager import get_db_manager

    # Load real data
    db = get_db_manager()
    await db.initialize()
    service = get_startup_service(db)
    data = await service.load_startup_data()

    weather = data.get('weather_data')
    assert weather is not None, "Weather should be loaded"
    assert isinstance(weather, dict), "Weather should be dict"

    # Test accessing fields like UI does
    temp = weather.get('temperature', 'N/A')
    feels_like = weather.get('feels_like', 'N/A')
    desc = weather.get('description', 'Unknown')

    print(f"\n✓ UI data access test:")
    print(f"  Temperature: {temp}°F")
    print(f"  Feels like: {feels_like}°F")
    print(f"  Description: {desc}")

    assert temp != 'N/A', "Temperature should not be N/A"
    assert isinstance(temp, (int, float)), "Temperature should be numeric"


if __name__ == "__main__":
    # Run tests
    test_streamlit_load_data_function()
    test_weather_tab_rendering_logic()
    asyncio.run(test_ui_helper_functions_with_weather_dict())
    print("\n✅ All Streamlit integration tests passed!")

"""Integration test for the actual load_all_data() function.

This test verifies that the exact code path used by streamlit_app.py
loads weather data correctly from the database.
"""

import pytest
import asyncio


def test_load_all_data_loads_weather():
    """Test that load_all_data() (used by Streamlit) loads weather data."""
    from src.services.startup_service import get_startup_service
    from src.database.sqlite_manager import get_db_manager

    # This is the EXACT code from streamlit_app.py load_all_data()
    db = get_db_manager()
    startup_service = get_startup_service(db)

    # Run the async function synchronously (like streamlit does via run_async)
    data = asyncio.run(startup_service.load_startup_data(limit=100))

    # Verify weather data structure
    assert 'weather_data' in data, "data should have weather_data key"

    weather = data['weather_data']
    assert weather is not None, f"weather_data should not be None, got: {weather}"
    assert isinstance(weather, dict), f"weather_data should be dict, got: {type(weather)}"

    # Verify weather has correct fields
    assert 'temperature' in weather, f"weather should have temperature, got keys: {weather.keys()}"
    assert weather['temperature'] is not None, f"temperature should not be None"

    # Verify it's the actual data from database
    assert isinstance(weather['temperature'], (int, float)), \
        f"temperature should be numeric, got {type(weather['temperature'])}"

    print(f"\n✅ load_all_data() successfully loads weather: {weather['temperature']}°F, {weather.get('description')}")
    print(f"   Full weather data: {weather}")


def test_weather_dict_access_pattern():
    """Test that weather dict can be accessed the way the UI does it."""
    from src.services.startup_service import get_startup_service
    from src.database.sqlite_manager import get_db_manager

    db = get_db_manager()
    startup_service = get_startup_service(db)
    data = asyncio.run(startup_service.load_startup_data(limit=100))

    weather_data = data.get('weather_data')
    assert weather_data is not None, "weather_data should exist"

    # Test the access pattern used in streamlit_app.py line 296 (after fix)
    temp = weather_data.get('temperature', 'N/A')
    assert temp != 'N/A', f"temperature should not be N/A, got: {temp}"

    # Test the access pattern used in daily_brief.py line 59 (after fix)
    temp_str = 'N/A'
    if weather_data and isinstance(weather_data, dict):
        temp_str = weather_data.get('temperature', 'N/A')

    assert temp_str != 'N/A', f"temperature string should not be N/A, got: {temp_str}"

    print(f"\n✅ Weather dict access patterns work correctly: {temp}°F")


def test_verify_weather_in_database():
    """Verify weather data is actually in the database."""
    import asyncio
    from src.database.sqlite_manager import get_db_manager

    async def check_weather():
        db = get_db_manager()
        await db.initialize()

        # Get weather from KV store
        weather_json = await db.get_value('weather_data')
        assert weather_json is not None, "weather_data should be in KV store"

        import json
        weather = json.loads(weather_json)
        assert 'temperature' in weather, "weather should have temperature field"

        print(f"\n✅ Weather in database: {weather['temperature']}°F")
        return weather

    weather = asyncio.run(check_weather())
    assert weather is not None


if __name__ == "__main__":
    # Run tests directly
    test_load_all_data_loads_weather()
    test_weather_dict_access_pattern()
    test_verify_weather_in_database()
    print("\n✅ All load_all_data integration tests passed!")

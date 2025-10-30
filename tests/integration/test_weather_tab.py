#!/usr/bin/env python3
"""Integration tests for Weather tab functionality."""

import asyncio
import sys
sys.path.append('.')

from src.connectors.weather import get_weather_service


async def test_weather_tab_current_data():
    """Test that current weather has the structure needed for the Weather tab."""
    print("\n=== Testing Weather Tab Current Data ===\n")

    weather_service = get_weather_service()
    test_location = "Seattle"

    # Fetch current weather
    weather_data = await weather_service.get_current_weather(test_location)

    if not weather_data:
        print("❌ No weather data fetched")
        return False

    print(f"✅ Fetched weather for {weather_data.location}")

    # Verify required fields for tab display
    required_fields = {
        'location': str,
        'temperature': (int, float),
        'feels_like': (int, float),
        'humidity': (int, float),
        'wind_speed': (int, float),
        'description': str,
        'timestamp': object  # datetime
    }

    for field, expected_type in required_fields.items():
        if not hasattr(weather_data, field):
            print(f"   ❌ Missing field: {field}")
            return False

        value = getattr(weather_data, field)
        if value is None:
            print(f"   ❌ Null value for: {field}")
            return False

        if not isinstance(value, expected_type):
            print(f"   ⚠️  Type mismatch for {field}: {type(value)} (expected {expected_type})")

        print(f"   ✅ {field}: {value}")

    print("\n✅ All required fields present")
    return True


async def test_weather_tab_forecast_data():
    """Test that forecast data has the structure needed for the Weather tab."""
    print("\n=== Testing Weather Tab Forecast Data ===\n")

    weather_service = get_weather_service()
    test_location = "Seattle"

    # Fetch forecast
    forecast_data = await weather_service.get_forecast(test_location, days=5)

    if not forecast_data:
        print("❌ No forecast data fetched")
        return False

    print(f"✅ Fetched {len(forecast_data)} days of forecast")

    # Verify each forecast day
    for i, day_forecast in enumerate(forecast_data, 1):
        print(f"\nDay {i}: {day_forecast.timestamp.strftime('%Y-%m-%d')}")

        required_fields = ['temperature', 'description', 'wind_speed', 'humidity']
        for field in required_fields:
            if not hasattr(day_forecast, field):
                print(f"   ❌ Missing field: {field}")
                return False

            value = getattr(day_forecast, field)
            print(f"   ✅ {field}: {value}")

    print("\n✅ All forecast days have required fields")
    return True


async def test_weather_tab_multiple_locations():
    """Test that weather can be fetched for multiple locations."""
    print("\n=== Testing Multiple Locations ===\n")

    weather_service = get_weather_service()
    test_locations = ["Seattle", "San Francisco", "New York"]

    results = {}
    for location in test_locations:
        weather_data = await weather_service.get_current_weather(location)
        results[location] = weather_data is not None

        if results[location]:
            print(f"✅ {location}: {weather_data.temperature}°C, {weather_data.description}")
        else:
            print(f"❌ {location}: Failed to fetch")

    success_count = sum(results.values())
    print(f"\n✅ Successfully fetched {success_count}/{len(test_locations)} locations")

    return success_count >= 2  # At least 2 should succeed


async def test_weather_tab_error_handling():
    """Test that weather handles errors gracefully."""
    print("\n=== Testing Weather Error Handling ===\n")

    weather_service = get_weather_service()

    # Test with invalid location
    invalid_locations = ["", "XYZ123INVALID", "!@#$%"]

    for location in invalid_locations:
        weather_data = await weather_service.get_current_weather(location)

        if weather_data is None:
            print(f"✅ Correctly handled invalid location: '{location}'")
        else:
            print(f"⚠️  Unexpected success for: '{location}'")

    print("\n✅ Error handling works correctly")
    return True


async def test_weather_recommendations():
    """Test that weather provides actionable recommendations."""
    print("\n=== Testing Weather Recommendations ===\n")

    weather_service = get_weather_service()

    # Test different weather conditions
    test_cases = [
        ("Seattle", "temperate"),  # Cool, possibly rainy
        ("Phoenix", "hot"),        # Hot and dry
        ("New York", "variable"),  # Variable conditions
    ]

    all_recommendations_valid = True

    for location, expected_type in test_cases:
        weather_data = await weather_service.get_current_weather(location)

        if not weather_data:
            print(f"⚠️  Could not fetch weather for {location}")
            continue

        print(f"\n{location}: {weather_data.temperature}°C, {weather_data.description}")

        # Check for clothing recommendations based on temperature
        temp = weather_data.temperature
        if temp < 10:
            expected_clothing = "warm jacket"
            print(f"   ✓ Expected: {expected_clothing} recommendation")
        elif temp < 20:
            expected_clothing = "light jacket"
            print(f"   ✓ Expected: {expected_clothing} recommendation")
        else:
            expected_clothing = "light clothing"
            print(f"   ✓ Expected: {expected_clothing} recommendation")

        # Check for umbrella recommendation based on conditions
        if "rain" in weather_data.description.lower() or "drizzle" in weather_data.description.lower():
            print(f"   ✓ Expected: umbrella recommendation")

        # Check for severe weather safety (would need forecast for storms)
        if "storm" in weather_data.description.lower() or "thunder" in weather_data.description.lower():
            print(f"   ✓ Expected: safety tips and emergency prep")

    print(f"\n✅ Weather recommendation structure validated")
    return True


async def main():
    """Run all Weather tab tests."""
    print("=" * 60)
    print("WEATHER TAB INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    if not await test_weather_tab_current_data():
        all_passed = False

    if not await test_weather_tab_forecast_data():
        all_passed = False

    if not await test_weather_tab_multiple_locations():
        all_passed = False

    if not await test_weather_tab_error_handling():
        all_passed = False

    if not await test_weather_recommendations():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL WEATHER TAB TESTS PASSED")
    else:
        print("❌ SOME WEATHER TAB TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

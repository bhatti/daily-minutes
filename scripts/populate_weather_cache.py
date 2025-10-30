#!/usr/bin/env python3
"""Quick script to populate weather cache for testing."""
import asyncio
import sys
sys.path.insert(0, '.')

async def main():
    from src.services.background_refresh_service import get_background_refresh_service

    service = get_background_refresh_service()
    print("Fetching weather data...")
    success = await service._refresh_weather()

    if success:
        print("✅ Weather data cached successfully!")
    else:
        print("❌ Failed to cache weather data")

if __name__ == "__main__":
    asyncio.run(main())

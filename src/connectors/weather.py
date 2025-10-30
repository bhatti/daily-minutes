"""Weather connector for fetching weather data from various sources."""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import httpx
from pydantic import HttpUrl

from src.core.logging import get_logger
from src.core.config import get_settings
from src.models.news import NewsArticle
from src.models.base import DataSource
from src.services.observability_service import get_observability_service

logger = get_logger(__name__)
settings = get_settings()
observability = get_observability_service()

# Handle SSL verification for proxy environments
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() != 'false'


@dataclass
class WeatherData:
    """Weather data structure."""
    location: str
    temperature: float  # Celsius
    feels_like: float
    humidity: float  # Percentage
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    description: str
    icon: str
    timestamp: datetime
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None
    visibility: Optional[float] = None  # meters
    uv_index: Optional[float] = None
    forecast: List[Dict[str, Any]] = field(default_factory=list)


class WeatherConnector:
    """Base weather connector interface."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather connector.

        Args:
            api_key: Optional API key for the weather service
        """
        self.api_key = api_key

    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather for a location.

        Args:
            location: City name or coordinates

        Returns:
            WeatherData object or None if failed
        """
        raise NotImplementedError

    async def get_forecast(self, location: str, days: int = 5) -> Optional[List[WeatherData]]:
        """Get weather forecast for a location.

        Args:
            location: City name or coordinates
            days: Number of days to forecast

        Returns:
            List of WeatherData objects or None if failed
        """
        raise NotImplementedError


class OpenWeatherMapConnector(WeatherConnector):
    """OpenWeatherMap API connector."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenWeatherMap connector.

        Args:
            api_key: OpenWeatherMap API key
        """
        super().__init__(api_key or settings.weather.openweather_api_key)
        if not self.api_key:
            logger.warning("openweather_no_api_key",
                         message="No OpenWeatherMap API key configured")

    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather from OpenWeatherMap.

        Args:
            location: City name or coordinates (lat,lon)

        Returns:
            WeatherData object or None if failed
        """
        if not self.api_key:
            return None

        op_id = observability.start_operation("weather.openweather.current")

        try:
            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                # Check if location is coordinates
                if "," in location:
                    lat, lon = location.split(",")
                    params = {
                        "lat": lat.strip(),
                        "lon": lon.strip(),
                        "appid": self.api_key,
                        "units": "metric"
                    }
                else:
                    params = {
                        "q": location,
                        "appid": self.api_key,
                        "units": "metric"
                    }

                response = await client.get(
                    f"{self.BASE_URL}/weather",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()

                weather = WeatherData(
                    location=data["name"],
                    temperature=data["main"]["temp"],
                    feels_like=data["main"]["feels_like"],
                    humidity=data["main"]["humidity"],
                    pressure=data["main"]["pressure"],
                    wind_speed=data["wind"]["speed"],
                    wind_direction=data["wind"].get("deg", 0),
                    description=data["weather"][0]["description"],
                    icon=data["weather"][0]["icon"],
                    timestamp=datetime.fromtimestamp(data["dt"], tz=timezone.utc),
                    sunrise=datetime.fromtimestamp(data["sys"]["sunrise"], tz=timezone.utc) if "sunrise" in data["sys"] else None,
                    sunset=datetime.fromtimestamp(data["sys"]["sunset"], tz=timezone.utc) if "sunset" in data["sys"] else None,
                    visibility=data.get("visibility"),
                )

                observability.track_metric("weather.temperature", weather.temperature,
                                         {"location": location, "source": "openweather"})
                observability.end_operation(op_id, success=True)

                logger.info("weather_fetched",
                          location=location,
                          temperature=weather.temperature,
                          source="openweather")

                return weather

        except httpx.HTTPStatusError as e:
            observability.track_error(e, "weather.openweather")
            observability.end_operation(op_id, success=False)
            logger.error("openweather_api_error",
                       status_code=e.response.status_code,
                       error=str(e))
            return None
        except Exception as e:
            observability.track_error(e, "weather.openweather")
            observability.end_operation(op_id, success=False)
            logger.error("openweather_fetch_failed", error=str(e))
            return None

    async def get_forecast(self, location: str, days: int = 5) -> Optional[List[WeatherData]]:
        """Get weather forecast from OpenWeatherMap.

        Args:
            location: City name or coordinates
            days: Number of days (max 5 for free tier)

        Returns:
            List of WeatherData objects or None if failed
        """
        if not self.api_key:
            return None

        op_id = observability.start_operation("weather.openweather.forecast")

        try:
            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                # Check if location is coordinates
                if "," in location:
                    lat, lon = location.split(",")
                    params = {
                        "lat": lat.strip(),
                        "lon": lon.strip(),
                        "appid": self.api_key,
                        "units": "metric",
                        "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
                    }
                else:
                    params = {
                        "q": location,
                        "appid": self.api_key,
                        "units": "metric",
                        "cnt": days * 8
                    }

                response = await client.get(
                    f"{self.BASE_URL}/forecast",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                forecasts = []

                # Group by day and take noon forecast for each day
                daily_forecasts = {}
                for item in data["list"]:
                    dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
                    day_key = dt.date()

                    # Prefer noon forecast (12:00) or closest to it
                    if day_key not in daily_forecasts or abs(dt.hour - 12) < abs(daily_forecasts[day_key]["hour"] - 12):
                        daily_forecasts[day_key] = {
                            "data": item,
                            "hour": dt.hour
                        }

                for day_data in daily_forecasts.values():
                    item = day_data["data"]
                    forecast = WeatherData(
                        location=data["city"]["name"],
                        temperature=item["main"]["temp"],
                        feels_like=item["main"]["feels_like"],
                        humidity=item["main"]["humidity"],
                        pressure=item["main"]["pressure"],
                        wind_speed=item["wind"]["speed"],
                        wind_direction=item["wind"].get("deg", 0),
                        description=item["weather"][0]["description"],
                        icon=item["weather"][0]["icon"],
                        timestamp=datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                        visibility=item.get("visibility"),
                    )
                    forecasts.append(forecast)

                observability.end_operation(op_id, success=True)
                logger.info("forecast_fetched",
                          location=location,
                          days=len(forecasts),
                          source="openweather")

                return forecasts[:days]  # Limit to requested days

        except Exception as e:
            observability.track_error(e, "weather.openweather")
            observability.end_operation(op_id, success=False)
            logger.error("openweather_forecast_failed", error=str(e))
            return None


class OpenMeteoConnector(WeatherConnector):
    """Open-Meteo API connector (no API key required)."""

    BASE_URL = "https://api.open-meteo.com/v1"

    def __init__(self):
        """Initialize Open-Meteo connector."""
        super().__init__(api_key=None)

    async def _geocode(self, location: str) -> Optional[tuple[float, float]]:
        """Geocode a location name to coordinates.

        Args:
            location: City name

        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        try:
            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                response = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return result["latitude"], result["longitude"]
                return None

        except Exception as e:
            logger.error("geocoding_failed", location=location, error=str(e))
            return None

    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather from Open-Meteo.

        Args:
            location: City name or coordinates (lat,lon)

        Returns:
            WeatherData object or None if failed
        """
        op_id = observability.start_operation("weather.openmeteo.current")

        try:
            # Parse coordinates or geocode location
            if "," in location:
                lat, lon = map(float, location.split(","))
                city_name = location
            else:
                coords = await self._geocode(location)
                if not coords:
                    observability.end_operation(op_id, success=False)
                    return None
                lat, lon = coords
                city_name = location

            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m,wind_direction_10m,weather_code",
                    "timezone": "auto"
                }

                response = await client.get(
                    f"{self.BASE_URL}/forecast",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                current = data["current"]

                # Map weather codes to descriptions
                weather_descriptions = {
                    0: "clear sky",
                    1: "mainly clear",
                    2: "partly cloudy",
                    3: "overcast",
                    45: "foggy",
                    48: "depositing rime fog",
                    51: "light drizzle",
                    53: "moderate drizzle",
                    55: "dense drizzle",
                    61: "slight rain",
                    63: "moderate rain",
                    65: "heavy rain",
                    71: "slight snow",
                    73: "moderate snow",
                    75: "heavy snow",
                    77: "snow grains",
                    80: "slight rain showers",
                    81: "moderate rain showers",
                    82: "violent rain showers",
                    85: "slight snow showers",
                    86: "heavy snow showers",
                    95: "thunderstorm",
                    96: "thunderstorm with slight hail",
                    99: "thunderstorm with heavy hail"
                }

                weather_code = int(current.get("weather_code", 0))
                description = weather_descriptions.get(weather_code, "unknown")

                # Map weather code to icon (simplified)
                if weather_code == 0:
                    icon = "01d"
                elif weather_code in [1, 2]:
                    icon = "02d"
                elif weather_code == 3:
                    icon = "03d"
                elif weather_code in [45, 48]:
                    icon = "50d"
                elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                    icon = "10d"
                elif weather_code in [71, 73, 75, 77, 85, 86]:
                    icon = "13d"
                elif weather_code in [95, 96, 99]:
                    icon = "11d"
                else:
                    icon = "01d"

                weather = WeatherData(
                    location=city_name,
                    temperature=current["temperature_2m"],
                    feels_like=current["apparent_temperature"],
                    humidity=current["relative_humidity_2m"],
                    pressure=current["pressure_msl"],
                    wind_speed=current["wind_speed_10m"],
                    wind_direction=current["wind_direction_10m"],
                    description=description,
                    icon=icon,
                    timestamp=datetime.fromisoformat(current["time"].replace("Z", "+00:00"))
                )

                observability.track_metric("weather.temperature", weather.temperature,
                                         {"location": location, "source": "openmeteo"})
                observability.end_operation(op_id, success=True)

                logger.info("weather_fetched",
                          location=location,
                          temperature=weather.temperature,
                          source="openmeteo")

                return weather

        except Exception as e:
            observability.track_error(e, "weather.openmeteo")
            observability.end_operation(op_id, success=False)
            logger.error("openmeteo_fetch_failed", error=str(e))
            return None

    async def get_forecast(self, location: str, days: int = 5) -> Optional[List[WeatherData]]:
        """Get weather forecast from Open-Meteo.

        Args:
            location: City name or coordinates
            days: Number of days

        Returns:
            List of WeatherData objects or None if failed
        """
        op_id = observability.start_operation("weather.openmeteo.forecast")

        try:
            # Parse coordinates or geocode location
            if "," in location:
                lat, lon = map(float, location.split(","))
                city_name = location
            else:
                coords = await self._geocode(location)
                if not coords:
                    observability.end_operation(op_id, success=False)
                    return None
                lat, lon = coords
                city_name = location

            async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_sum,rain_sum,wind_speed_10m_max,wind_gusts_10m_max,weather_code",
                    "forecast_days": days,
                    "timezone": "auto"
                }

                response = await client.get(
                    f"{self.BASE_URL}/forecast",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                daily = data["daily"]

                forecasts = []
                for i in range(min(days, len(daily["time"]))):
                    # Use average of max and min for temperature
                    avg_temp = (daily["temperature_2m_max"][i] + daily["temperature_2m_min"][i]) / 2
                    avg_feels_like = (daily["apparent_temperature_max"][i] + daily["apparent_temperature_min"][i]) / 2

                    weather_code = int(daily["weather_code"][i])

                    # Map weather codes (same as above)
                    weather_descriptions = {
                        0: "clear sky",
                        1: "mainly clear",
                        2: "partly cloudy",
                        3: "overcast",
                        45: "foggy",
                        48: "depositing rime fog",
                        51: "light drizzle",
                        53: "moderate drizzle",
                        55: "dense drizzle",
                        61: "slight rain",
                        63: "moderate rain",
                        65: "heavy rain",
                        71: "slight snow",
                        73: "moderate snow",
                        75: "heavy snow",
                        77: "snow grains",
                        80: "slight rain showers",
                        81: "moderate rain showers",
                        82: "violent rain showers",
                        85: "slight snow showers",
                        86: "heavy snow showers",
                        95: "thunderstorm",
                        96: "thunderstorm with slight hail",
                        99: "thunderstorm with heavy hail"
                    }

                    description = weather_descriptions.get(weather_code, "unknown")

                    # Simple icon mapping
                    if weather_code == 0:
                        icon = "01d"
                    elif weather_code in [1, 2]:
                        icon = "02d"
                    elif weather_code == 3:
                        icon = "03d"
                    elif weather_code in [45, 48]:
                        icon = "50d"
                    elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                        icon = "10d"
                    elif weather_code in [71, 73, 75, 77, 85, 86]:
                        icon = "13d"
                    elif weather_code in [95, 96, 99]:
                        icon = "11d"
                    else:
                        icon = "01d"

                    forecast = WeatherData(
                        location=city_name,
                        temperature=avg_temp,
                        feels_like=avg_feels_like,
                        humidity=0,  # Not provided in daily forecast
                        pressure=0,  # Not provided in daily forecast
                        wind_speed=daily["wind_speed_10m_max"][i],
                        wind_direction=0,  # Not provided in daily forecast
                        description=description,
                        icon=icon,
                        timestamp=datetime.fromisoformat(daily["time"][i] + "T12:00:00+00:00"),
                        forecast=[{
                            "temp_max": daily["temperature_2m_max"][i],
                            "temp_min": daily["temperature_2m_min"][i],
                            "precipitation": daily["precipitation_sum"][i],
                            "rain": daily["rain_sum"][i],
                            "wind_gusts": daily["wind_gusts_10m_max"][i]
                        }]
                    )
                    forecasts.append(forecast)

                observability.end_operation(op_id, success=True)
                logger.info("forecast_fetched",
                          location=location,
                          days=len(forecasts),
                          source="openmeteo")

                return forecasts

        except Exception as e:
            observability.track_error(e, "weather.openmeteo")
            observability.end_operation(op_id, success=False)
            logger.error("openmeteo_forecast_failed", error=str(e))
            return None


class WeatherService:
    """Service for managing multiple weather sources."""

    def __init__(self):
        """Initialize weather service."""
        self.connectors: List[WeatherConnector] = []

        # Add OpenWeatherMap if API key is available
        if settings.weather.openweather_api_key:
            self.connectors.append(OpenWeatherMapConnector())
            logger.info("weather_connector_added", connector="OpenWeatherMap")

        # Always add Open-Meteo as fallback (no API key required)
        self.connectors.append(OpenMeteoConnector())
        logger.info("weather_connector_added", connector="Open-Meteo")

    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather from available sources.

        Args:
            location: City name or coordinates

        Returns:
            WeatherData object or None if all sources fail
        """
        for connector in self.connectors:
            try:
                weather = await connector.get_current_weather(location)
                if weather:
                    return weather
            except Exception as e:
                logger.warning("weather_connector_failed",
                             connector=connector.__class__.__name__,
                             error=str(e))
                continue

        logger.error("all_weather_connectors_failed", location=location)
        return None

    async def get_forecast(self, location: str, days: int = 5) -> Optional[List[WeatherData]]:
        """Get weather forecast from available sources.

        Args:
            location: City name or coordinates
            days: Number of days

        Returns:
            List of WeatherData objects or None if all sources fail
        """
        for connector in self.connectors:
            try:
                forecast = await connector.get_forecast(location, days)
                if forecast:
                    return forecast
            except Exception as e:
                logger.warning("forecast_connector_failed",
                             connector=connector.__class__.__name__,
                             error=str(e))
                continue

        logger.error("all_forecast_connectors_failed", location=location)
        return None

    def to_article(self, weather: WeatherData) -> NewsArticle:
        """Convert WeatherData to NewsArticle for unified processing.

        Args:
            weather: WeatherData object

        Returns:
            NewsArticle object
        """
        # Create a weather summary
        summary = f"""
        Current weather in {weather.location}:
        Temperature: {weather.temperature}°C (feels like {weather.feels_like}°C)
        Conditions: {weather.description}
        Humidity: {weather.humidity}%
        Wind: {weather.wind_speed} m/s
        Pressure: {weather.pressure} hPa
        """

        if weather.sunrise and weather.sunset:
            summary += f"\nSunrise: {weather.sunrise.strftime('%H:%M')}"
            summary += f"\nSunset: {weather.sunset.strftime('%H:%M')}"

        return NewsArticle(
            title=f"Weather in {weather.location}: {weather.description}",
            url=HttpUrl(f"https://weather.local/{weather.location.lower().replace(' ', '-')}"),
            content=summary.strip(),
            published_at=weather.timestamp,
            source=DataSource.CUSTOM,
            source_name="Weather Service",
            author="Weather Service",
            tags=["weather", weather.location.lower(), weather.description],
            description=f"Current weather: {weather.temperature}°C, {weather.description}"
        )


# Global weather service instance
_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    """Get or create weather service instance.

    Returns:
        WeatherService instance
    """
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service
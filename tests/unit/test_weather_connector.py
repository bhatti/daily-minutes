"""Unit tests for weather connectors."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from src.connectors.weather import (
    WeatherData,
    OpenWeatherMapConnector,
    OpenMeteoConnector,
    WeatherService
)


class TestWeatherData:
    """Test WeatherData dataclass."""

    def test_weather_data_creation(self):
        """Test creating WeatherData object."""
        weather = WeatherData(
            location="San Francisco",
            temperature=20.5,
            feels_like=19.0,
            humidity=65,
            pressure=1013,
            wind_speed=5.5,
            wind_direction=180,
            description="partly cloudy",
            icon="02d",
            timestamp=datetime.now(timezone.utc)
        )

        assert weather.location == "San Francisco"
        assert weather.temperature == 20.5
        assert weather.feels_like == 19.0
        assert weather.humidity == 65
        assert weather.pressure == 1013
        assert weather.wind_speed == 5.5
        assert weather.wind_direction == 180
        assert weather.description == "partly cloudy"
        assert weather.icon == "02d"
        assert weather.timestamp is not None

    def test_weather_data_optional_fields(self):
        """Test WeatherData with optional fields."""
        weather = WeatherData(
            location="London",
            temperature=15.0,
            feels_like=14.0,
            humidity=70,
            pressure=1010,
            wind_speed=3.0,
            wind_direction=90,
            description="rainy",
            icon="10d",
            timestamp=datetime.now(timezone.utc),
            sunrise=datetime.now(timezone.utc),
            sunset=datetime.now(timezone.utc),
            visibility=10000,
            uv_index=3.5,
            forecast=[{"temp_max": 18, "temp_min": 12}]
        )

        assert weather.sunrise is not None
        assert weather.sunset is not None
        assert weather.visibility == 10000
        assert weather.uv_index == 3.5
        assert len(weather.forecast) == 1


class TestOpenWeatherMapConnector:
    """Test OpenWeatherMap connector."""

    @pytest.fixture
    def connector(self):
        """Create connector with test API key."""
        return OpenWeatherMapConnector(api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_get_current_weather_success(self, connector):
        """Test successful weather fetch."""
        mock_response = {
            "name": "San Francisco",
            "main": {
                "temp": 20.5,
                "feels_like": 19.0,
                "humidity": 65,
                "pressure": 1013
            },
            "wind": {
                "speed": 5.5,
                "deg": 180
            },
            "weather": [{
                "description": "partly cloudy",
                "icon": "02d"
            }],
            "dt": int(datetime.now(timezone.utc).timestamp()),
            "sys": {
                "sunrise": int(datetime.now(timezone.utc).timestamp()),
                "sunset": int(datetime.now(timezone.utc).timestamp())
            },
            "visibility": 10000
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            weather = await connector.get_current_weather("San Francisco")

            assert weather is not None
            assert weather.location == "San Francisco"
            assert weather.temperature == 20.5
            assert weather.feels_like == 19.0
            assert weather.humidity == 65
            assert weather.pressure == 1013
            assert weather.wind_speed == 5.5
            assert weather.wind_direction == 180
            assert weather.description == "partly cloudy"
            assert weather.icon == "02d"

    @pytest.mark.asyncio
    async def test_get_current_weather_no_api_key(self):
        """Test weather fetch with no API key."""
        connector = OpenWeatherMapConnector(api_key=None)
        weather = await connector.get_current_weather("San Francisco")
        assert weather is None

    @pytest.mark.asyncio
    async def test_get_current_weather_coordinates(self, connector):
        """Test weather fetch with coordinates."""
        mock_response = {
            "name": "Location",
            "main": {
                "temp": 15.0,
                "feels_like": 14.0,
                "humidity": 70,
                "pressure": 1010
            },
            "wind": {
                "speed": 3.0,
                "deg": 90
            },
            "weather": [{
                "description": "clear sky",
                "icon": "01d"
            }],
            "dt": int(datetime.now(timezone.utc).timestamp()),
            "sys": {}
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            weather = await connector.get_current_weather("37.7749,-122.4194")

            assert weather is not None
            assert weather.temperature == 15.0
            mock_client_instance.get.assert_called_once()
            call_args = mock_client_instance.get.call_args
            assert "lat" in call_args[1]["params"]
            assert "lon" in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_get_current_weather_http_error(self, connector):
        """Test weather fetch with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404)
            )

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            weather = await connector.get_current_weather("InvalidCity")
            assert weather is None

    @pytest.mark.asyncio
    async def test_get_forecast_success(self, connector):
        """Test successful forecast fetch."""
        mock_response = {
            "city": {"name": "San Francisco"},
            "list": [
                {
                    "dt": int(datetime.now(timezone.utc).timestamp()),
                    "main": {
                        "temp": 20.0,
                        "feels_like": 19.0,
                        "humidity": 60,
                        "pressure": 1015
                    },
                    "wind": {"speed": 4.0, "deg": 200},
                    "weather": [{"description": "sunny", "icon": "01d"}]
                },
                {
                    "dt": int(datetime.now(timezone.utc).timestamp()) + 86400,
                    "main": {
                        "temp": 22.0,
                        "feels_like": 21.0,
                        "humidity": 55,
                        "pressure": 1014
                    },
                    "wind": {"speed": 5.0, "deg": 210},
                    "weather": [{"description": "cloudy", "icon": "03d"}]
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            forecast = await connector.get_forecast("San Francisco", days=2)

            assert forecast is not None
            assert len(forecast) <= 2
            if len(forecast) > 0:
                assert forecast[0].location == "San Francisco"


class TestOpenMeteoConnector:
    """Test Open-Meteo connector."""

    @pytest.fixture
    def connector(self):
        """Create Open-Meteo connector."""
        return OpenMeteoConnector()

    @pytest.mark.asyncio
    async def test_geocode_success(self, connector):
        """Test successful geocoding."""
        mock_response = {
            "results": [{
                "latitude": 37.7749,
                "longitude": -122.4194,
                "name": "San Francisco"
            }]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            coords = await connector._geocode("San Francisco")

            assert coords is not None
            assert coords[0] == 37.7749
            assert coords[1] == -122.4194

    @pytest.mark.asyncio
    async def test_geocode_no_results(self, connector):
        """Test geocoding with no results."""
        mock_response = {"results": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            coords = await connector._geocode("NonexistentPlace")
            assert coords is None

    @pytest.mark.asyncio
    async def test_get_current_weather_success(self, connector):
        """Test successful weather fetch from Open-Meteo."""
        # Mock geocoding
        with patch.object(connector, "_geocode", return_value=(37.7749, -122.4194)):
            mock_response = {
                "current": {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "temperature_2m": 18.5,
                    "apparent_temperature": 17.0,
                    "relative_humidity_2m": 68,
                    "pressure_msl": 1012,
                    "wind_speed_10m": 4.2,
                    "wind_direction_10m": 225,
                    "weather_code": 2
                }
            }

            with patch("httpx.AsyncClient") as mock_client:
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = mock_response
                mock_response_obj.raise_for_status.return_value = None

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response_obj
                mock_client.return_value.__aenter__.return_value = mock_client_instance

                weather = await connector.get_current_weather("San Francisco")

                assert weather is not None
                assert weather.location == "San Francisco"
                assert weather.temperature == 18.5
                assert weather.feels_like == 17.0
                assert weather.humidity == 68
                assert weather.pressure == 1012
                assert weather.wind_speed == 4.2
                assert weather.wind_direction == 225
                assert weather.description == "partly cloudy"

    @pytest.mark.asyncio
    async def test_get_current_weather_with_coordinates(self, connector):
        """Test weather fetch with coordinates."""
        mock_response = {
            "current": {
                "time": datetime.now(timezone.utc).isoformat(),
                "temperature_2m": 15.0,
                "apparent_temperature": 14.0,
                "relative_humidity_2m": 75,
                "pressure_msl": 1008,
                "wind_speed_10m": 6.0,
                "wind_direction_10m": 180,
                "weather_code": 61
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response_obj
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            weather = await connector.get_current_weather("37.7749,-122.4194")

            assert weather is not None
            assert weather.temperature == 15.0
            assert weather.description == "slight rain"

    @pytest.mark.asyncio
    async def test_get_forecast_success(self, connector):
        """Test successful forecast fetch from Open-Meteo."""
        with patch.object(connector, "_geocode", return_value=(37.7749, -122.4194)):
            mock_response = {
                "daily": {
                    "time": [
                        datetime.now(timezone.utc).date().isoformat(),
                        (datetime.now(timezone.utc).date()).isoformat()
                    ],
                    "temperature_2m_max": [22.0, 24.0],
                    "temperature_2m_min": [15.0, 16.0],
                    "apparent_temperature_max": [21.0, 23.0],
                    "apparent_temperature_min": [14.0, 15.0],
                    "precipitation_sum": [0.0, 2.5],
                    "rain_sum": [0.0, 2.5],
                    "wind_speed_10m_max": [5.0, 7.0],
                    "wind_gusts_10m_max": [10.0, 15.0],
                    "weather_code": [1, 61]
                }
            }

            with patch("httpx.AsyncClient") as mock_client:
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = mock_response
                mock_response_obj.raise_for_status.return_value = None

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response_obj
                mock_client.return_value.__aenter__.return_value = mock_client_instance

                forecast = await connector.get_forecast("San Francisco", days=2)

                assert forecast is not None
                assert len(forecast) == 2
                assert forecast[0].location == "San Francisco"
                assert forecast[0].description == "mainly clear"
                assert forecast[1].description == "slight rain"


class TestWeatherService:
    """Test WeatherService."""

    @pytest.fixture
    def service(self):
        """Create weather service."""
        with patch("src.connectors.weather.settings") as mock_settings:
            mock_settings.weather.openweather_api_key = None
            return WeatherService()

    @pytest.mark.asyncio
    async def test_get_current_weather_fallback(self, service):
        """Test weather fetch with fallback to Open-Meteo."""
        # First connector fails, second succeeds
        mock_weather = WeatherData(
            location="San Francisco",
            temperature=20.0,
            feels_like=19.0,
            humidity=65,
            pressure=1013,
            wind_speed=5.0,
            wind_direction=180,
            description="clear",
            icon="01d",
            timestamp=datetime.now(timezone.utc)
        )

        service.connectors = [
            AsyncMock(get_current_weather=AsyncMock(side_effect=Exception("API Error"))),
            AsyncMock(get_current_weather=AsyncMock(return_value=mock_weather))
        ]

        weather = await service.get_current_weather("San Francisco")

        assert weather is not None
        assert weather.location == "San Francisco"
        assert weather.temperature == 20.0

    @pytest.mark.asyncio
    async def test_get_current_weather_all_fail(self, service):
        """Test weather fetch when all connectors fail."""
        service.connectors = [
            AsyncMock(get_current_weather=AsyncMock(return_value=None)),
            AsyncMock(get_current_weather=AsyncMock(return_value=None))
        ]

        weather = await service.get_current_weather("San Francisco")
        assert weather is None

    def test_to_article_conversion(self, service):
        """Test converting WeatherData to NewsArticle."""
        weather = WeatherData(
            location="San Francisco",
            temperature=20.0,
            feels_like=19.0,
            humidity=65,
            pressure=1013,
            wind_speed=5.0,
            wind_direction=180,
            description="partly cloudy",
            icon="02d",
            timestamp=datetime.now(timezone.utc),
            sunrise=datetime.now(timezone.utc),
            sunset=datetime.now(timezone.utc)
        )

        article = service.to_article(weather)

        assert article.title == "Weather in San Francisco: partly cloudy"
        assert "20.0°C" in article.content
        assert "feels like 19.0°C" in article.content
        assert "65%" in article.content
        assert article.source == "custom"  # DataSource.CUSTOM with use_enum_values=True
        assert article.source_name == "Weather Service"
        assert "weather" in article.tags
        assert "san francisco" in article.tags
"""Models for weather data and forecasts."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from src.models.base import BaseModel, Priority


class WeatherCondition(str, Enum):
    """Weather condition enum."""

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"
    MIST = "mist"
    HAZE = "haze"
    WINDY = "windy"


class WeatherAlert(str, Enum):
    """Weather alert severity levels."""

    INFO = "info"
    ADVISORY = "advisory"
    WATCH = "watch"
    WARNING = "warning"
    EMERGENCY = "emergency"


class WeatherData(BaseModel):
    """Model for current weather data."""

    location: str = Field(..., description="Location name")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    timezone: str = Field("UTC", description="Timezone")

    temperature: float = Field(..., description="Temperature in Celsius")
    feels_like: float = Field(..., description="Feels like temperature")
    temperature_min: float = Field(..., description="Minimum temperature")
    temperature_max: float = Field(..., description="Maximum temperature")

    condition: WeatherCondition = Field(..., description="Weather condition")
    description: str = Field(..., description="Weather description")
    icon: Optional[str] = Field(None, description="Weather icon code")

    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., gt=0, description="Atmospheric pressure (hPa)")
    visibility: Optional[float] = Field(None, ge=0, description="Visibility in meters")

    wind_speed: float = Field(0.0, ge=0, description="Wind speed (m/s)")
    wind_direction: Optional[int] = Field(None, ge=0, le=360, description="Wind direction degrees")
    wind_gust: Optional[float] = Field(None, ge=0, description="Wind gust speed")

    clouds: int = Field(0, ge=0, le=100, description="Cloud coverage percentage")
    rain_1h: Optional[float] = Field(None, ge=0, description="Rain in last hour (mm)")
    rain_3h: Optional[float] = Field(None, ge=0, description="Rain in last 3 hours (mm)")
    snow_1h: Optional[float] = Field(None, ge=0, description="Snow in last hour (mm)")
    snow_3h: Optional[float] = Field(None, ge=0, description="Snow in last 3 hours (mm)")

    uv_index: Optional[float] = Field(None, ge=0, le=11, description="UV index")
    air_quality_index: Optional[int] = Field(None, ge=0, description="Air quality index")

    sunrise: Optional[datetime] = Field(None, description="Sunrise time")
    sunset: Optional[datetime] = Field(None, description="Sunset time")

    observation_time: datetime = Field(..., description="Observation timestamp")
    source: str = Field("openweather", description="Data source")

    def get_temperature_fahrenheit(self) -> float:
        """Convert temperature to Fahrenheit."""
        return (self.temperature * 9/5) + 32

    def get_wind_speed_mph(self) -> float:
        """Convert wind speed to mph."""
        return self.wind_speed * 2.237

    def is_severe_weather(self) -> bool:
        """Check if weather conditions are severe."""
        severe_conditions = [
            WeatherCondition.THUNDERSTORM,
            WeatherCondition.SNOW,
        ]
        return (
            self.condition in severe_conditions or
            self.wind_speed > 20 or  # > 20 m/s is strong wind
            (self.rain_1h and self.rain_1h > 50) or  # Heavy rain
            (self.snow_1h and self.snow_1h > 10)  # Heavy snow
        )

    def get_clothing_recommendation(self) -> str:
        """Get clothing recommendation based on weather."""
        if self.temperature < 0:
            return "Heavy winter coat, gloves, hat, scarf"
        elif self.temperature < 10:
            return "Warm jacket, long pants"
        elif self.temperature < 20:
            return "Light jacket or sweater"
        elif self.temperature < 25:
            return "Comfortable casual wear"
        elif self.temperature < 30:
            return "Light, breathable clothing"
        else:
            return "Very light clothing, stay hydrated"

    def get_activity_suitability(self) -> Dict[str, bool]:
        """Determine suitability for various activities."""
        return {
            "outdoor_exercise": (
                not self.is_severe_weather() and
                10 <= self.temperature <= 25 and
                self.condition not in [WeatherCondition.RAIN, WeatherCondition.SNOW]
            ),
            "driving": not (
                self.visibility and self.visibility < 1000 or
                self.condition in [WeatherCondition.FOG, WeatherCondition.SNOW] or
                self.wind_speed > 25
            ),
            "outdoor_dining": (
                15 <= self.temperature <= 28 and
                self.condition in [WeatherCondition.CLEAR, WeatherCondition.PARTLY_CLOUDY] and
                self.wind_speed < 10
            ),
            "beach": (
                self.temperature >= 22 and
                self.condition in [WeatherCondition.CLEAR, WeatherCondition.PARTLY_CLOUDY] and
                self.wind_speed < 15
            ),
        }


class WeatherForecast(BaseModel):
    """Model for weather forecast data."""

    location: str = Field(..., description="Location name")
    forecast_date: datetime = Field(..., description="Forecast date")

    periods: List[WeatherData] = Field(default_factory=list, description="Forecast periods")

    daily_summary: Optional[str] = Field(None, description="Daily summary")
    weekly_summary: Optional[str] = Field(None, description="Weekly summary")

    alerts: List["WeatherAlertData"] = Field(default_factory=list, description="Weather alerts")

    def get_today_forecast(self) -> Optional[WeatherData]:
        """Get today's forecast."""
        today = datetime.now().date()
        for period in self.periods:
            if period.observation_time.date() == today:
                return period
        return None

    def get_upcoming_days(self, days: int = 5) -> List[WeatherData]:
        """Get forecast for upcoming days."""
        return self.periods[:days]

    def has_alerts(self) -> bool:
        """Check if there are any weather alerts."""
        return len(self.alerts) > 0

    def get_severe_alerts(self) -> List["WeatherAlertData"]:
        """Get severe weather alerts (warning level or higher)."""
        severe_levels = [WeatherAlert.WARNING, WeatherAlert.EMERGENCY]
        return [alert for alert in self.alerts if alert.severity in severe_levels]


class WeatherAlertData(BaseModel):
    """Model for weather alert data."""

    alert_id: str = Field(..., description="Alert identifier")
    severity: WeatherAlert = Field(..., description="Alert severity")

    event: str = Field(..., description="Alert event type")
    headline: str = Field(..., description="Alert headline")
    description: str = Field(..., description="Detailed description")

    start_time: datetime = Field(..., description="Alert start time")
    end_time: datetime = Field(..., description="Alert end time")

    areas: List[str] = Field(default_factory=list, description="Affected areas")

    urgency: str = Field("Unknown", description="Urgency level")
    certainty: str = Field("Unknown", description="Certainty level")

    instructions: Optional[str] = Field(None, description="Safety instructions")

    def is_active(self) -> bool:
        """Check if alert is currently active."""
        now = datetime.now()
        return self.start_time <= now <= self.end_time

    def calculate_importance(self) -> float:
        """Calculate alert importance."""
        severity_scores = {
            WeatherAlert.INFO: 0.2,
            WeatherAlert.ADVISORY: 0.4,
            WeatherAlert.WATCH: 0.6,
            WeatherAlert.WARNING: 0.8,
            WeatherAlert.EMERGENCY: 1.0,
        }

        score = severity_scores.get(self.severity, 0.5)

        # Boost if currently active
        if self.is_active():
            score = min(1.0, score + 0.2)

        # Boost based on urgency
        if self.urgency == "Immediate":
            score = min(1.0, score + 0.2)

        return score


class WeatherSummary(BaseModel):
    """Model for weather summary in daily minutes."""

    date: datetime = Field(default_factory=datetime.now, description="Summary date")

    current: Optional[WeatherData] = Field(None, description="Current weather")
    forecast: Optional[WeatherForecast] = Field(None, description="Weather forecast")

    highlights: List[str] = Field(default_factory=list, description="Weather highlights")
    recommendations: List[str] = Field(default_factory=list, description="Weather-based recommendations")

    commute_impact: Optional[str] = Field(None, description="Impact on commute")
    outdoor_plans_impact: Optional[str] = Field(None, description="Impact on outdoor plans")

    priority: Priority = Field(Priority.LOW, description="Weather priority")

    def generate_brief(self) -> str:
        """Generate weather brief."""
        if not self.current:
            return "Weather data unavailable"

        brief = f"🌡️ {self.current.temperature:.1f}°C ({self.current.get_temperature_fahrenheit():.1f}°F)\n"
        brief += f"☁️ {self.current.description.capitalize()}\n"

        if self.current.rain_1h and self.current.rain_1h > 0:
            brief += f"🌧️ Rain: {self.current.rain_1h}mm\n"

        if self.current.wind_speed > 10:
            brief += f"💨 Wind: {self.current.wind_speed:.1f} m/s\n"

        if self.forecast and self.forecast.has_alerts():
            alerts = self.forecast.get_severe_alerts()
            if alerts:
                brief += f"⚠️ {len(alerts)} weather alerts\n"

        if self.recommendations:
            brief += f"\n💡 {self.recommendations[0]}"

        return brief

    def calculate_importance(self) -> float:
        """Calculate weather importance for daily minutes."""
        score = 0.3  # Base score

        if self.current and self.current.is_severe_weather():
            score = 0.8

        if self.forecast and self.forecast.has_alerts():
            severe_alerts = self.forecast.get_severe_alerts()
            if severe_alerts:
                score = max(score, max(alert.calculate_importance() for alert in severe_alerts))

        # Boost if bad weather affects commute
        if self.commute_impact and "delay" in self.commute_impact.lower():
            score = min(1.0, score + 0.2)

        # Priority modifier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        return min(1.0, max(0.0, score))
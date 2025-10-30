"""Base models and enums for the application."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, field_validator


class DataSource(str, Enum):
    """Enum for data sources."""

    RSS = "rss"
    EMAIL = "email"
    CALENDAR = "calendar"
    SLACK = "slack"
    NEWS_API = "news_api"
    HACKERNEWS = "hackernews"
    REDDIT = "reddit"
    TWITTER = "twitter"
    CUSTOM = "custom"


class Priority(str, Enum):
    """Priority levels for items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class BaseModel(PydanticBaseModel):
    """Base model with common fields and configuration."""

    id: Optional[str] = Field(None, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
        "populate_by_name": True,
        "validate_assignment": True,
    }

    @field_validator("updated_at", mode="before")
    @classmethod
    def set_updated_at(cls, v: Optional[datetime]) -> datetime:
        """Set updated_at to current time if not provided."""
        return v or datetime.now()

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata key-value pair."""
        self.metadata[key] = value
        self.update_timestamp()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata = {}
        self.update_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """Create model from JSON string."""
        return cls.model_validate_json(json_str)
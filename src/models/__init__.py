"""Data models for the Daily Minutes application.

This module provides Pydantic models for:
- News and RSS feed data
- Email messages and threads
- Calendar events
- Chat/messaging (base and platform-specific)
- Aggregated daily summaries
"""

from src.models.base import BaseModel, DataSource, Priority
from src.models.calendar import CalendarEvent, CalendarReminder, EventType
from src.models.chat import (
    BaseChannel,
    BaseMessage,
    BaseThread,
    ChannelType,
    MessageType,
)
from src.models.email import (
    EmailAttachment,
    EmailFolder,
    EmailMessage,
    EmailStatus,
    EmailThread,
)
from src.models.news import NewsArticle, NewsSummary, RSSFeed
from src.models.slack import SlackChannel, SlackMessage, SlackThread, SlackWorkspace
from src.models.summary import (
    DailyMinutes,
    SectionSummary,
    SourceMetadata,
    SummarySection,
)

__all__ = [
    # Base models
    "BaseModel",
    "DataSource",
    "Priority",
    # News models
    "NewsArticle",
    "NewsSummary",
    "RSSFeed",
    # Email models
    "EmailMessage",
    "EmailThread",
    "EmailAttachment",
    "EmailStatus",
    "EmailFolder",
    # Calendar models
    "CalendarEvent",
    "CalendarReminder",
    "EventType",
    # Chat base models
    "BaseMessage",
    "BaseChannel",
    "BaseThread",
    "MessageType",
    "ChannelType",
    # Slack models
    "SlackMessage",
    "SlackChannel",
    "SlackThread",
    "SlackWorkspace",
    # Summary models
    "DailyMinutes",
    "SummarySection",
    "SectionSummary",
    "SourceMetadata",
]
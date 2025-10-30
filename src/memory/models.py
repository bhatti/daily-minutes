"""Data models for RAG Memory System.

This module defines the data structures for storing and retrieving memories:
- Daily briefs
- Action items
- Meeting contexts
- User preferences
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class MemoryType(Enum):
    """Types of memories that can be stored."""
    DAILY_BRIEF = "daily_brief"
    ACTION_ITEM = "action_item"
    MEETING_CONTEXT = "meeting_context"
    USER_PREFERENCE = "user_preference"
    EMAIL_SUMMARY = "email_summary"
    CONVERSATION = "conversation"


class ActionItemStatus(Enum):
    """Status of action items."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Memory:
    """Base memory unit for vector storage."""

    id: str  # Unique identifier
    type: MemoryType  # Type of memory
    content: str  # Text content to be embedded
    metadata: Dict[str, Any]  # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None  # Vector embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class DailyBriefMemory(Memory):
    """Memory of a daily brief."""

    def __init__(
        self,
        id: str,
        summary: str,
        key_points: List[str],
        action_items: List[str],
        emails_count: int,
        calendar_events_count: int,
        news_items_count: int,
        timestamp: datetime = None,
    ):
        """Initialize daily brief memory."""
        timestamp = timestamp or datetime.now()

        # Create content for embedding
        content = f"""Daily Brief from {timestamp.strftime('%Y-%m-%d')}

Summary: {summary}

Key Points:
{chr(10).join(f'- {point}' for point in key_points)}

Action Items:
{chr(10).join(f'- {item}' for item in action_items)}

Statistics: {emails_count} emails, {calendar_events_count} events, {news_items_count} news items
"""

        metadata = {
            "summary": summary,
            "key_points": key_points,
            "action_items": action_items,
            "emails_count": emails_count,
            "calendar_events_count": calendar_events_count,
            "news_items_count": news_items_count,
            "date": timestamp.strftime('%Y-%m-%d'),
        }

        super().__init__(
            id=id,
            type=MemoryType.DAILY_BRIEF,
            content=content,
            metadata=metadata,
            timestamp=timestamp,
        )


@dataclass
class ActionItemMemory(Memory):
    """Memory of an action item."""

    def __init__(
        self,
        id: str,
        description: str,
        status: ActionItemStatus,
        source: str,  # Where it came from (email, calendar, etc.)
        due_date: Optional[datetime] = None,
        completed_date: Optional[datetime] = None,
        timestamp: datetime = None,
    ):
        """Initialize action item memory."""
        timestamp = timestamp or datetime.now()

        # Create content for embedding
        content = f"""Action Item: {description}
Status: {status.value}
Source: {source}
Created: {timestamp.strftime('%Y-%m-%d')}
{f'Due: {due_date.strftime("%Y-%m-%d")}' if due_date else ''}
{f'Completed: {completed_date.strftime("%Y-%m-%d")}' if completed_date else ''}
"""

        metadata = {
            "description": description,
            "status": status.value,
            "source": source,
            "due_date": due_date.isoformat() if due_date else None,
            "completed_date": completed_date.isoformat() if completed_date else None,
        }

        super().__init__(
            id=id,
            type=MemoryType.ACTION_ITEM,
            content=content,
            metadata=metadata,
            timestamp=timestamp,
        )


@dataclass
class MeetingContextMemory(Memory):
    """Memory of a meeting context."""

    def __init__(
        self,
        id: str,
        meeting_title: str,
        meeting_date: datetime,
        attendees: List[str],
        description: Optional[str] = None,
        preparation_notes: Optional[List[str]] = None,
        location: Optional[str] = None,
        timestamp: datetime = None,
    ):
        """Initialize meeting context memory."""
        timestamp = timestamp or datetime.now()

        # Create content for embedding
        content = f"""Meeting: {meeting_title}
Date: {meeting_date.strftime('%Y-%m-%d %H:%M')}
Attendees: {', '.join(attendees)}
{f'Location: {location}' if location else ''}
{f'Description: {description}' if description else ''}

{f'Preparation Notes:\n' + chr(10).join(f'- {note}' for note in preparation_notes) if preparation_notes else ''}
"""

        metadata = {
            "meeting_title": meeting_title,
            "meeting_date": meeting_date.isoformat(),
            "attendees": attendees,
            "description": description,
            "preparation_notes": preparation_notes or [],
            "location": location,
        }

        super().__init__(
            id=id,
            type=MemoryType.MEETING_CONTEXT,
            content=content,
            metadata=metadata,
            timestamp=timestamp,
        )


@dataclass
class UserPreferenceMemory(Memory):
    """Memory of user preferences."""

    def __init__(
        self,
        id: str,
        preference_key: str,
        preference_value: Any,
        category: str,  # e.g., "brief_style", "notification", "priority"
        user_id: str = "default",
        confidence_score: float = 1.0,
        timestamp: datetime = None,
    ):
        """Initialize user preference memory."""
        timestamp = timestamp or datetime.now()

        # Create content for embedding
        content = f"""User Preference: {preference_key}
Category: {category}
Value: {preference_value}
User: {user_id}
Confidence: {confidence_score}
Set: {timestamp.strftime('%Y-%m-%d')}
"""

        metadata = {
            "preference_key": preference_key,
            "preference_value": preference_value,
            "category": category,
            "user_id": user_id,
            "confidence": confidence_score,
            "key": preference_key,  # Add key alias for easier lookup
            "value": preference_value,  # Add value alias
        }

        super().__init__(
            id=id,
            type=MemoryType.USER_PREFERENCE,
            content=content,
            metadata=metadata,
            timestamp=timestamp,
        )


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""

    memory: Memory
    relevance_score: float  # Similarity/relevance score
    distance: Optional[float] = None  # Vector distance (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
            "distance": self.distance,
        }

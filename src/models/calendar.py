"""Models for calendar events and reminders."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import EmailStr, Field, HttpUrl, field_validator

from src.models.base import BaseModel, Priority


class EventStatus(str, Enum):
    """Event status enum."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class EventType(str, Enum):
    """Event type enum."""

    MEETING = "meeting"
    APPOINTMENT = "appointment"
    TASK = "task"
    REMINDER = "reminder"
    BIRTHDAY = "birthday"
    HOLIDAY = "holiday"
    MILESTONE = "milestone"  # Added for work releases, project deadlines
    STANDUP = "standup"      # Added for routine meetings
    ONE_ON_ONE = "one_on_one"
    OTHER = "other"


class RecurrenceFrequency(str, Enum):
    """Recurrence frequency enum."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class CalendarEvent(BaseModel):
    """Model for a calendar event."""

    event_id: str = Field(..., description="Unique event ID")
    calendar_id: Optional[str] = Field(None, description="Calendar ID")

    title: str = Field(..., description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    event_type: EventType = Field(EventType.OTHER, description="Event type")

    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    all_day: bool = Field(False, description="All-day event flag")
    timezone: str = Field("UTC", description="Event timezone")

    location: Optional[str] = Field(None, description="Event location")
    location_url: Optional[HttpUrl] = Field(None, description="Location URL (maps, etc.)")

    organizer_email: Optional[EmailStr] = Field(None, description="Organizer email")
    organizer_name: Optional[str] = Field(None, description="Organizer name")

    attendees: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of attendees with email and response status"
    )

    status: EventStatus = Field(EventStatus.CONFIRMED, description="Event status")
    priority: Priority = Field(Priority.MEDIUM, description="Event priority")

    meeting_url: Optional[HttpUrl] = Field(None, description="Virtual meeting URL")
    conference_data: Dict[str, str] = Field(
        default_factory=dict,
        description="Conference call details"
    )

    is_recurring: bool = Field(False, description="Recurring event flag")
    recurrence_frequency: Optional[RecurrenceFrequency] = Field(
        None,
        description="Recurrence frequency"
    )
    recurrence_interval: Optional[int] = Field(None, gt=0, description="Recurrence interval")
    recurrence_end: Optional[datetime] = Field(None, description="Recurrence end date")

    reminders: List["CalendarReminder"] = Field(
        default_factory=list,
        description="Event reminders"
    )

    categories: List[str] = Field(default_factory=list, description="Event categories/tags")
    attachments: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Event attachments"
    )

    is_private: bool = Field(False, description="Private event flag")
    is_tentative: bool = Field(False, description="Tentative event flag")
    response_required: bool = Field(False, description="Response required flag")
    user_response: Optional[str] = Field(None, description="User's response (yes/no/maybe)")

    preparation_time: Optional[int] = Field(
        None,
        ge=0,
        description="Preparation time needed in minutes"
    )
    travel_time: Optional[int] = Field(None, ge=0, description="Travel time needed in minutes")

    notes: Optional[str] = Field(None, description="Personal notes")
    action_items: List[str] = Field(default_factory=list, description="Action items")

    # Additional fields for importance distinction
    is_routine: bool = Field(False, description="Mark as routine/regular event")
    importance_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Manual importance override (0-1)"
    )

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v: datetime, info) -> datetime:
        """Validate end time is after start time."""
        if "start_time" in info.data and v <= info.data["start_time"]:
            raise ValueError("End time must be after start time")
        return v

    @field_validator("categories", mode="before")
    @classmethod
    def clean_categories(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate categories."""
        if not v:
            return []
        return list(set(cat.strip().lower() for cat in v if cat.strip()))

    def get_duration(self) -> timedelta:
        """Get event duration."""
        return self.end_time - self.start_time

    def get_duration_hours(self) -> float:
        """Get event duration in hours."""
        return self.get_duration().total_seconds() / 3600

    def is_upcoming(self, hours: int = 24) -> bool:
        """Check if event is upcoming within specified hours."""
        now = datetime.now()
        return now < self.start_time <= now + timedelta(hours=hours)

    def is_ongoing(self) -> bool:
        """Check if event is currently ongoing."""
        now = datetime.now()
        return self.start_time <= now < self.end_time

    def is_past(self) -> bool:
        """Check if event is in the past."""
        return self.end_time < datetime.now()

    def has_conflict(self, other: "CalendarEvent") -> bool:
        """Check if this event conflicts with another event."""
        # Events don't conflict if either is cancelled
        if self.status == EventStatus.CANCELLED or other.status == EventStatus.CANCELLED:
            return False

        # Check time overlap
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)

    def add_attendee(self, email: str, name: Optional[str] = None, response: str = "pending") -> None:
        """Add an attendee to the event."""
        attendee = {
            "email": email,
            "name": name or email,
            "response": response
        }
        self.attendees.append(attendee)
        self.update_timestamp()

    def update_response(self, response: str) -> None:
        """Update user's response to the event."""
        valid_responses = ["yes", "no", "maybe", "accepted", "declined", "tentative"]
        if response.lower() in valid_responses:
            self.user_response = response.lower()
            if response.lower() in ["no", "declined"]:
                self.status = EventStatus.CANCELLED
            elif response.lower() in ["maybe", "tentative"]:
                self.is_tentative = True
            self.update_timestamp()

    def add_reminder(self, reminder: "CalendarReminder") -> None:
        """Add a reminder to the event."""
        self.reminders.append(reminder)
        self.update_timestamp()

    def get_total_time_needed(self) -> int:
        """Get total time needed including preparation and travel (in minutes)."""
        event_duration = int(self.get_duration().total_seconds() / 60)
        prep_time = self.preparation_time or 0
        travel = self.travel_time or 0
        return event_duration + prep_time + travel

    def calculate_importance(self) -> float:
        """Calculate event importance score with smart distinction between routine and special events.

        This method intelligently weighs different factors:
        - Special events (birthdays, milestones) get higher base scores
        - Routine/recurring events get lower base scores
        - Upcoming timing increases importance
        - Response requirements and action items add urgency
        """
        # Check for manual override first
        if self.importance_override is not None:
            return self.importance_override

        # Base score based on event type
        special_event_types = {
            EventType.BIRTHDAY: 0.8,
            EventType.HOLIDAY: 0.7,
            EventType.MILESTONE: 0.9,  # Work releases, project deadlines
            EventType.APPOINTMENT: 0.7,  # Doctor appointments, etc.
        }

        routine_event_types = {
            EventType.STANDUP: 0.3,
            EventType.MEETING: 0.4,
        }

        # Start with type-based score
        if self.event_type in special_event_types:
            score = special_event_types[self.event_type]
        elif self.event_type in routine_event_types:
            score = routine_event_types[self.event_type]
        else:
            score = 0.5  # Default for OTHER

        # Reduce score for routine/recurring events
        if self.is_routine or (self.is_recurring and self.recurrence_frequency == RecurrenceFrequency.DAILY):
            score *= 0.5  # Halve importance for daily routines
        elif self.is_recurring and self.recurrence_frequency == RecurrenceFrequency.WEEKLY:
            score *= 0.7  # Reduce for weekly routines

        # Priority modifier (but less impact for routine events)
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }

        if not self.is_routine:
            score *= priority_multipliers.get(self.priority, 1.0)
        else:
            # Routine events get smaller priority boost
            score *= (priority_multipliers.get(self.priority, 1.0) * 0.5 + 0.5)

        # Time-based urgency (more important as event approaches)
        if self.is_upcoming(hours=3):
            score = min(1.0, score + 0.3)
        elif self.is_upcoming(hours=24):
            score = min(1.0, score + 0.2)
        elif self.is_upcoming(hours=72):  # 3 days for birthdays, etc.
            if self.event_type in [EventType.BIRTHDAY, EventType.MILESTONE]:
                score = min(1.0, score + 0.2)

        # Response required adds urgency
        if self.response_required and not self.user_response:
            score = min(1.0, score + 0.2)

        # Large meetings or important attendees
        if len(self.attendees) > 10:  # Large meeting
            score = min(1.0, score + 0.1)
        elif len(self.attendees) > 50:  # All-hands, etc.
            score = min(1.0, score + 0.2)

        # Action items indicate importance
        if self.action_items:
            score = min(1.0, score + 0.1 * min(len(self.action_items), 3))

        # Cap routine events at 0.6 unless urgent
        if self.is_routine and self.priority != Priority.URGENT:
            score = min(0.6, score)

        return min(1.0, max(0.0, score))

    def is_special_event(self) -> bool:
        """Check if this is a special (non-routine) event."""
        special_types = [EventType.BIRTHDAY, EventType.HOLIDAY, EventType.MILESTONE, EventType.APPOINTMENT]

        # Special if: special type, not routine, not frequently recurring, or high priority
        return (
            self.event_type in special_types or
            not self.is_routine or
            (self.is_recurring and self.recurrence_frequency in [RecurrenceFrequency.YEARLY, RecurrenceFrequency.MONTHLY]) or
            self.priority in [Priority.HIGH, Priority.URGENT] or
            (self.importance_override and self.importance_override > 0.7)
        )

    def generate_summary(self) -> str:
        """Generate event summary."""
        # Add special event indicator
        special_indicator = "⭐ " if self.is_special_event() else ""

        summary = f"{special_indicator}{self.title}\n"
        summary += f"When: {self.start_time.strftime('%Y-%m-%d %H:%M')} - "
        summary += f"{self.end_time.strftime('%H:%M')}\n"

        if self.event_type != EventType.OTHER:
            summary += f"Type: {self.event_type.value.replace('_', ' ').title()}\n"

        if self.location:
            summary += f"Where: {self.location}\n"

        if self.meeting_url:
            summary += f"Meeting: {self.meeting_url}\n"

        if self.attendees:
            summary += f"Attendees: {len(self.attendees)}\n"

        if self.is_recurring:
            summary += f"Recurring: {self.recurrence_frequency.value}\n"

        if self.description:
            summary += f"\n{self.description}\n"

        if self.action_items:
            summary += "\nAction Items:\n"
            for item in self.action_items:
                summary += f"- {item}\n"

        return summary


class CalendarReminder(BaseModel):
    """Model for calendar event reminder."""

    reminder_id: Optional[str] = Field(None, description="Unique reminder ID")
    event_id: Optional[str] = Field(None, description="Associated event ID")

    minutes_before: int = Field(15, gt=0, description="Minutes before event")
    reminder_type: str = Field("notification", description="Reminder type (notification/email/sms)")

    message: Optional[str] = Field(None, description="Custom reminder message")
    is_sent: bool = Field(False, description="Whether reminder has been sent")
    sent_at: Optional[datetime] = Field(None, description="When reminder was sent")

    def get_reminder_time(self, event_start: datetime) -> datetime:
        """Calculate when reminder should be sent."""
        return event_start - timedelta(minutes=self.minutes_before)

    def should_send(self, event_start: datetime) -> bool:
        """Check if reminder should be sent now."""
        if self.is_sent:
            return False

        reminder_time = self.get_reminder_time(event_start)
        return datetime.now() >= reminder_time

    def mark_as_sent(self) -> None:
        """Mark reminder as sent."""
        self.is_sent = True
        self.sent_at = datetime.now()
        self.update_timestamp()

    def format_message(self, event: CalendarEvent) -> str:
        """Format reminder message for the event."""
        if self.message:
            return self.message

        # Add special indicator for important events
        prefix = "⭐ Important: " if event.is_special_event() else "Reminder: "
        message = f"{prefix}{event.title}"

        if self.minutes_before < 60:
            message += f" in {self.minutes_before} minutes"
        else:
            hours = self.minutes_before // 60
            message += f" in {hours} hour{'s' if hours > 1 else ''}"

        if event.location:
            message += f" at {event.location}"

        if event.meeting_url:
            message += f" - Join: {event.meeting_url}"

        return message
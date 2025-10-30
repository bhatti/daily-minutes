"""Data models for Daily Minutes application.

All models use Pydantic v2 for validation and serialization.

RLHF Support:
- boost_labels: Keywords/labels that increase importance (user preferences)
- filter_labels: Keywords/labels that decrease importance or filter out
- These enable Reinforcement Learning from Human Feedback
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from src.core.logging import get_logger

logger = get_logger(__name__)


class ImportanceScoringMixin(BaseModel):
    """Mixin for models that support importance scoring with RLHF.

    Provides common fields and methods for importance scoring based on:
    - AI-computed base score
    - User-defined boost labels (increases importance)
    - User-defined filter labels (decreases importance)

    TODO: Implement full RLHF pipeline:
    - Learn user preferences over time
    - Adaptive label suggestion based on user feedback
    - Multi-model ensemble for importance scoring
    - A/B testing for scoring algorithms
    """

    # Base importance score (0.0-1.0)
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="AI-computed importance score"
    )

    # RLHF fields for user feedback
    boost_labels: Set[str] = Field(
        default_factory=set,
        description="Labels/keywords that boost importance (RLHF)"
    )
    filter_labels: Set[str] = Field(
        default_factory=set,
        description="Labels/keywords that filter/reduce importance (RLHF)"
    )

    def apply_rlhf_boost(
        self,
        content_text: str,
        boost_multiplier: float = 0.3,
        filter_multiplier: float = -0.3
    ) -> float:
        """
        Apply RLHF label-based boosting to importance score.

        TODO: This is a simple implementation. Future enhancements:
        - Use LLM to understand semantic similarity to labels
        - Learn optimal multipliers from user feedback
        - Consider label combinations (e.g., "urgent" + "client")
        - Time-based decay for old preferences

        Args:
            content_text: Text to check for label matches (case-insensitive)
            boost_multiplier: How much to increase score for boost labels
            filter_multiplier: How much to decrease score for filter labels

        Returns:
            Adjusted importance score (clamped to 0.0-1.0)
        """
        content_lower = content_text.lower()
        adjusted_score = self.importance_score

        # Apply boosts
        for label in self.boost_labels:
            if label.lower() in content_lower:
                adjusted_score += boost_multiplier
                logger.debug(
                    "rlhf_boost_applied",
                    label=label,
                    score_before=self.importance_score,
                    score_after=adjusted_score
                )

        # Apply filters
        for label in self.filter_labels:
            if label.lower() in content_lower:
                adjusted_score += filter_multiplier
                logger.debug(
                    "rlhf_filter_applied",
                    label=label,
                    score_before=self.importance_score,
                    score_after=adjusted_score
                )

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_score))


class EmailMessage(ImportanceScoringMixin):
    """Email message with AI-enhanced metadata and RLHF support.

    Attributes:
        id: Unique email identifier
        subject: Email subject line
        sender: Sender email address
        received_at: Timestamp when email was received
        body: Full email body content
        snippet: Short preview/snippet of email
        labels: Gmail-style labels (INBOX, IMPORTANT, etc.)
        importance_score: AI-computed importance (0.0-1.0) [from mixin]
        boost_labels: User-defined labels to boost importance [from mixin]
        filter_labels: User-defined labels to filter/reduce importance [from mixin]
        has_action_items: Whether email contains action items
        action_items: Extracted action items/todos
        category: Email category (work, personal, spam, etc.)
    """

    id: str = Field(..., min_length=1, description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: EmailStr = Field(..., description="Sender email address")
    received_at: datetime = Field(..., description="Email received timestamp")
    body: str = Field(..., description="Full email body content")
    snippet: Optional[str] = Field(None, description="Email preview text")
    labels: List[str] = Field(default_factory=list, description="Email labels")
    is_read: bool = Field(default=False, description="Whether email has been read")

    # AI-enhanced fields
    has_action_items: bool = Field(
        default=False,
        description="Whether email contains action items"
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Extracted action items"
    )
    ai_summary: Optional[str] = Field(
        None,
        description="AI-generated summary/key points of email"
    )
    category: Optional[str] = Field(
        None,
        description="Email category (work, personal, spam, etc.)"
    )

    def model_post_init(self, __context) -> None:
        """Log model creation for observability."""
        logger.debug(
            "email_message_created",
            email_id=self.id,
            subject=self.subject[:50] if len(self.subject) > 50 else self.subject,
            importance=self.importance_score,
            has_actions=self.has_action_items,
            action_count=len(self.action_items)
        )


class CalendarEvent(ImportanceScoringMixin):
    """Calendar event with importance scoring and RLHF support.

    Attributes:
        id: Unique event identifier
        summary: Event title/summary
        description: Detailed event description
        start_time: Event start timestamp
        end_time: Event end timestamp
        location: Event location
        attendees: List of attendee email addresses
        is_recurring: Whether event is recurring
        recurrence_rule: iCal recurrence rule (RRULE)
        importance_score: AI-computed importance (0.0-1.0) [from mixin]
        boost_labels: User-defined labels to boost importance [from mixin]
        filter_labels: User-defined labels to filter/reduce importance [from mixin]
        requires_preparation: Whether event requires prep work
        is_focus_time: Whether this is dedicated focus time
    """

    id: str = Field(..., min_length=1, description="Unique event identifier")
    summary: str = Field(..., description="Event title/summary")
    description: Optional[str] = Field(None, description="Event description")
    start_time: datetime = Field(..., description="Event start timestamp")
    end_time: datetime = Field(..., description="Event end timestamp")
    location: Optional[str] = Field(None, description="Event location")
    attendees: List[EmailStr] = Field(
        default_factory=list,
        description="Attendee email addresses"
    )
    is_recurring: bool = Field(default=False, description="Is recurring event")
    recurrence_rule: Optional[str] = Field(
        None,
        description="iCal recurrence rule (RRULE)"
    )

    # AI-enhanced fields
    requires_preparation: bool = Field(
        default=False,
        description="Requires preparation"
    )
    preparation_notes: List[str] = Field(
        default_factory=list,
        description="Preparation notes and action items"
    )
    is_focus_time: bool = Field(
        default=False,
        description="Dedicated focus time"
    )

    @property
    def duration_minutes(self) -> int:
        """Calculate event duration in minutes.

        Returns:
            Duration in minutes
        """
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    def model_post_init(self, __context) -> None:
        """Log model creation for observability."""
        logger.debug(
            "calendar_event_created",
            event_id=self.id,
            summary=self.summary[:50] if len(self.summary) > 50 else self.summary,
            duration_minutes=self.duration_minutes,
            importance=self.importance_score,
            requires_prep=self.requires_preparation,
            is_focus=self.is_focus_time,
            attendee_count=len(self.attendees)
        )

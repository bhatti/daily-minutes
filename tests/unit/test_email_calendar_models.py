#!/usr/bin/env python3
"""Unit tests for Email and Calendar data models (TDD approach).

Following TDD: Write tests first, then implement models.
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError


def test_email_message_creation():
    """Test EmailMessage model creation with valid data."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_123",
        subject="Important: Q4 Planning",
        sender="boss@company.com",
        received_at=datetime.now(),
        body="Let's meet to discuss Q4 planning...",
        importance_score=0.9,
        has_action_items=True
    )

    assert email.id == "msg_123"
    assert email.subject == "Important: Q4 Planning"
    assert email.sender == "boss@company.com"
    assert email.importance_score == 0.9
    assert email.has_action_items is True


def test_email_message_with_snippet():
    """Test EmailMessage with optional snippet field."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_456",
        subject="Test Email",
        sender="test@example.com",
        received_at=datetime.now(),
        body="Full email body text here...",
        snippet="Email preview text..."
    )

    assert email.snippet == "Email preview text..."


def test_email_message_with_labels():
    """Test EmailMessage with Gmail-style labels."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_789",
        subject="Project Update",
        sender="team@company.com",
        received_at=datetime.now(),
        body="Project status update...",
        labels=["IMPORTANT", "WORK", "STARRED"]
    )

    assert len(email.labels) == 3
    assert "IMPORTANT" in email.labels


def test_email_message_with_action_items():
    """Test EmailMessage with extracted action items."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_101",
        subject="Action Items from Meeting",
        sender="manager@company.com",
        received_at=datetime.now(),
        body="Please complete the following tasks...",
        has_action_items=True,
        action_items=[
            "Review Q4 budget proposal",
            "Schedule team meeting",
            "Update project timeline"
        ]
    )

    assert email.has_action_items is True
    assert len(email.action_items) == 3
    assert "Review Q4 budget proposal" in email.action_items


def test_email_message_categorization():
    """Test EmailMessage category field."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_202",
        subject="Weekend Plans",
        sender="friend@example.com",
        received_at=datetime.now(),
        body="Want to grab coffee this weekend?",
        category="personal"
    )

    assert email.category == "personal"


def test_email_message_defaults():
    """Test EmailMessage default values."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_303",
        subject="Minimal Email",
        sender="test@example.com",
        received_at=datetime.now(),
        body="Body text"
    )

    # Check defaults
    assert email.snippet is None
    assert email.labels == []
    assert email.importance_score == 0.5  # Default medium importance
    assert email.has_action_items is False
    assert email.action_items == []
    assert email.category is None


def test_email_message_validation_empty_id():
    """Test EmailMessage validation rejects empty ID."""
    from src.core.models import EmailMessage

    with pytest.raises(ValidationError) as exc_info:
        EmailMessage(
            id="",  # Invalid: empty ID
            subject="Test",
            sender="test@example.com",
            received_at=datetime.now(),
            body="Test body"
        )

    assert "id" in str(exc_info.value).lower()


def test_email_message_validation_invalid_email():
    """Test EmailMessage validation rejects invalid email format."""
    from src.core.models import EmailMessage

    with pytest.raises(ValidationError) as exc_info:
        EmailMessage(
            id="msg_404",
            subject="Test",
            sender="not-an-email",  # Invalid email format
            received_at=datetime.now(),
            body="Test body"
        )

    # Pydantic v2 should raise validation error for invalid email
    assert "email" in str(exc_info.value).lower() or "sender" in str(exc_info.value).lower()


def test_email_message_validation_importance_score_range():
    """Test EmailMessage importance_score must be between 0 and 1."""
    from src.core.models import EmailMessage

    # Test score > 1
    with pytest.raises(ValidationError) as exc_info:
        EmailMessage(
            id="msg_505",
            subject="Test",
            sender="test@example.com",
            received_at=datetime.now(),
            body="Test body",
            importance_score=1.5  # Invalid: > 1.0
        )

    assert "importance_score" in str(exc_info.value).lower()

    # Test score < 0
    with pytest.raises(ValidationError) as exc_info:
        EmailMessage(
            id="msg_606",
            subject="Test",
            sender="test@example.com",
            received_at=datetime.now(),
            body="Test body",
            importance_score=-0.1  # Invalid: < 0.0
        )

    assert "importance_score" in str(exc_info.value).lower()


def test_calendar_event_creation():
    """Test CalendarEvent model creation with valid data."""
    from src.core.models import CalendarEvent

    start = datetime.now()
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id="evt_123",
        summary="Q4 Planning Meeting",
        start_time=start,
        end_time=end,
        attendees=["alice@company.com", "bob@company.com"],
        importance_score=0.85
    )

    assert event.id == "evt_123"
    assert event.summary == "Q4 Planning Meeting"
    assert event.start_time == start
    assert event.end_time == end
    assert len(event.attendees) == 2
    assert event.importance_score == 0.85


def test_calendar_event_duration_calculation():
    """Test CalendarEvent duration_minutes property."""
    from src.core.models import CalendarEvent

    start = datetime(2025, 10, 26, 10, 0, 0)
    end = datetime(2025, 10, 26, 11, 30, 0)  # 90 minutes later

    event = CalendarEvent(
        id="evt_456",
        summary="Team Standup",
        start_time=start,
        end_time=end
    )

    assert event.duration_minutes == 90


def test_calendar_event_with_description():
    """Test CalendarEvent with optional description."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_789",
        summary="Project Review",
        description="Quarterly project review and planning session",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2)
    )

    assert event.description == "Quarterly project review and planning session"


def test_calendar_event_with_location():
    """Test CalendarEvent with location field."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_101",
        summary="Client Meeting",
        location="Conference Room B",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )

    assert event.location == "Conference Room B"


def test_calendar_event_recurring():
    """Test CalendarEvent with recurring settings."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_202",
        summary="Daily Standup",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=15),
        is_recurring=True,
        recurrence_rule="RRULE:FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR"
    )

    assert event.is_recurring is True
    assert "FREQ=DAILY" in event.recurrence_rule


def test_calendar_event_requires_preparation():
    """Test CalendarEvent preparation flag."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_303",
        summary="Board Presentation",
        start_time=datetime.now() + timedelta(days=2),
        end_time=datetime.now() + timedelta(days=2, hours=1),
        requires_preparation=True,
        importance_score=0.95
    )

    assert event.requires_preparation is True
    assert event.importance_score == 0.95


def test_calendar_event_focus_time():
    """Test CalendarEvent focus time detection."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_404",
        summary="Deep Work - Code Review",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2),
        is_focus_time=True
    )

    assert event.is_focus_time is True


def test_calendar_event_defaults():
    """Test CalendarEvent default values."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_505",
        summary="Minimal Event",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )

    # Check defaults
    assert event.description is None
    assert event.location is None
    assert event.attendees == []
    assert event.is_recurring is False
    assert event.recurrence_rule is None
    assert event.importance_score == 0.5
    assert event.requires_preparation is False
    assert event.is_focus_time is False


def test_calendar_event_validation_empty_id():
    """Test CalendarEvent validation rejects empty ID."""
    from src.core.models import CalendarEvent

    with pytest.raises(ValidationError) as exc_info:
        CalendarEvent(
            id="",  # Invalid: empty ID
            summary="Test Event",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

    assert "id" in str(exc_info.value).lower()


def test_calendar_event_validation_invalid_attendee_email():
    """Test CalendarEvent validates attendee email addresses."""
    from src.core.models import CalendarEvent

    with pytest.raises(ValidationError) as exc_info:
        CalendarEvent(
            id="evt_606",
            summary="Test Event",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            attendees=["valid@example.com", "not-an-email"]  # One invalid
        )

    assert "attendees" in str(exc_info.value).lower() or "email" in str(exc_info.value).lower()


def test_calendar_event_validation_importance_score_range():
    """Test CalendarEvent importance_score must be between 0 and 1."""
    from src.core.models import CalendarEvent

    # Test score > 1
    with pytest.raises(ValidationError) as exc_info:
        CalendarEvent(
            id="evt_707",
            summary="Test Event",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            importance_score=2.0  # Invalid: > 1.0
        )

    assert "importance_score" in str(exc_info.value).lower()


def test_email_rlhf_boost_labels():
    """Test EmailMessage RLHF boost labels functionality."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_rlhf1",
        subject="Urgent: Client meeting tomorrow",
        sender="boss@company.com",
        received_at=datetime.now(),
        body="Please prepare materials for the urgent client meeting tomorrow.",
        importance_score=0.5,
        boost_labels={"urgent", "client"}  # User cares about these
    )

    # Apply RLHF boosting
    adjusted_score = email.apply_rlhf_boost(
        content_text=f"{email.subject} {email.body}"
    )

    # Should boost score due to "urgent" and "client" matches
    assert adjusted_score > email.importance_score
    assert adjusted_score <= 1.0


def test_email_rlhf_filter_labels():
    """Test EmailMessage RLHF filter labels functionality."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_rlhf2",
        subject="Newsletter: Weekly updates",
        sender="marketing@company.com",
        received_at=datetime.now(),
        body="Here are this week's newsletter updates...",
        importance_score=0.7,
        filter_labels={"newsletter", "marketing"}  # User wants to filter these
    )

    # Apply RLHF filtering
    adjusted_score = email.apply_rlhf_boost(
        content_text=f"{email.subject} {email.body}"
    )

    # Should reduce score due to "newsletter" match
    assert adjusted_score < email.importance_score
    assert adjusted_score >= 0.0


def test_calendar_rlhf_boost_labels():
    """Test CalendarEvent RLHF boost labels functionality."""
    from src.core.models import CalendarEvent

    event = CalendarEvent(
        id="evt_rlhf1",
        summary="Client presentation - Board review",
        description="Quarterly board presentation for key client",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2),
        importance_score=0.6,
        boost_labels={"client", "board", "presentation"}
    )

    # Apply RLHF boosting
    adjusted_score = event.apply_rlhf_boost(
        content_text=f"{event.summary} {event.description}"
    )

    # Should boost significantly (3 label matches)
    assert adjusted_score > event.importance_score
    assert adjusted_score <= 1.0


def test_rlhf_mixed_boost_and_filter():
    """Test RLHF with both boost and filter labels."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_rlhf3",
        subject="Newsletter about urgent security update",
        sender="security@company.com",
        received_at=datetime.now(),
        body="This newsletter contains urgent information about security...",
        importance_score=0.5,
        boost_labels={"urgent", "security"},  # Important topics
        filter_labels={"newsletter"}  # Usually low priority
    )

    # Apply RLHF
    adjusted_score = email.apply_rlhf_boost(
        content_text=f"{email.subject} {email.body}"
    )

    # Net effect: +0.6 boost (urgent + security) - 0.3 filter (newsletter) = +0.3
    # So 0.5 + 0.3 = 0.8
    assert adjusted_score > email.importance_score  # Net positive boost


def test_rlhf_case_insensitive():
    """Test RLHF label matching is case-insensitive."""
    from src.core.models import EmailMessage

    email = EmailMessage(
        id="msg_rlhf4",
        subject="URGENT: System Alert",
        sender="alerts@company.com",
        received_at=datetime.now(),
        body="URGENT system maintenance required...",
        importance_score=0.5,
        boost_labels={"urgent"}  # lowercase label
    )

    # Apply RLHF (should match "URGENT" in uppercase text)
    adjusted_score = email.apply_rlhf_boost(
        content_text=email.subject
    )

    assert adjusted_score > email.importance_score


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

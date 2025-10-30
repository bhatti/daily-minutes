#!/usr/bin/env python3
"""Integration tests for Email and Calendar tabs in Streamlit app.

These tests verify end-to-end functionality of the email and calendar features,
including:
- Tab rendering
- Data fetching
- Filtering and search
- Error handling
- Integration with services
"""

import asyncio
import sys
import pytest
sys.path.append('.')

from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from src.core.models import EmailMessage, CalendarEvent


class TestEmailTabIntegration:
    """Integration tests for Email tab functionality."""

    @pytest.mark.asyncio
    async def test_email_tab_can_import(self):
        """Test that email tab components can be imported."""
        try:
            from src.ui.components.email_components import render_email_tab
            assert render_email_tab is not None
            print("✅ Email tab imports successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import email tab: {e}")

    @pytest.mark.asyncio
    async def test_email_service_integration(self):
        """Test email service can fetch emails."""
        from src.services.email_service import get_email_service

        service = get_email_service()
        assert service is not None
        print("✅ Email service initialized")

        # Test that fetch_emails method exists and is callable
        assert hasattr(service, 'fetch_emails')
        assert callable(service.fetch_emails)
        print("✅ Email service has fetch_emails method")

    @pytest.mark.asyncio
    async def test_email_formatter_integration(self):
        """Test email formatter integrates with components."""
        from src.ui.formatters.email_formatter import get_email_formatter
        from src.ui.components.email_components import prepare_email_display_data

        # Create sample email
        email = EmailMessage(
            id="test-1",
            subject="Test Email",
            sender="test@example.com",
            received_at=datetime.now(),
            body="Test body",
            importance_score=0.8,
            is_read=False
        )

        # Test formatter works
        formatter = get_email_formatter()
        assert formatter is not None

        # Test component helper uses formatter
        display_data = prepare_email_display_data(email)
        assert display_data is not None
        assert "subject" in display_data
        assert "sender_display" in display_data
        print("✅ Email formatter integrates with components")

    @pytest.mark.asyncio
    async def test_email_filtering_integration(self):
        """Test email filtering works end-to-end."""
        from src.ui.components.email_components import should_show_email

        # Create test emails
        unread_important = EmailMessage(
            id="1",
            subject="Important",
            sender="boss@example.com",
            received_at=datetime.now(),
            body="Urgent",
            importance_score=0.9,
            is_read=False
        )

        read_unimportant = EmailMessage(
            id="2",
            subject="Newsletter",
            sender="news@example.com",
            received_at=datetime.now(),
            body="Updates",
            importance_score=0.3,
            is_read=True
        )

        # Test filters
        assert should_show_email(unread_important, filter_unread=True) is True
        assert should_show_email(read_unimportant, filter_unread=True) is False
        assert should_show_email(unread_important, filter_important=True) is True
        assert should_show_email(read_unimportant, filter_important=True) is False
        print("✅ Email filtering works correctly")

    @pytest.mark.asyncio
    async def test_email_search_integration(self):
        """Test email search works end-to-end."""
        from src.ui.components.email_components import should_show_email

        email = EmailMessage(
            id="1",
            subject="Meeting about project alpha",
            sender="alice@example.com",
            received_at=datetime.now(),
            body="Let's discuss alpha project",
            is_read=False
        )

        # Test search
        assert should_show_email(email, search_query="alpha") is True
        assert should_show_email(email, search_query="alice") is True
        assert should_show_email(email, search_query="beta") is False
        print("✅ Email search works correctly")


class TestCalendarTabIntegration:
    """Integration tests for Calendar tab functionality."""

    @pytest.mark.asyncio
    async def test_calendar_tab_can_import(self):
        """Test that calendar tab components can be imported."""
        try:
            from src.ui.components.calendar_components import render_calendar_tab
            assert render_calendar_tab is not None
            print("✅ Calendar tab imports successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import calendar tab: {e}")

    @pytest.mark.asyncio
    async def test_calendar_service_integration(self):
        """Test calendar service can fetch events."""
        from src.services.calendar_service import get_calendar_service

        service = get_calendar_service()
        assert service is not None
        print("✅ Calendar service initialized")

        # Test that fetch_events method exists and is callable
        assert hasattr(service, 'fetch_events')
        assert callable(service.fetch_events)
        print("✅ Calendar service has fetch_events method")

    @pytest.mark.asyncio
    async def test_calendar_formatter_integration(self):
        """Test calendar formatter integrates with components."""
        from src.ui.formatters.calendar_formatter import get_calendar_formatter
        from src.ui.components.calendar_components import prepare_event_display_data

        # Create sample event
        event = CalendarEvent(
            id="test-1",
            summary="Test Meeting",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=2),
            importance_score=0.8
        )

        # Test formatter works
        formatter = get_calendar_formatter()
        assert formatter is not None

        # Test component helper uses formatter
        display_data = prepare_event_display_data(event)
        assert display_data is not None
        assert "summary" in display_data
        assert "time_range" in display_data
        assert "duration" in display_data
        print("✅ Calendar formatter integrates with components")

    @pytest.mark.asyncio
    async def test_calendar_filtering_integration(self):
        """Test calendar filtering works end-to-end."""
        from src.ui.components.calendar_components import should_show_event

        # Create test events
        important_prep = CalendarEvent(
            id="1",
            summary="Board meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            importance_score=0.9,
            requires_preparation=True
        )

        unimportant_no_prep = CalendarEvent(
            id="2",
            summary="Coffee chat",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            importance_score=0.3,
            requires_preparation=False
        )

        # Test filters
        assert should_show_event(important_prep, filter_important=True) is True
        assert should_show_event(unimportant_no_prep, filter_important=True) is False
        assert should_show_event(important_prep, filter_preparation=True) is True
        assert should_show_event(unimportant_no_prep, filter_preparation=True) is False
        print("✅ Calendar filtering works correctly")

    @pytest.mark.asyncio
    async def test_calendar_conflict_detection(self):
        """Test calendar conflict detection works."""
        from src.ui.components.calendar_components import get_event_conflicts

        event1 = CalendarEvent(
            id="1",
            summary="Meeting A",
            start_time=datetime(2025, 10, 26, 10, 0),
            end_time=datetime(2025, 10, 26, 11, 0)
        )

        event2 = CalendarEvent(
            id="2",
            summary="Meeting B (overlaps)",
            start_time=datetime(2025, 10, 26, 10, 30),
            end_time=datetime(2025, 10, 26, 11, 30)
        )

        event3 = CalendarEvent(
            id="3",
            summary="Meeting C (no overlap)",
            start_time=datetime(2025, 10, 26, 12, 0),
            end_time=datetime(2025, 10, 26, 13, 0)
        )

        all_events = [event1, event2, event3]
        conflicts = get_event_conflicts(event1, all_events)

        assert len(conflicts) == 1
        assert conflicts[0].id == "2"
        print("✅ Calendar conflict detection works correctly")


class TestEmailCalendarCrossIntegration:
    """Test integration between email and calendar features."""

    @pytest.mark.asyncio
    async def test_both_services_can_coexist(self):
        """Test that email and calendar services can be used together."""
        from src.services.email_service import get_email_service
        from src.services.calendar_service import get_calendar_service

        email_service = get_email_service()
        calendar_service = get_calendar_service()

        assert email_service is not None
        assert calendar_service is not None
        assert email_service is not calendar_service
        print("✅ Both services coexist independently")

    @pytest.mark.asyncio
    async def test_both_formatters_can_coexist(self):
        """Test that email and calendar formatters can be used together."""
        from src.ui.formatters.email_formatter import get_email_formatter
        from src.ui.formatters.calendar_formatter import get_calendar_formatter

        email_formatter = get_email_formatter()
        calendar_formatter = get_calendar_formatter()

        assert email_formatter is not None
        assert calendar_formatter is not None
        assert email_formatter is not calendar_formatter
        print("✅ Both formatters coexist independently")

    @pytest.mark.asyncio
    async def test_models_are_compatible(self):
        """Test that EmailMessage and CalendarEvent models work together."""
        email = EmailMessage(
            id="1",
            subject="Meeting reminder",
            sender="calendar@example.com",
            received_at=datetime.now(),
            body="Don't forget the meeting",
            is_read=False
        )

        event = CalendarEvent(
            id="1",
            summary="Team meeting",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=2)
        )

        # Both should have importance scores (from mixin)
        assert hasattr(email, 'importance_score')
        assert hasattr(event, 'importance_score')
        print("✅ Models are compatible and share common traits")


@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test a complete workflow using both email and calendar."""
    print("\n=== Testing Full Email & Calendar Integration ===\n")

    # Test email workflow
    print("1. Testing Email Workflow...")
    from src.ui.components.email_components import (
        prepare_email_display_data,
        should_show_email,
        group_emails_for_display
    )

    emails = [
        EmailMessage(
            id=f"email-{i}",
            subject=f"Email {i}",
            sender=f"user{i}@example.com",
            received_at=datetime.now() - timedelta(hours=i),
            body=f"Body {i}",
            importance_score=0.5 + (i * 0.1),
            is_read=i % 2 == 0
        )
        for i in range(5)
    ]

    # Filter and group
    unread_emails = [e for e in emails if should_show_email(e, filter_unread=True)]
    grouped_emails = group_emails_for_display(emails)

    assert len(unread_emails) > 0
    assert len(grouped_emails) > 0
    print(f"   ✅ Processed {len(emails)} emails, {len(unread_emails)} unread")

    # Test calendar workflow
    print("\n2. Testing Calendar Workflow...")
    from src.ui.components.calendar_components import (
        prepare_event_display_data,
        should_show_event,
        group_events_for_display,
        get_event_conflicts
    )

    now = datetime.now()
    events = [
        CalendarEvent(
            id=f"event-{i}",
            summary=f"Event {i}",
            start_time=now + timedelta(hours=i),
            end_time=now + timedelta(hours=i+1),
            importance_score=0.5 + (i * 0.1),
            requires_preparation=i % 2 == 0
        )
        for i in range(5)
    ]

    # Filter and group
    prep_events = [e for e in events if should_show_event(e, filter_preparation=True)]
    grouped_events = group_events_for_display(events)

    assert len(prep_events) > 0
    assert len(grouped_events) > 0
    print(f"   ✅ Processed {len(events)} events, {len(prep_events)} need prep")

    # Test data preparation
    print("\n3. Testing Data Preparation...")
    email_display = prepare_email_display_data(emails[0])
    event_display = prepare_event_display_data(events[0])

    assert "subject" in email_display
    assert "summary" in event_display
    print("   ✅ Display data prepared successfully")

    print("\n✅ Full integration workflow completed successfully!\n")
    return True


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("EMAIL & CALENDAR INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    # Email tab tests
    print("\n--- Email Tab Tests ---")
    email_tests = TestEmailTabIntegration()
    try:
        await email_tests.test_email_tab_can_import()
        await email_tests.test_email_service_integration()
        await email_tests.test_email_formatter_integration()
        await email_tests.test_email_filtering_integration()
        await email_tests.test_email_search_integration()
    except Exception as e:
        print(f"❌ Email tab test failed: {e}")
        all_passed = False

    # Calendar tab tests
    print("\n--- Calendar Tab Tests ---")
    calendar_tests = TestCalendarTabIntegration()
    try:
        await calendar_tests.test_calendar_tab_can_import()
        await calendar_tests.test_calendar_service_integration()
        await calendar_tests.test_calendar_formatter_integration()
        await calendar_tests.test_calendar_filtering_integration()
        await calendar_tests.test_calendar_conflict_detection()
    except Exception as e:
        print(f"❌ Calendar tab test failed: {e}")
        all_passed = False

    # Cross-integration tests
    print("\n--- Cross-Integration Tests ---")
    cross_tests = TestEmailCalendarCrossIntegration()
    try:
        await cross_tests.test_both_services_can_coexist()
        await cross_tests.test_both_formatters_can_coexist()
        await cross_tests.test_models_are_compatible()
    except Exception as e:
        print(f"❌ Cross-integration test failed: {e}")
        all_passed = False

    # Full workflow test
    try:
        await test_full_integration_workflow()
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Load Mock Email and Calendar Data for UI Testing.

This script demonstrates how to load mock data into the Daily Minutes application
for testing the email and calendar features without real email/calendar services.

Usage:
    # From Python/Streamlit session:
    from tests.load_mock_data import load_mock_emails, load_mock_calendar_events

    emails = load_mock_emails()
    events = load_mock_calendar_events()
"""

from datetime import datetime, timedelta
from typing import List
import random

from src.core.models import EmailMessage, CalendarEvent
from tests.mock_calendar_data import generate_mock_calendar_events


def load_mock_emails(count: int = 20) -> List[EmailMessage]:
    """Generate mock email messages for testing.

    Args:
        count: Number of emails to generate

    Returns:
        List of EmailMessage objects with realistic data
    """
    # Email templates (matching the enhanced templates in mock_email_server.py)
    templates = [
        {
            "subject": "URGENT: Production Database Issue - Action Required",
            "sender": "it@company.com",
            "body": """Hi Team,

We've detected a critical issue with the production database.

ACTION ITEMS:
1. TODO: Review error logs in /var/log/mysql/error.log
2. TODO: Check replication status
3. TODO: Prepare rollback plan

This needs immediate attention. Please respond ASAP.

Thanks,
IT Team""",
            "importance_score": 0.95,
            "has_action_items": True,
            "action_items": [
                "Review error logs in /var/log/mysql/error.log",
                "Check replication status",
                "Prepare rollback plan"
            ],
            "is_read": False
        },
        {
            "subject": "Q4 Planning Meeting - Please Prepare",
            "sender": "pm@company.com",
            "body": """Team,

Our Q4 planning session is scheduled for next week.

PREPARATION NEEDED:
- TODO: Review Q3 metrics dashboard
- TODO: Prepare your team's roadmap proposals
- TODO: Submit budget estimates by Friday

Looking forward to productive discussions!

Best,
Project Manager""",
            "importance_score": 0.85,
            "has_action_items": True,
            "action_items": [
                "Review Q3 metrics dashboard",
                "Prepare your team's roadmap proposals",
                "Submit budget estimates by Friday"
            ],
            "is_read": False
        },
        {
            "subject": "Team Lunch - Casual Friday",
            "sender": "colleague@company.com",
            "body": """Hey everyone!

Let's do team lunch this Friday at 12pm.

No work talk, just good food and conversation!

Reply if you're joining.

Cheers!""",
            "importance_score": 0.3,
            "has_action_items": False,
            "action_items": [],
            "is_read": True
        },
        {
            "subject": "Client Escalation - Immediate Response Needed",
            "sender": "success@company.com",
            "body": """URGENT,

Major client (Acme Corp) is escalating service issues.

IMMEDIATE ACTIONS:
1. TODO: Call client contact: John Smith (555-1234)
2. TODO: Prepare incident report
3. TODO: Schedule resolution meeting today

They're threatening to switch vendors. This is top priority.

- Customer Success""",
            "importance_score": 0.97,
            "has_action_items": True,
            "action_items": [
                "Call client contact: John Smith (555-1234)",
                "Prepare incident report",
                "Schedule resolution meeting today"
            ],
            "is_read": False
        },
        {
            "subject": "Weekly Tech Digest - AI & Cloud Updates",
            "sender": "newsletter@techdigest.com",
            "body": """Your weekly tech updates:

- New AI models released
- Cloud cost optimization tips
- Security best practices

Read more at techdigest.com

Unsubscribe | Update preferences""",
            "importance_score": 0.2,
            "has_action_items": False,
            "action_items": [],
            "is_read": True
        },
        {
            "subject": "Code Review: Feature/user-authentication PR #245",
            "sender": "dev@company.com",
            "body": """Hi,

Please review my PR for the new authentication system.

Changes:
- Implemented OAuth2 flow
- Added JWT token validation
- Updated user session management

TODO: Review and approve PR #245

Tests are passing. Ready for staging deployment.

Thanks!
Developer""",
            "importance_score": 0.75,
            "has_action_items": True,
            "action_items": ["Review and approve PR #245"],
            "is_read": False
        },
        {
            "subject": "Performance Review - Self Assessment Due",
            "sender": "hr@company.com",
            "body": """Dear Team Member,

It's time for quarterly performance reviews.

REQUIRED ACTIONS:
- TODO: Complete self-assessment form by Oct 30
- TODO: List 3 key achievements this quarter
- TODO: Set goals for next quarter

Link: performance.company.com/review

HR Department""",
            "importance_score": 0.8,
            "has_action_items": True,
            "action_items": [
                "Complete self-assessment form by Oct 30",
                "List 3 key achievements this quarter",
                "Set goals for next quarter"
            ],
            "is_read": False
        },
        {
            "subject": "You've won a FREE vacation! Click here!",
            "sender": "spam@suspicious.ru",
            "body": """Congratulations! You're our lucky winner!

Claim your FREE all-expenses-paid vacation now!

Click here: totally-legit-site.ru

(This is obviously spam for testing filters)""",
            "importance_score": 0.1,
            "has_action_items": False,
            "action_items": [],
            "is_read": True
        },
        {
            "subject": "Security Training - Mandatory Completion",
            "sender": "security@company.com",
            "body": """Important Security Notice,

Annual security training is now available.

REQUIREMENTS:
- TODO: Complete cybersecurity module (2 hours)
- TODO: Pass the final quiz (80% required)
- TODO: Submit completion certificate

Deadline: November 15th

This is mandatory for all employees.

Security Team""",
            "importance_score": 0.78,
            "has_action_items": True,
            "action_items": [
                "Complete cybersecurity module (2 hours)",
                "Pass the final quiz (80% required)",
                "Submit completion certificate"
            ],
            "is_read": False
        },
        {
            "subject": "Office Supplies Order - Your Input",
            "sender": "admin@company.com",
            "body": """Hi team,

We're placing the monthly office supplies order.

If you need anything (pens, notepads, etc), reply by Wednesday.

No rush - just FYI.

Admin""",
            "importance_score": 0.25,
            "has_action_items": False,
            "action_items": [],
            "is_read": True
        }
    ]

    emails = []
    now = datetime.now()

    for i in range(count):
        template = templates[i % len(templates)]

        # Vary received time (some from hours ago, some from days ago)
        hours_ago = random.randint(1, 72)
        received_at = now - timedelta(hours=hours_ago)

        email = EmailMessage(
            id=f"mock_email_{i}",
            subject=template["subject"],
            sender=template["sender"],
            body=template["body"],
            received_at=received_at,
            importance_score=template["importance_score"],
            has_action_items=template["has_action_items"],
            action_items=template["action_items"],
            is_read=template["is_read"]
        )

        emails.append(email)

    # Sort by received_at (newest first)
    emails.sort(key=lambda e: e.received_at, reverse=True)

    return emails


def load_mock_calendar_events(count: int = 12, days_ahead: int = 7) -> List[CalendarEvent]:
    """Generate mock calendar events for testing.

    Args:
        count: Number of events to generate
        days_ahead: Spread events over this many days

    Returns:
        List of CalendarEvent objects with realistic data
    """
    return generate_mock_calendar_events(count=count, days_ahead=days_ahead)


if __name__ == "__main__":
    """Test the mock data loaders."""
    print("📧 Loading Mock Email Data\n")
    emails = load_mock_emails(count=20)
    print(f"✅ Loaded {len(emails)} mock emails\n")

    # Show statistics
    unread = sum(1 for e in emails if not e.is_read)
    important = sum(1 for e in emails if e.importance_score >= 0.7)
    with_actions = sum(1 for e in emails if e.has_action_items)

    print(f"📊 Email Statistics:")
    print(f"   Total: {len(emails)}")
    print(f"   Unread: {unread}")
    print(f"   Important: {important}")
    print(f"   With Action Items: {with_actions}\n")

    # Show sample emails
    print("📬 Sample Emails (Top 3):\n")
    for i, email in enumerate(emails[:3], 1):
        print(f"{i}. {email.subject}")
        print(f"   From: {email.sender}")
        print(f"   Importance: {email.importance_score:.2f}")
        print(f"   Unread: {not email.is_read}")
        if email.has_action_items:
            print(f"   Action Items: {len(email.action_items)}")
        print()

    print("=" * 60)
    print()

    print("📅 Loading Mock Calendar Events\n")
    events = load_mock_calendar_events(count=12, days_ahead=7)
    print(f"✅ Loaded {len(events)} mock calendar events\n")

    # Show statistics
    now = datetime.now()
    upcoming = [e for e in events if e.start_time >= now]
    important_events = sum(1 for e in upcoming if e.importance_score >= 0.7)
    prep_required = sum(1 for e in upcoming if e.requires_preparation)
    focus_time = sum(1 for e in upcoming if e.is_focus_time)

    print(f"📊 Calendar Statistics:")
    print(f"   Total: {len(events)}")
    print(f"   Upcoming: {len(upcoming)}")
    print(f"   Important: {important_events}")
    print(f"   Prep Required: {prep_required}")
    print(f"   Focus Time: {focus_time}\n")

    # Show sample events
    print("📆 Sample Events (Next 3):\n")
    for i, event in enumerate(upcoming[:3], 1):
        time_until = event.start_time - now
        hours_until = time_until.total_seconds() / 3600

        print(f"{i}. {event.summary}")
        print(f"   Time: {event.start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   In: {hours_until:.1f} hours")
        print(f"   Importance: {event.importance_score:.2f}")
        print(f"   Prep Required: {event.requires_preparation}")
        print()

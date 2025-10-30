#!/usr/bin/env python3
"""Mock Calendar Data Generator - Realistic calendar events for testing.

Generates test calendar events with various importance levels, preparation requirements,
and realistic scenarios for testing the Daily Minutes calendar integration.

Usage:
    from tests.mock_calendar_data import generate_mock_calendar_events

    events = generate_mock_calendar_events(count=10)
    # Returns list of CalendarEvent objects
"""

from datetime import datetime, timedelta
from typing import List
import random

from src.core.models import CalendarEvent


# Calendar event templates with realistic scenarios
CALENDAR_EVENT_TEMPLATES = [
    {
        "category": "critical_meeting",
        "summary": "Board Meeting - Q4 Results Review",
        "description": """Quarterly board meeting to review Q4 performance and strategy for next year.

AGENDA:
- Q4 financial results
- Product roadmap review
- Strategic initiatives for 2026
- Budget approval

PREPARATION REQUIRED:
- Review financial reports
- Prepare product demo
- Draft strategic proposals""",
        "location": "Executive Conference Room / Zoom",
        "duration_minutes": 120,
        "attendees": ["ceo@company.com", "cfo@company.com", "board@company.com", "vp-eng@company.com"],
        "importance": "high",
        "requires_preparation": True,
        "preparation_notes": [
            "Review Q4 financial dashboard and prepare key metrics",
            "Update product roadmap presentation with latest milestones",
            "Prepare answers for potential board questions",
            "Test demo environment 1 hour before meeting"
        ],
        "is_focus_time": False
    },
    {
        "category": "focus_time",
        "summary": "Deep Work - Feature Development",
        "description": """Dedicated focus time for implementing the new authentication system.

NO MEETINGS - DO NOT DISTURB

Goals:
- Complete OAuth2 implementation
- Write comprehensive tests
- Update documentation""",
        "location": "My Desk (No interruptions)",
        "duration_minutes": 180,
        "attendees": [],
        "importance": "high",
        "requires_preparation": False,
        "preparation_notes": [],
        "is_focus_time": True
    },
    {
        "category": "one_on_one",
        "summary": "1:1 with Sarah (Engineering Manager)",
        "description": """Regular 1:1 check-in.

Topics:
- Career development discussion
- Current project status
- Blockers and support needed
- Feedback and growth areas""",
        "location": "Coffee shop / Video call",
        "duration_minutes": 30,
        "attendees": ["sarah.chen@company.com"],
        "importance": "medium",
        "requires_preparation": True,
        "preparation_notes": [
            "Update project status doc with current progress",
            "List any blockers or support needed",
            "Think about career goals for next quarter"
        ],
        "is_focus_time": False
    },
    {
        "category": "team_meeting",
        "summary": "Sprint Planning - Engineering Team",
        "description": """Bi-weekly sprint planning session.

- Review completed sprint
- Plan next sprint stories
- Estimate effort
- Assign tasks
- Discuss dependencies""",
        "location": "Conference Room A",
        "duration_minutes": 90,
        "attendees": ["engineering-team@company.com", "pm@company.com", "scrum@company.com"],
        "importance": "high",
        "requires_preparation": True,
        "preparation_notes": [
            "Review backlog items before meeting",
            "Prepare technical estimates for complex stories",
            "Identify any dependencies or blockers"
        ],
        "is_focus_time": False
    },
    {
        "category": "interview",
        "summary": "Technical Interview - Senior Backend Engineer",
        "description": """Technical interview for senior backend engineer position.

INTERVIEW PLAN:
- System design discussion (30 min)
- Coding challenge (45 min)
- Q&A and team culture (15 min)

Candidate: Alex Johnson
Position: Senior Backend Engineer
Panel: You + 2 other engineers""",
        "location": "Interview Room 2 / Zoom",
        "duration_minutes": 90,
        "attendees": ["alex.johnson.candidate@gmail.com", "interview-panel@company.com"],
        "importance": "high",
        "requires_preparation": True,
        "preparation_notes": [
            "Review candidate resume and GitHub profile",
            "Prepare system design scenario questions",
            "Set up coding challenge environment",
            "Review interview rubric and scoring criteria"
        ],
        "is_focus_time": False
    },
    {
        "category": "standup",
        "summary": "Daily Standup - Engineering",
        "description": """Quick daily sync.

- What did you do yesterday?
- What are you doing today?
- Any blockers?

Keep it short and focused!""",
        "location": "Zoom - Daily Standup Room",
        "duration_minutes": 15,
        "attendees": ["engineering-team@company.com"],
        "importance": "medium",
        "requires_preparation": False,
        "preparation_notes": [],
        "is_focus_time": False
    },
    {
        "category": "client_meeting",
        "summary": "Client Demo - Acme Corp Integration",
        "description": """Demonstrate new integration features to Acme Corp stakeholders.

DEMO AGENDA:
1. Overview of integration architecture (10 min)
2. Live demo of key features (30 min)
3. Performance and security overview (10 min)
4. Q&A and next steps (10 min)

This is a high-value client - make sure everything works!""",
        "location": "Client Site / Zoom Webinar",
        "duration_minutes": 60,
        "attendees": ["acme-team@acmecorp.com", "sales@company.com", "success@company.com", "you@company.com"],
        "importance": "high",
        "requires_preparation": True,
        "preparation_notes": [
            "Test demo environment thoroughly - have backup ready",
            "Prepare presentation slides with key screenshots",
            "Review client's specific use cases and pain points",
            "Coordinate with sales on pricing discussion",
            "Have answers ready for common security questions"
        ],
        "is_focus_time": False
    },
    {
        "category": "personal",
        "summary": "Dentist Appointment",
        "description": """Regular dental cleaning and checkup.

Remember to bring insurance card.""",
        "location": "Downtown Dental - 123 Main St",
        "duration_minutes": 60,
        "attendees": [],
        "importance": "low",
        "requires_preparation": False,
        "preparation_notes": [],
        "is_focus_time": False
    },
    {
        "category": "workshop",
        "summary": "Security Training Workshop - OWASP Top 10",
        "description": """Mandatory security training covering OWASP Top 10 vulnerabilities.

TOPICS:
- Injection attacks
- Broken authentication
- XSS and CSRF
- Security misconfiguration
- Best practices for secure coding

Required for all engineering staff.""",
        "location": "Training Room / Zoom",
        "duration_minutes": 120,
        "attendees": ["engineering-all@company.com", "security@company.com"],
        "importance": "medium",
        "requires_preparation": False,
        "preparation_notes": [],
        "is_focus_time": False
    },
    {
        "category": "team_social",
        "summary": "Team Lunch - Welcome New Team Members",
        "description": """Casual team lunch to welcome new engineers who joined this month.

No work talk - just getting to know each other!

Company is covering the bill.""",
        "location": "Italian Restaurant - Downtown",
        "duration_minutes": 90,
        "attendees": ["engineering-team@company.com"],
        "importance": "low",
        "requires_preparation": False,
        "preparation_notes": [],
        "is_focus_time": False
    },
    {
        "category": "code_review",
        "summary": "Architecture Review - Microservices Migration",
        "description": """Technical deep-dive on proposed microservices architecture.

REVIEW ITEMS:
- Service boundaries and responsibilities
- Data consistency patterns
- API contracts and versioning
- Deployment and rollback strategy
- Monitoring and observability

This is a critical architectural decision - come prepared!""",
        "location": "Engineering War Room",
        "duration_minutes": 120,
        "attendees": ["eng-leads@company.com", "architects@company.com", "devops@company.com"],
        "importance": "high",
        "requires_preparation": True,
        "preparation_notes": [
            "Review architecture proposal document thoroughly",
            "Identify potential issues with service boundaries",
            "Research industry best practices for similar migrations",
            "Prepare questions about data consistency approach"
        ],
        "is_focus_time": False
    },
    {
        "category": "conflict_test",
        "summary": "Overlapping Meeting - Sales Demo",
        "description": """This event is designed to create a scheduling conflict for testing.

Sales demo for potential customer.""",
        "location": "Zoom",
        "duration_minutes": 60,
        "attendees": ["sales-team@company.com", "you@company.com"],
        "importance": "medium",
        "requires_preparation": True,
        "preparation_notes": [
            "Prepare demo environment",
            "Review customer use case"
        ],
        "is_focus_time": False
    }
]


def generate_mock_calendar_events(
    count: int = 12,
    start_date: datetime = None,
    days_ahead: int = 7
) -> List[CalendarEvent]:
    """Generate realistic mock calendar events for testing.

    Args:
        count: Number of events to generate (default: 12)
        start_date: Starting date for events (default: now)
        days_ahead: Spread events over this many days (default: 7)

    Returns:
        List of CalendarEvent objects with realistic data
    """
    if start_date is None:
        start_date = datetime.now()

    events = []

    # Create a conflict event pair for testing
    conflict_event_created = False

    for i in range(count):
        # Use templates, cycling through them
        template = CALENDAR_EVENT_TEMPLATES[i % len(CALENDAR_EVENT_TEMPLATES)]

        # Extract template data
        summary = template["summary"]
        description = template["description"]
        location = template["location"]
        duration_minutes = template["duration_minutes"]
        attendees = template.get("attendees", [])
        importance = template["importance"]
        requires_preparation = template["requires_preparation"]
        preparation_notes = template.get("preparation_notes", [])
        is_focus_time = template.get("is_focus_time", False)

        # Calculate importance score based on level
        if importance == "high":
            importance_score = random.uniform(0.8, 0.95)
        elif importance == "medium":
            importance_score = random.uniform(0.5, 0.7)
        else:
            importance_score = random.uniform(0.1, 0.4)

        # Distribute events across the time range
        days_offset = (i * days_ahead) // count
        hours_offset = random.randint(8, 17)  # Business hours 8am-5pm
        minutes_offset = random.choice([0, 15, 30, 45])  # Round to 15-min intervals

        start_time = start_date + timedelta(
            days=days_offset,
            hours=hours_offset - start_date.hour,
            minutes=minutes_offset - start_date.minute,
            seconds=-start_date.second,
            microseconds=-start_date.microsecond
        )

        end_time = start_time + timedelta(minutes=duration_minutes)

        # Special case: create an intentional conflict for testing
        # Make the 12th event (conflict_test) overlap with a previous event
        if template["category"] == "conflict_test" and i > 0 and not conflict_event_created:
            # Overlap with the first event
            start_time = events[0].start_time + timedelta(minutes=15)
            end_time = start_time + timedelta(minutes=duration_minutes)
            conflict_event_created = True

        # Create event
        event = CalendarEvent(
            id=f"mock_event_{i}",
            summary=summary,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            attendees=attendees,
            importance_score=importance_score,
            requires_preparation=requires_preparation,
            preparation_notes=preparation_notes,
            is_focus_time=is_focus_time
        )

        events.append(event)

    # Sort events by start time
    events.sort(key=lambda e: e.start_time)

    return events


def get_mock_events_by_category() -> dict:
    """Get mock events organized by category for testing.

    Returns:
        Dictionary mapping category names to event templates
    """
    categories = {}
    for template in CALENDAR_EVENT_TEMPLATES:
        category = template["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(template)
    return categories


def get_mock_event_statistics() -> dict:
    """Get statistics about the mock event templates.

    Returns:
        Dictionary with counts and percentages
    """
    total = len(CALENDAR_EVENT_TEMPLATES)

    high_importance = sum(1 for t in CALENDAR_EVENT_TEMPLATES if t["importance"] == "high")
    requires_prep = sum(1 for t in CALENDAR_EVENT_TEMPLATES if t["requires_preparation"])
    focus_time = sum(1 for t in CALENDAR_EVENT_TEMPLATES if t.get("is_focus_time", False))

    return {
        "total_templates": total,
        "high_importance": high_importance,
        "requires_preparation": requires_prep,
        "focus_time": focus_time,
        "categories": list(set(t["category"] for t in CALENDAR_EVENT_TEMPLATES))
    }


if __name__ == "__main__":
    """Test the mock calendar data generator."""
    print("📅 Mock Calendar Data Generator\n")

    # Generate events
    events = generate_mock_calendar_events(count=12, days_ahead=7)

    print(f"✅ Generated {len(events)} mock calendar events\n")

    # Show statistics
    stats = get_mock_event_statistics()
    print("📊 Template Statistics:")
    print(f"   Total templates: {stats['total_templates']}")
    print(f"   High importance: {stats['high_importance']}")
    print(f"   Requires prep: {stats['requires_preparation']}")
    print(f"   Focus time: {stats['focus_time']}")
    print(f"   Categories: {', '.join(stats['categories'])}\n")

    # Show sample events
    print("📋 Sample Events:\n")
    for i, event in enumerate(events[:5], 1):
        print(f"{i}. {event.summary}")
        print(f"   Time: {event.start_time.strftime('%Y-%m-%d %H:%M')} - {event.end_time.strftime('%H:%M')}")
        print(f"   Duration: {(event.end_time - event.start_time).total_seconds() / 60:.0f} minutes")
        print(f"   Importance: {event.importance_score:.2f}")
        print(f"   Prep required: {event.requires_preparation}")
        if event.attendees:
            print(f"   Attendees: {', '.join(event.attendees[:3])}{'...' if len(event.attendees) > 3 else ''}")
        print()

    print(f"... and {len(events) - 5} more events")

"""Unit tests for RAG Memory data models.

Following TDD principles - test all models before marking complete.
"""

import pytest
from datetime import datetime, timedelta
from src.memory.models import (
    Memory,
    MemoryType,
    ActionItemStatus,
    DailyBriefMemory,
    ActionItemMemory,
    MeetingContextMemory,
    UserPreferenceMemory,
    RetrievalResult,
)


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_type_values(self):
        """Test that all memory types have correct values."""
        assert MemoryType.DAILY_BRIEF.value == "daily_brief"
        assert MemoryType.ACTION_ITEM.value == "action_item"
        assert MemoryType.MEETING_CONTEXT.value == "meeting_context"
        assert MemoryType.USER_PREFERENCE.value == "user_preference"
        assert MemoryType.EMAIL_SUMMARY.value == "email_summary"
        assert MemoryType.CONVERSATION.value == "conversation"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string value."""
        assert MemoryType("daily_brief") == MemoryType.DAILY_BRIEF
        assert MemoryType("action_item") == MemoryType.ACTION_ITEM


class TestActionItemStatus:
    """Test ActionItemStatus enum."""

    def test_action_item_status_values(self):
        """Test that all statuses have correct values."""
        assert ActionItemStatus.PENDING.value == "pending"
        assert ActionItemStatus.IN_PROGRESS.value == "in_progress"
        assert ActionItemStatus.COMPLETED.value == "completed"
        assert ActionItemStatus.CANCELLED.value == "cancelled"


class TestBaseMemory:
    """Test base Memory class."""

    def test_memory_creation(self):
        """Test creating a basic memory."""
        memory = Memory(
            id="test-123",
            type=MemoryType.CONVERSATION,
            content="Test content",
            metadata={"key": "value"},
        )

        assert memory.id == "test-123"
        assert memory.type == MemoryType.CONVERSATION
        assert memory.content == "Test content"
        assert memory.metadata == {"key": "value"}
        assert memory.embedding is None
        assert isinstance(memory.timestamp, datetime)

    def test_memory_to_dict(self):
        """Test converting memory to dictionary."""
        timestamp = datetime(2025, 1, 15, 10, 30)
        memory = Memory(
            id="test-123",
            type=MemoryType.EMAIL_SUMMARY,
            content="Email summary",
            metadata={"sender": "test@example.com"},
            timestamp=timestamp,
        )

        result = memory.to_dict()

        assert result["id"] == "test-123"
        assert result["type"] == "email_summary"
        assert result["content"] == "Email summary"
        assert result["metadata"] == {"sender": "test@example.com"}
        assert result["timestamp"] == "2025-01-15T10:30:00"

    def test_memory_from_dict(self):
        """Test creating memory from dictionary."""
        data = {
            "id": "test-456",
            "type": "conversation",
            "content": "Test conversation",
            "metadata": {"user": "Alice"},
            "timestamp": "2025-01-15T14:00:00",
        }

        memory = Memory.from_dict(data)

        assert memory.id == "test-456"
        assert memory.type == MemoryType.CONVERSATION
        assert memory.content == "Test conversation"
        assert memory.metadata == {"user": "Alice"}
        assert memory.timestamp == datetime(2025, 1, 15, 14, 0, 0)

    def test_memory_with_embedding(self):
        """Test memory with vector embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        memory = Memory(
            id="test-789",
            type=MemoryType.DAILY_BRIEF,
            content="Brief content",
            metadata={},
            embedding=embedding,
        )

        assert memory.embedding == embedding


class TestDailyBriefMemory:
    """Test DailyBriefMemory class."""

    def test_daily_brief_creation(self):
        """Test creating a daily brief memory."""
        brief = DailyBriefMemory(
            id="brief-001",
            summary="Daily summary for today",
            key_points=["Point 1", "Point 2", "Point 3"],
            action_items=["Action 1", "Action 2"],
            emails_count=15,
            calendar_events_count=5,
            news_items_count=20,
        )

        assert brief.id == "brief-001"
        assert brief.type == MemoryType.DAILY_BRIEF
        assert "Daily summary for today" in brief.content
        assert "Point 1" in brief.content
        assert "Action 1" in brief.content
        assert brief.metadata["summary"] == "Daily summary for today"
        assert brief.metadata["emails_count"] == 15

    def test_daily_brief_content_includes_statistics(self):
        """Test that daily brief content includes all statistics."""
        brief = DailyBriefMemory(
            id="brief-002",
            summary="Test brief",
            key_points=["Key point"],
            action_items=["Task"],
            emails_count=10,
            calendar_events_count=3,
            news_items_count=25,
        )

        assert "10 emails" in brief.content
        assert "3 events" in brief.content
        assert "25 news items" in brief.content

    def test_daily_brief_with_custom_timestamp(self):
        """Test creating daily brief with custom timestamp."""
        timestamp = datetime(2025, 1, 10, 8, 0)
        brief = DailyBriefMemory(
            id="brief-003",
            summary="Morning brief",
            key_points=["Morning update"],
            action_items=[],
            emails_count=5,
            calendar_events_count=2,
            news_items_count=10,
            timestamp=timestamp,
        )

        assert brief.timestamp == timestamp
        assert "2025-01-10" in brief.content
        assert brief.metadata["date"] == "2025-01-10"

    def test_daily_brief_metadata_structure(self):
        """Test that daily brief metadata has all required fields."""
        brief = DailyBriefMemory(
            id="brief-004",
            summary="Test",
            key_points=["A", "B"],
            action_items=["X", "Y"],
            emails_count=1,
            calendar_events_count=2,
            news_items_count=3,
        )

        metadata = brief.metadata
        assert "summary" in metadata
        assert "key_points" in metadata
        assert "action_items" in metadata
        assert "emails_count" in metadata
        assert "calendar_events_count" in metadata
        assert "news_items_count" in metadata
        assert "date" in metadata


class TestActionItemMemory:
    """Test ActionItemMemory class."""

    def test_action_item_creation(self):
        """Test creating an action item memory."""
        item = ActionItemMemory(
            id="action-001",
            description="Complete project report",
            status=ActionItemStatus.PENDING,
            source="email",
        )

        assert item.id == "action-001"
        assert item.type == MemoryType.ACTION_ITEM
        assert "Complete project report" in item.content
        assert "pending" in item.content
        assert item.metadata["description"] == "Complete project report"
        assert item.metadata["status"] == "pending"
        assert item.metadata["source"] == "email"

    def test_action_item_with_due_date(self):
        """Test action item with due date."""
        due_date = datetime(2025, 1, 20, 17, 0)
        item = ActionItemMemory(
            id="action-002",
            description="Submit proposal",
            status=ActionItemStatus.IN_PROGRESS,
            source="calendar",
            due_date=due_date,
        )

        assert "Due: 2025-01-20" in item.content
        assert item.metadata["due_date"] == due_date.isoformat()

    def test_action_item_with_completion(self):
        """Test completed action item."""
        completed_date = datetime(2025, 1, 15, 14, 30)
        item = ActionItemMemory(
            id="action-003",
            description="Review document",
            status=ActionItemStatus.COMPLETED,
            source="meeting",
            completed_date=completed_date,
        )

        assert "Completed: 2025-01-15" in item.content
        assert item.metadata["status"] == "completed"
        assert item.metadata["completed_date"] == completed_date.isoformat()

    def test_action_item_status_transitions(self):
        """Test action item with different statuses."""
        for status in [
            ActionItemStatus.PENDING,
            ActionItemStatus.IN_PROGRESS,
            ActionItemStatus.COMPLETED,
            ActionItemStatus.CANCELLED,
        ]:
            item = ActionItemMemory(
                id=f"action-{status.value}",
                description="Test task",
                status=status,
                source="test",
            )

            assert item.metadata["status"] == status.value
            assert status.value in item.content.lower()


class TestMeetingContextMemory:
    """Test MeetingContextMemory class."""

    def test_meeting_context_creation(self):
        """Test creating a meeting context memory."""
        meeting_date = datetime(2025, 1, 20, 14, 0)
        meeting = MeetingContextMemory(
            id="meeting-001",
            meeting_title="Sprint Planning",
            meeting_date=meeting_date,
            attendees=["Alice", "Bob", "Charlie"],
        )

        assert meeting.id == "meeting-001"
        assert meeting.type == MemoryType.MEETING_CONTEXT
        assert "Sprint Planning" in meeting.content
        assert "Alice" in meeting.content
        assert "2025-01-20 14:00" in meeting.content
        assert meeting.metadata["meeting_title"] == "Sprint Planning"
        assert meeting.metadata["attendees"] == ["Alice", "Bob", "Charlie"]

    def test_meeting_context_with_location(self):
        """Test meeting with location."""
        meeting_date = datetime(2025, 1, 21, 10, 0)
        meeting = MeetingContextMemory(
            id="meeting-002",
            meeting_title="Client Meeting",
            meeting_date=meeting_date,
            attendees=["Alice", "Client"],
            location="Conference Room A",
        )

        assert "Location: Conference Room A" in meeting.content
        assert meeting.metadata["location"] == "Conference Room A"

    def test_meeting_context_with_description(self):
        """Test meeting with description."""
        meeting_date = datetime(2025, 1, 22, 15, 0)
        meeting = MeetingContextMemory(
            id="meeting-003",
            meeting_title="Team Sync",
            meeting_date=meeting_date,
            attendees=["Team"],
            description="Weekly team synchronization meeting",
        )

        assert "Description: Weekly team synchronization meeting" in meeting.content
        assert meeting.metadata["description"] == "Weekly team synchronization meeting"

    def test_meeting_context_with_preparation_notes(self):
        """Test meeting with preparation notes."""
        meeting_date = datetime(2025, 1, 23, 9, 0)
        prep_notes = ["Review last week's progress", "Prepare demo", "Bring laptop"]
        meeting = MeetingContextMemory(
            id="meeting-004",
            meeting_title="Demo Session",
            meeting_date=meeting_date,
            attendees=["Dev Team", "Stakeholders"],
            preparation_notes=prep_notes,
        )

        assert "Preparation Notes:" in meeting.content
        assert "Review last week's progress" in meeting.content
        assert "Prepare demo" in meeting.content
        assert meeting.metadata["preparation_notes"] == prep_notes

    def test_meeting_context_metadata_structure(self):
        """Test meeting context metadata has all fields."""
        meeting_date = datetime(2025, 1, 24, 11, 0)
        meeting = MeetingContextMemory(
            id="meeting-005",
            meeting_title="Test Meeting",
            meeting_date=meeting_date,
            attendees=["User"],
            description="Test",
            preparation_notes=["Note"],
            location="Zoom",
        )

        metadata = meeting.metadata
        assert "meeting_title" in metadata
        assert "meeting_date" in metadata
        assert "attendees" in metadata
        assert "description" in metadata
        assert "preparation_notes" in metadata
        assert "location" in metadata


class TestUserPreferenceMemory:
    """Test UserPreferenceMemory class."""

    def test_user_preference_creation(self):
        """Test creating a user preference memory."""
        pref = UserPreferenceMemory(
            id="pref-001",
            preference_key="brief_style",
            preference_value="detailed",
            category="brief_style",
        )

        assert pref.id == "pref-001"
        assert pref.type == MemoryType.USER_PREFERENCE
        assert "brief_style" in pref.content
        assert "detailed" in pref.content
        assert pref.metadata["preference_key"] == "brief_style"
        assert pref.metadata["preference_value"] == "detailed"
        assert pref.metadata["category"] == "brief_style"

    def test_user_preference_with_different_types(self):
        """Test user preferences with different value types."""
        # String value
        pref1 = UserPreferenceMemory(
            id="pref-002",
            preference_key="notification_time",
            preference_value="08:00",
            category="notification",
        )
        assert pref1.metadata["preference_value"] == "08:00"

        # Boolean value
        pref2 = UserPreferenceMemory(
            id="pref-003",
            preference_key="enable_weather",
            preference_value=True,
            category="features",
        )
        assert pref2.metadata["preference_value"] is True

        # Integer value
        pref3 = UserPreferenceMemory(
            id="pref-004",
            preference_key="max_news_items",
            preference_value=50,
            category="priority",
        )
        assert pref3.metadata["preference_value"] == 50

        # List value
        pref4 = UserPreferenceMemory(
            id="pref-005",
            preference_key="favorite_sources",
            preference_value=["TechCrunch", "HackerNews"],
            category="news",
        )
        assert pref4.metadata["preference_value"] == ["TechCrunch", "HackerNews"]

    def test_user_preference_categories(self):
        """Test user preferences with different categories."""
        categories = ["brief_style", "notification", "priority", "features"]

        for i, category in enumerate(categories):
            pref = UserPreferenceMemory(
                id=f"pref-cat-{i}",
                preference_key=f"test_key_{i}",
                preference_value="test",
                category=category,
            )

            assert pref.metadata["category"] == category
            assert category in pref.content


class TestRetrievalResult:
    """Test RetrievalResult class."""

    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        memory = Memory(
            id="test-001",
            type=MemoryType.DAILY_BRIEF,
            content="Test content",
            metadata={},
        )

        result = RetrievalResult(
            memory=memory,
            relevance_score=0.85,
            distance=0.15,
        )

        assert result.memory == memory
        assert result.relevance_score == 0.85
        assert result.distance == 0.15

    def test_retrieval_result_without_distance(self):
        """Test retrieval result without distance metric."""
        memory = Memory(
            id="test-002",
            type=MemoryType.ACTION_ITEM,
            content="Task",
            metadata={},
        )

        result = RetrievalResult(
            memory=memory,
            relevance_score=0.92,
        )

        assert result.memory == memory
        assert result.relevance_score == 0.92
        assert result.distance is None

    def test_retrieval_result_to_dict(self):
        """Test converting retrieval result to dictionary."""
        timestamp = datetime(2025, 1, 15, 12, 0)
        memory = Memory(
            id="test-003",
            type=MemoryType.MEETING_CONTEXT,
            content="Meeting content",
            metadata={"title": "Test Meeting"},
            timestamp=timestamp,
        )

        result = RetrievalResult(
            memory=memory,
            relevance_score=0.75,
            distance=0.25,
        )

        result_dict = result.to_dict()

        assert "memory" in result_dict
        assert "relevance_score" in result_dict
        assert "distance" in result_dict
        assert result_dict["relevance_score"] == 0.75
        assert result_dict["distance"] == 0.25
        assert result_dict["memory"]["id"] == "test-003"

    def test_retrieval_results_sorting_by_relevance(self):
        """Test that retrieval results can be sorted by relevance."""
        memories = [
            Memory(id=f"mem-{i}", type=MemoryType.CONVERSATION, content=f"Content {i}", metadata={})
            for i in range(5)
        ]

        results = [
            RetrievalResult(memory=memories[0], relevance_score=0.5),
            RetrievalResult(memory=memories[1], relevance_score=0.9),
            RetrievalResult(memory=memories[2], relevance_score=0.3),
            RetrievalResult(memory=memories[3], relevance_score=0.7),
            RetrievalResult(memory=memories[4], relevance_score=0.85),
        ]

        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        assert sorted_results[0].relevance_score == 0.9
        assert sorted_results[1].relevance_score == 0.85
        assert sorted_results[2].relevance_score == 0.7
        assert sorted_results[3].relevance_score == 0.5
        assert sorted_results[4].relevance_score == 0.3

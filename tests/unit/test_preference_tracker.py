"""Unit tests for User Preference Tracker.

Following TDD - write tests first, then implement.
Tests the preference tracking service that learns from user interactions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
import shutil
import os

from src.services.preference_tracker import PreferenceTracker, PreferenceCategory
from src.memory.models import UserPreferenceMemory, MemoryType
from src.memory.retrieval import MemoryRetriever


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB, clean up after test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
async def preference_tracker(temp_chroma_dir):
    """Create a PreferenceTracker instance with temporary directory."""
    tracker = PreferenceTracker(persist_directory=temp_chroma_dir)
    return tracker


class TestPreferenceTrackerInitialization:
    """Test PreferenceTracker initialization."""

    def test_tracker_initialization(self, temp_chroma_dir):
        """Test creating tracker with default parameters."""
        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        assert tracker.memory_retriever is not None
        assert tracker.user_id == "default"

    def test_tracker_with_custom_user_id(self, temp_chroma_dir):
        """Test creating tracker with custom user ID."""
        tracker = PreferenceTracker(
            persist_directory=temp_chroma_dir,
            user_id="user-123"
        )

        assert tracker.user_id == "user-123"


class TestStorePreference:
    """Test storing user preferences."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_content_preference(self, MockOllama, temp_chroma_dir):
        """Test storing a content type preference."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            category=PreferenceCategory.CONTENT,
            key="preferred_news_topics",
            value=["AI", "Technology", "Science"],
            confidence=0.9
        )

        # Verify preference was stored in memory
        prefs = await tracker.get_preferences(PreferenceCategory.CONTENT)
        assert len(prefs) >= 1
        assert any(p.metadata.get("key") == "preferred_news_topics" for p in prefs)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_formatting_preference(self, MockOllama, temp_chroma_dir):
        """Test storing a formatting preference."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            category=PreferenceCategory.FORMATTING,
            key="brief_length",
            value="concise",
            confidence=0.8
        )

        prefs = await tracker.get_preferences(PreferenceCategory.FORMATTING)
        assert len(prefs) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_store_priority_preference(self, MockOllama, temp_chroma_dir):
        """Test storing a priority preference."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            category=PreferenceCategory.PRIORITY,
            key="important_senders",
            value=["boss@company.com", "client@company.com"],
            confidence=1.0
        )

        prefs = await tracker.get_preferences(PreferenceCategory.PRIORITY)
        assert len(prefs) >= 1


class TestRetrievePreferences:
    """Test retrieving user preferences."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_all_preferences(self, MockOllama, temp_chroma_dir):
        """Test retrieving all preferences."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        # Store multiple preferences
        await tracker.store_preference(
            PreferenceCategory.CONTENT, "topic", ["AI"], 0.9
        )
        await tracker.store_preference(
            PreferenceCategory.FORMATTING, "length", "short", 0.8
        )

        all_prefs = await tracker.get_all_preferences()
        assert len(all_prefs) >= 2

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_preferences_by_category(self, MockOllama, temp_chroma_dir):
        """Test retrieving preferences filtered by category."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            PreferenceCategory.CONTENT, "topics", ["Tech"], 0.9
        )
        await tracker.store_preference(
            PreferenceCategory.FORMATTING, "style", "bullet", 0.8
        )

        content_prefs = await tracker.get_preferences(PreferenceCategory.CONTENT)
        assert all(p.metadata.get("category") == "content" for p in content_prefs)

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_preference_by_key(self, MockOllama, temp_chroma_dir):
        """Test retrieving a specific preference by key."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            PreferenceCategory.CONTENT, "news_sources", ["HN", "RSS"], 0.9
        )

        pref = await tracker.get_preference("news_sources")
        assert pref is not None
        assert pref.metadata.get("key") == "news_sources"


class TestUpdatePreference:
    """Test updating existing preferences."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_update_preference_value(self, MockOllama, temp_chroma_dir):
        """Test updating an existing preference value."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        # Store initial preference
        await tracker.store_preference(
            PreferenceCategory.CONTENT, "topics", ["AI"], 0.8
        )

        # Update the preference
        await tracker.update_preference(
            key="topics",
            value=["AI", "Machine Learning", "Robotics"],
            confidence=0.95
        )

        pref = await tracker.get_preference("topics")
        value = pref.metadata.get("value")
        # Parse JSON if string
        import json
        if isinstance(value, str):
            value = json.loads(value)
        assert len(value) == 3
        assert "Robotics" in value

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_update_preference_confidence(self, MockOllama, temp_chroma_dir):
        """Test updating preference confidence."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            PreferenceCategory.FORMATTING, "style", "verbose", 0.5
        )

        # User confirms preference, increase confidence
        await tracker.update_preference(
            key="style",
            value="verbose",
            confidence=0.9
        )

        pref = await tracker.get_preference("style")
        assert pref.metadata.get("confidence") >= 0.9


class TestLearnFromFeedback:
    """Test learning preferences from user feedback."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_learn_from_positive_feedback(self, MockOllama, temp_chroma_dir):
        """Test learning from positive user feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        # User likes AI news
        await tracker.learn_from_feedback(
            feedback_type="liked_article",
            context={"topic": "Artificial Intelligence", "source": "HackerNews"}
        )

        # Check if preference was created
        all_prefs = await tracker.get_all_preferences()
        assert len(all_prefs) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_learn_from_negative_feedback(self, MockOllama, temp_chroma_dir):
        """Test learning from negative user feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        # User dislikes sports news
        await tracker.learn_from_feedback(
            feedback_type="disliked_article",
            context={"topic": "Sports", "source": "RSS"}
        )

        all_prefs = await tracker.get_all_preferences()
        assert len(all_prefs) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_reinforce_existing_preference(self, MockOllama, temp_chroma_dir):
        """Test reinforcing an existing preference with repeated feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        # Store initial preference with low confidence
        await tracker.store_preference(
            PreferenceCategory.CONTENT, "liked_topics", ["AI"], 0.6
        )

        # User repeatedly likes AI articles
        for _ in range(3):
            await tracker.learn_from_feedback(
                feedback_type="liked_article",
                context={"topic": "AI"}
            )

        # Confidence should increase
        pref = await tracker.get_preference("liked_topics")
        # Note: Implementation should increase confidence with reinforcement


class TestPreferenceExpiry:
    """Test preference expiry and staleness."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_recent_preferences(self, MockOllama, temp_chroma_dir):
        """Test retrieving only recent preferences."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            PreferenceCategory.CONTENT, "recent_pref", ["value"], 0.9
        )

        recent_prefs = await tracker.get_recent_preferences(days=7)
        assert len(recent_prefs) >= 1


class TestPreferenceExport:
    """Test exporting preferences for analysis."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_export_preferences_to_dict(self, MockOllama, temp_chroma_dir):
        """Test exporting all preferences as dictionary."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        tracker = PreferenceTracker(persist_directory=temp_chroma_dir)

        await tracker.store_preference(
            PreferenceCategory.CONTENT, "topics", ["AI"], 0.9
        )

        export = await tracker.export_preferences()
        assert isinstance(export, dict)
        assert len(export) > 0

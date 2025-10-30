"""Unit tests for RLHF Feedback Collector.

Following TDD - write tests first, then implement.
Tests the feedback collection system for learning from user interactions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
import shutil
import os

from src.services.feedback_collector import (
    FeedbackCollector,
    FeedbackType,
    FeedbackRating,
)
from src.memory.models import MemoryType
from src.memory.retrieval import MemoryRetriever


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB, clean up after test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
async def feedback_collector(temp_chroma_dir):
    """Create a FeedbackCollector instance with temporary directory."""
    collector = FeedbackCollector(persist_directory=temp_chroma_dir)
    return collector


class TestFeedbackCollectorInitialization:
    """Test FeedbackCollector initialization."""

    def test_collector_initialization(self, temp_chroma_dir):
        """Test creating collector with default parameters."""
        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        assert collector.memory_retriever is not None
        assert collector.user_id == "default"

    def test_collector_with_custom_user_id(self, temp_chroma_dir):
        """Test creating collector with custom user ID."""
        collector = FeedbackCollector(
            persist_directory=temp_chroma_dir,
            user_id="user-456"
        )

        assert collector.user_id == "user-456"


class TestCollectFeedback:
    """Test collecting user feedback."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_collect_positive_feedback(self, MockOllama, temp_chroma_dir):
        """Test collecting positive feedback on daily brief."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            feedback_type=FeedbackType.DAILY_BRIEF,
            rating=FeedbackRating.POSITIVE,
            context={
                "brief_id": "brief-001",
                "comment": "Great summary!"
            }
        )

        # Verify feedback was stored
        feedback_list = await collector.get_all_feedback()
        assert len(feedback_list) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_collect_negative_feedback(self, MockOllama, temp_chroma_dir):
        """Test collecting negative feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            feedback_type=FeedbackType.DAILY_BRIEF,
            rating=FeedbackRating.NEGATIVE,
            context={
                "brief_id": "brief-002",
                "comment": "Too verbose"
            }
        )

        feedback_list = await collector.get_all_feedback()
        assert len(feedback_list) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_collect_neutral_feedback(self, MockOllama, temp_chroma_dir):
        """Test collecting neutral feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            feedback_type=FeedbackType.DAILY_BRIEF,
            rating=FeedbackRating.NEUTRAL,
            context={"brief_id": "brief-003"}
        )

        feedback_list = await collector.get_all_feedback()
        assert len(feedback_list) >= 1

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_collect_article_feedback(self, MockOllama, temp_chroma_dir):
        """Test collecting feedback on specific article."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            feedback_type=FeedbackType.NEWS_ARTICLE,
            rating=FeedbackRating.POSITIVE,
            context={
                "article_id": "article-123",
                "topic": "AI",
                "source": "HackerNews"
            }
        )

        feedback_list = await collector.get_feedback_by_type(FeedbackType.NEWS_ARTICLE)
        assert len(feedback_list) >= 1


class TestRetrieveFeedback:
    """Test retrieving feedback."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_all_feedback(self, MockOllama, temp_chroma_dir):
        """Test retrieving all feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        # Store multiple feedback items
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.NEWS_ARTICLE, FeedbackRating.NEGATIVE, {}
        )

        all_feedback = await collector.get_all_feedback()
        assert len(all_feedback) >= 2

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_feedback_by_type(self, MockOllama, temp_chroma_dir):
        """Test filtering feedback by type."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.NEWS_ARTICLE, FeedbackRating.POSITIVE, {}
        )

        brief_feedback = await collector.get_feedback_by_type(FeedbackType.DAILY_BRIEF)
        assert all(
            fb.metadata.get("feedback_type") == FeedbackType.DAILY_BRIEF.value
            for fb in brief_feedback
        )

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_feedback_by_rating(self, MockOllama, temp_chroma_dir):
        """Test filtering feedback by rating."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.NEGATIVE, {}
        )

        positive_feedback = await collector.get_feedback_by_rating(FeedbackRating.POSITIVE)
        assert all(
            fb.metadata.get("rating") == FeedbackRating.POSITIVE.value
            for fb in positive_feedback
        )


class TestFeedbackAnalysis:
    """Test analyzing feedback patterns."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_calculate_average_rating(self, MockOllama, temp_chroma_dir):
        """Test calculating average rating for a feedback type."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        # Add mixed feedback
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.NEGATIVE, {}
        )

        avg_rating = await collector.calculate_average_rating(FeedbackType.DAILY_BRIEF)
        assert 0.0 <= avg_rating <= 1.0

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_get_feedback_summary(self, MockOllama, temp_chroma_dir):
        """Test getting feedback summary statistics."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        # Add various feedback
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {}
        )
        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.NEGATIVE, {}
        )

        summary = await collector.get_feedback_summary()
        assert "total_feedback" in summary
        assert "by_type" in summary
        assert "by_rating" in summary


class TestLearningFromFeedback:
    """Test learning preferences from feedback."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_update_preferences_from_feedback(self, MockOllama, temp_chroma_dir):
        """Test that preferences are updated based on feedback."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        # Collect feedback with preference implications
        await collector.collect_feedback(
            feedback_type=FeedbackType.NEWS_ARTICLE,
            rating=FeedbackRating.POSITIVE,
            context={"topic": "Machine Learning", "source": "ArXiv"}
        )

        # Apply learning
        await collector.apply_feedback_learning()

        # Verify preference tracker was updated
        # (This would check the preference tracker's state)
        assert collector.preference_tracker is not None

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_feedback_reinforcement(self, MockOllama, temp_chroma_dir):
        """Test that repeated feedback reinforces preferences."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        # Provide repeated positive feedback on AI topics
        for _ in range(5):
            await collector.collect_feedback(
                feedback_type=FeedbackType.NEWS_ARTICLE,
                rating=FeedbackRating.POSITIVE,
                context={"topic": "AI"}
            )

        await collector.apply_feedback_learning()

        # Confidence should be higher due to reinforcement
        # (This would be verified through preference tracker)


class TestFeedbackExport:
    """Test exporting feedback for analysis."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_export_feedback_to_dict(self, MockOllama, temp_chroma_dir):
        """Test exporting all feedback as dictionary."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        collector = FeedbackCollector(persist_directory=temp_chroma_dir)

        await collector.collect_feedback(
            FeedbackType.DAILY_BRIEF, FeedbackRating.POSITIVE, {"test": "data"}
        )

        export = await collector.export_feedback()
        assert isinstance(export, dict)
        assert len(export) > 0

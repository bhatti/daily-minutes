"""Unit tests for NewsAgent with mocks."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

import pytest

from src.agents.news_agent import (
    NewsAgent,
    NewsAgentState,
    NewsAgentAction
)
from src.models.news import NewsArticle, DataSource, Priority
from src.models.preferences import UserPreferences


@pytest.fixture
def mock_news_agent():
    """Create NewsAgent with mocked dependencies."""
    with patch('src.agents.news_agent.HackerNewsConnector') as mock_hn, \
         patch('src.agents.news_agent.RSSConnector') as mock_rss, \
         patch('src.agents.news_agent.CacheManager') as mock_cache, \
         patch('src.agents.news_agent.ObservabilityManager') as mock_obs:

        # Create agent
        agent = NewsAgent(max_articles=10)

        # Mock cache methods
        agent.cache_manager.get = MagicMock(return_value=None)
        agent.cache_manager.put = MagicMock()

        # Mock observability
        agent.observability.create_trace = MagicMock(return_value="trace-123")
        agent.observability.add_span = MagicMock()
        agent.observability.end_span = MagicMock()
        agent.observability.end_trace = MagicMock()
        agent.observability.log_llm_call = MagicMock()

        # Mock connectors
        agent.hn_connector = mock_hn.return_value
        agent.rss_connector = mock_rss.return_value

        return agent


@pytest.fixture
def sample_articles():
    """Create sample articles for testing."""
    return [
        NewsArticle(
            title="AI Breakthrough",
            url="https://example.com/ai",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            author="researcher",
            published_at=datetime(2024, 1, 1, 12, 0),
            priority=Priority.HIGH,
            relevance_score=0.9,
            tags=["ai", "breakthrough"]
        ),
        NewsArticle(
            title="Python 4.0 Released",
            url="https://example.com/python",
            source=DataSource.RSS,
            source_name="Python Blog",
            author="guido",
            published_at=datetime(2024, 1, 1, 11, 0),
            priority=Priority.MEDIUM,
            relevance_score=0.7,
            tags=["python", "release"]
        ),
        NewsArticle(
            title="Tech News Update",
            url="https://example.com/tech",
            source=DataSource.RSS,
            source_name="TechCrunch",
            author="writer",
            published_at=datetime(2024, 1, 1, 10, 0),
            priority=Priority.LOW,
            relevance_score=0.4,
            tags=["tech", "news"]
        )
    ]


@pytest.fixture
def initial_state():
    """Create initial agent state."""
    return NewsAgentState(
        current_action=NewsAgentAction.INITIALIZE,
        articles=[],
        sources_checked=[],
        error_count=0,
        reasoning="Starting news aggregation"
    )


@pytest.mark.asyncio
async def test_initialization(mock_news_agent):
    """Test agent initialization."""
    assert mock_news_agent.max_articles == 10
    assert mock_news_agent.workflow_state == "initialized"
    assert mock_news_agent.cache_manager is not None
    assert mock_news_agent.observability is not None


@pytest.mark.asyncio
async def test_reason_step_no_articles(mock_news_agent, initial_state):
    """Test reasoning step when no articles are fetched."""
    state = await mock_news_agent._reason_step(initial_state)

    assert state.current_action == NewsAgentAction.FETCH_NEWS
    assert "No articles fetched yet" in state.reasoning
    assert state.retries == 0


@pytest.mark.asyncio
async def test_reason_step_with_articles(mock_news_agent, initial_state, sample_articles):
    """Test reasoning step when articles are available."""
    initial_state.articles = sample_articles
    initial_state.sources_checked = ["hackernews", "rss"]

    state = await mock_news_agent._reason_step(initial_state)

    assert state.current_action == NewsAgentAction.RANK_ARTICLES
    assert "Successfully fetched 3 articles" in state.reasoning


@pytest.mark.asyncio
async def test_reason_step_max_retries(mock_news_agent, initial_state):
    """Test reasoning step when max retries reached."""
    initial_state.retries = 3
    initial_state.error_count = 3

    state = await mock_news_agent._reason_step(initial_state)

    assert state.current_action == NewsAgentAction.COMPLETE
    assert "Maximum retries" in state.reasoning


@pytest.mark.asyncio
async def test_act_fetch_news(mock_news_agent, initial_state, sample_articles):
    """Test act step for fetching news."""
    initial_state.current_action = NewsAgentAction.FETCH_NEWS

    # Mock connector responses
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        return_value=[sample_articles[0]]
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        return_value=[sample_articles[1], sample_articles[2]]
    )

    state = await mock_news_agent._act_step(initial_state)

    assert len(state.articles) == 3
    assert "hackernews" in state.sources_checked
    assert "rss" in state.sources_checked
    assert state.current_action == NewsAgentAction.RANK_ARTICLES


@pytest.mark.asyncio
async def test_act_fetch_news_with_errors(mock_news_agent, initial_state):
    """Test act step when fetching fails."""
    initial_state.current_action = NewsAgentAction.FETCH_NEWS

    # Mock connector failures
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        side_effect=Exception("API error")
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        side_effect=Exception("Network error")
    )

    state = await mock_news_agent._act_step(initial_state)

    assert len(state.articles) == 0
    assert state.error_count == 2
    assert state.retries == 1
    assert state.current_action == NewsAgentAction.RETRY


@pytest.mark.asyncio
async def test_act_rank_articles(mock_news_agent, initial_state, sample_articles):
    """Test act step for ranking articles."""
    initial_state.current_action = NewsAgentAction.RANK_ARTICLES
    initial_state.articles = sample_articles

    state = await mock_news_agent._act_step(initial_state)

    # Articles should be sorted by importance (high to low)
    assert state.articles[0].title == "AI Breakthrough"  # Highest importance
    assert state.articles[1].title == "Python 4.0 Released"
    assert state.articles[2].title == "Tech News Update"  # Lowest importance
    assert state.current_action == NewsAgentAction.DEDUPLICATE


@pytest.mark.asyncio
async def test_act_deduplicate(mock_news_agent, initial_state):
    """Test act step for deduplication."""
    # Create duplicate articles
    articles = [
        NewsArticle(
            title="Same Article",
            url="https://example.com/1",
            source=DataSource.HACKERNEWS,
            source_name="HN"
        ),
        NewsArticle(
            title="Same Article",
            url="https://example.com/1",
            source=DataSource.RSS,
            source_name="RSS"
        ),
        NewsArticle(
            title="Different Article",
            url="https://example.com/2",
            source=DataSource.RSS,
            source_name="RSS"
        )
    ]

    initial_state.current_action = NewsAgentAction.DEDUPLICATE
    initial_state.articles = articles

    state = await mock_news_agent._act_step(initial_state)

    # Should remove duplicate
    assert len(state.articles) == 2
    unique_urls = {a.url for a in state.articles}
    assert len(unique_urls) == 2
    assert state.current_action == NewsAgentAction.PERSONALIZE


@pytest.mark.asyncio
async def test_act_personalize_with_preferences(mock_news_agent, initial_state, sample_articles):
    """Test act step for personalization with user preferences."""
    initial_state.current_action = NewsAgentAction.PERSONALIZE
    initial_state.articles = sample_articles

    # Mock user preferences
    preferences = UserPreferences(
        user_id="test-user",
        interests=["ai", "python"],
        excluded_topics=["crypto"],
        preferred_sources=["HackerNews"]
    )

    # Mock cache to return preferences
    mock_news_agent.cache_manager.get = MagicMock(return_value=preferences)

    state = await mock_news_agent._act_step(initial_state)

    # Articles should be re-scored based on preferences
    assert state.articles[0].tags[0] in ["ai", "python"]  # Should prioritize these
    assert state.current_action == NewsAgentAction.APPLY_RL


@pytest.mark.asyncio
async def test_act_apply_rl(mock_news_agent, initial_state, sample_articles):
    """Test act step for applying reinforcement learning."""
    initial_state.current_action = NewsAgentAction.APPLY_RL
    initial_state.articles = sample_articles

    # Mock RL state
    rl_state = ReinforcementState(
        q_values={"ai": 0.9, "python": 0.8, "tech": 0.3}
    )

    # Mock cache to return RL state
    mock_news_agent.cache_manager.get = MagicMock(
        side_effect=lambda key, **kwargs: rl_state if "rl_state" in key else None
    )

    state = await mock_news_agent._act_step(initial_state)

    # Articles should be adjusted based on Q-values
    assert state.current_action == NewsAgentAction.COMPLETE


@pytest.mark.asyncio
async def test_act_complete(mock_news_agent, initial_state, sample_articles):
    """Test act step for completion."""
    initial_state.current_action = NewsAgentAction.COMPLETE
    initial_state.articles = sample_articles[:2]  # Limit to 2 articles

    state = await mock_news_agent._act_step(initial_state)

    # Should truncate to max_articles
    assert len(state.articles) <= mock_news_agent.max_articles
    # Observability should be called
    mock_news_agent.observability.end_trace.assert_called_once()


@pytest.mark.asyncio
async def test_run_complete_workflow(mock_news_agent, sample_articles):
    """Test complete workflow execution."""
    # Mock connector responses
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        return_value=[sample_articles[0]]
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        return_value=[sample_articles[1], sample_articles[2]]
    )

    # Run workflow
    articles = await mock_news_agent.run()

    assert len(articles) == 3
    assert all(isinstance(a, NewsArticle) for a in articles)
    # Should be sorted by importance
    assert articles[0].priority == Priority.HIGH


@pytest.mark.asyncio
async def test_search_news(mock_news_agent):
    """Test searching for specific news."""
    # Mock search results
    search_results = [
        NewsArticle(
            title="Python Tutorial",
            url="https://example.com/python",
            source=DataSource.HACKERNEWS,
            source_name="HN"
        )
    ]

    mock_news_agent.hn_connector.search_stories = AsyncMock(
        return_value=search_results
    )

    results = await mock_news_agent.search_news("python")

    assert len(results) == 1
    assert results[0].title == "Python Tutorial"
    mock_news_agent.hn_connector.search_stories.assert_called_once_with("python", limit=20)


@pytest.mark.asyncio
async def test_get_trending_topics(mock_news_agent):
    """Test getting trending topics."""
    # Mock trending topics
    mock_news_agent.hn_connector.get_trending_topics = AsyncMock(
        return_value=["ai", "python", "rust"]
    )

    topics = await mock_news_agent.get_trending_topics()

    assert len(topics) == 3
    assert "ai" in topics
    mock_news_agent.hn_connector.get_trending_topics.assert_called_once()


@pytest.mark.asyncio
async def test_update_user_feedback(mock_news_agent):
    """Test updating user feedback for RL."""
    article = NewsArticle(
        title="Test Article",
        url="https://example.com/test",
        source=DataSource.RSS,
        source_name="Test",
        tags=["python", "ai"]
    )

    # Mock getting existing RL state
    rl_state = ReinforcementState(
        q_values={"python": 0.5, "ai": 0.5}
    )
    mock_news_agent.cache_manager.get = MagicMock(return_value=rl_state)

    # Update feedback
    await mock_news_agent.update_user_feedback(
        article=article,
        clicked=True,
        time_spent=120,
        shared=True
    )

    # Cache should be updated with new Q-values
    mock_news_agent.cache_manager.put.assert_called()

    # Check the updated RL state was saved
    call_args = mock_news_agent.cache_manager.put.call_args
    saved_state = call_args[0][1]  # Second argument is the value

    # Q-values should increase for positive feedback
    assert saved_state.q_values["python"] > 0.5
    assert saved_state.q_values["ai"] > 0.5


@pytest.mark.asyncio
async def test_generate_summary(mock_news_agent, sample_articles):
    """Test generating news summary."""
    summary = await mock_news_agent.generate_summary(sample_articles)

    assert summary.total_articles == 3
    assert summary.unread_articles == 3  # All new articles are unread
    assert len(summary.top_articles) == 3
    assert summary.source == DataSource.CUSTOM

    # Check categories are extracted from tags
    assert "ai" in summary.categories
    assert "python" in summary.categories


@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_news_agent):
    """Test workflow state transitions."""
    # Initial state
    assert mock_news_agent.workflow_state == "initialized"

    # Mock connectors
    mock_news_agent.hn_connector.execute_async = AsyncMock(return_value=[])
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(return_value=[])

    # Run workflow
    await mock_news_agent.run()

    # Should transition through states
    assert mock_news_agent.workflow_state == "completed"


@pytest.mark.asyncio
async def test_caching_behavior(mock_news_agent, sample_articles):
    """Test caching of articles."""
    # First run - should fetch from sources
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        return_value=sample_articles[:1]
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        return_value=sample_articles[1:]
    )

    articles1 = await mock_news_agent.run()

    # Cache should be updated
    mock_news_agent.cache_manager.put.assert_called()

    # Second run - setup cache hit
    mock_news_agent.cache_manager.get = MagicMock(return_value=articles1)
    mock_news_agent.hn_connector.execute_async.reset_mock()
    mock_news_agent.rss_connector.fetch_all_feeds.reset_mock()

    articles2 = await mock_news_agent.run()

    # Should not call connectors again (used cache)
    mock_news_agent.hn_connector.execute_async.assert_not_called()
    mock_news_agent.rss_connector.fetch_all_feeds.assert_not_called()


@pytest.mark.asyncio
async def test_error_recovery(mock_news_agent):
    """Test error recovery mechanism."""
    # First attempt fails
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        side_effect=[Exception("First failure"), []]  # Fails first, succeeds second
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        return_value=[]
    )

    # Should retry and recover
    articles = await mock_news_agent.run()

    # Should have called HN connector twice (initial + retry)
    assert mock_news_agent.hn_connector.execute_async.call_count == 2

    # Workflow should complete despite initial error
    assert mock_news_agent.workflow_state == "completed"


@pytest.mark.asyncio
async def test_observability_tracing(mock_news_agent, sample_articles):
    """Test observability tracing throughout workflow."""
    mock_news_agent.hn_connector.execute_async = AsyncMock(
        return_value=sample_articles
    )
    mock_news_agent.rss_connector.fetch_all_feeds = AsyncMock(
        return_value=[]
    )

    await mock_news_agent.run()

    # Trace should be created and ended
    mock_news_agent.observability.create_trace.assert_called_once()
    mock_news_agent.observability.end_trace.assert_called_once()

    # Multiple spans should be created for different steps
    assert mock_news_agent.observability.add_span.call_count > 3  # At least reason, act, complete
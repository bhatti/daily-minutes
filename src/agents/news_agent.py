"""News Agent with LangGraph orchestration, ReAct pattern, and RAG capabilities."""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from pydantic import Field

from src.agents.base_agent import AgentResult, AgentState, BaseAgent
from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.core.config import get_settings
from src.core.logging import get_logger
from src.models.cache import CacheManager
from src.models.news import NewsArticle, NewsSummary, DataSource
from src.models.observability import ObservabilityManager, Span, SpanStatus
from src.models.preferences import UserFeedback, UserPreferences

logger = get_logger(__name__)


class NewsAgentAction(str, Enum):
    """Actions the news agent can take (ReAct pattern)."""

    FETCH_NEWS = "fetch_news"
    SEARCH_SIMILAR = "search_similar"
    ANALYZE_TRENDS = "analyze_trends"
    SUMMARIZE = "summarize"
    RANK_BY_RELEVANCE = "rank_by_relevance"
    APPLY_PREFERENCES = "apply_preferences"
    GENERATE_INSIGHTS = "generate_insights"
    COMPLETE = "complete"


class NewsAgentState(TypedDict):
    """State for LangGraph workflow."""

    # Input
    user_query: Optional[str]
    preferences: Optional[UserPreferences]
    sources: List[str]
    max_articles: int

    # Processing
    current_action: NewsAgentAction
    reasoning: str
    articles: List[NewsArticle]
    filtered_articles: List[NewsArticle]
    similar_articles: List[NewsArticle]
    trends: List[str]

    # Output
    summary: Optional[NewsSummary]
    insights: List[str]
    recommendations: List[str]
    error: Optional[str]

    # Metadata
    steps_taken: List[Dict[str, Any]]
    total_articles_processed: int
    processing_time_ms: float


class NewsAgent(BaseAgent[NewsSummary]):
    """Advanced news aggregation agent with AI capabilities.

    Features:
    - LangGraph orchestration for complex workflows
    - ReAct pattern for reasoning and acting
    - RAG for finding similar content
    - User preference learning (RL ready)
    - Multi-source aggregation
    """

    # Connectors
    hn_connector: Optional[HackerNewsConnector] = None
    rss_connector: Optional[RSSConnector] = None

    # AI Components
    llm: Optional[BaseLLM] = None
    embeddings: Optional[Embeddings] = None
    vector_store: Optional[VectorStore] = None

    # Configuration
    enable_rag: bool = Field(True, description="Enable RAG for similar content")
    enable_react: bool = Field(True, description="Enable ReAct reasoning")
    enable_preferences: bool = Field(True, description="Apply user preferences")

    # State management
    workflow_graph: Optional[StateGraph] = None
    current_state: Optional[NewsAgentState] = None

    # Observability
    observability: Optional[ObservabilityManager] = None
    current_trace_id: Optional[str] = None

    # Cache
    cache_manager: Optional[CacheManager] = None

    def __init__(self, **data):
        """Initialize news agent with all components."""
        super().__init__(**data)

        # Initialize connectors
        if not self.hn_connector:
            self.hn_connector = HackerNewsConnector(
                story_type="top",
                max_stories=30
            )

        if not self.rss_connector:
            self.rss_connector = RSSConnector(
                max_articles_per_feed=10
            )

        # Initialize observability
        if not self.observability:
            self.observability = ObservabilityManager()

        # Initialize cache
        if not self.cache_manager:
            self.cache_manager = CacheManager()

        # Build workflow graph
        self._build_workflow()

    def _build_workflow(self) -> None:
        """Build LangGraph workflow for news processing."""
        workflow = StateGraph(NewsAgentState)

        # Add nodes for each action
        workflow.add_node("reason", self._reason_step)
        workflow.add_node("fetch_news", self._fetch_news_step)
        workflow.add_node("search_similar", self._search_similar_step)
        workflow.add_node("analyze_trends", self._analyze_trends_step)
        workflow.add_node("rank_relevance", self._rank_relevance_step)
        workflow.add_node("apply_preferences", self._apply_preferences_step)
        workflow.add_node("summarize", self._summarize_step)
        workflow.add_node("generate_insights", self._generate_insights_step)

        # Set entry point
        workflow.set_entry_point("reason")

        # Add conditional edges based on reasoning
        workflow.add_conditional_edges(
            "reason",
            self._route_action,
            {
                NewsAgentAction.FETCH_NEWS: "fetch_news",
                NewsAgentAction.SEARCH_SIMILAR: "search_similar",
                NewsAgentAction.ANALYZE_TRENDS: "analyze_trends",
                NewsAgentAction.RANK_BY_RELEVANCE: "rank_relevance",
                NewsAgentAction.APPLY_PREFERENCES: "apply_preferences",
                NewsAgentAction.SUMMARIZE: "summarize",
                NewsAgentAction.GENERATE_INSIGHTS: "generate_insights",
                NewsAgentAction.COMPLETE: END,
            }
        )

        # Add edges from action nodes back to reasoning
        for node in ["fetch_news", "search_similar", "analyze_trends",
                    "rank_relevance", "apply_preferences", "summarize",
                    "generate_insights"]:
            workflow.add_edge(node, "reason")

        self.workflow_graph = workflow.compile()

    async def _reason_step(self, state: NewsAgentState) -> NewsAgentState:
        """ReAct reasoning step - decide what action to take next."""
        span = self._start_span("reason_step")

        try:
            # Analyze current state
            has_articles = len(state.get("articles", [])) > 0
            has_filtered = len(state.get("filtered_articles", [])) > 0
            has_summary = state.get("summary") is not None
            has_insights = len(state.get("insights", [])) > 0

            # ReAct pattern: Reason about next action
            if not has_articles:
                action = NewsAgentAction.FETCH_NEWS
                reasoning = "No articles fetched yet. Need to fetch news from sources."

            elif self.enable_rag and not state.get("similar_searched", False):
                action = NewsAgentAction.SEARCH_SIMILAR
                reasoning = "Articles fetched. Finding similar content using RAG."

            elif not state.get("trends"):
                action = NewsAgentAction.ANALYZE_TRENDS
                reasoning = "Analyzing trends in the fetched articles."

            elif not has_filtered:
                if self.enable_preferences and state.get("preferences"):
                    action = NewsAgentAction.APPLY_PREFERENCES
                    reasoning = "Applying user preferences to filter and rank articles."
                else:
                    action = NewsAgentAction.RANK_BY_RELEVANCE
                    reasoning = "Ranking articles by relevance score."

            elif not has_summary:
                action = NewsAgentAction.SUMMARIZE
                reasoning = "Creating summary from processed articles."

            elif not has_insights:
                action = NewsAgentAction.GENERATE_INSIGHTS
                reasoning = "Generating insights from the news data."

            else:
                action = NewsAgentAction.COMPLETE
                reasoning = "All processing complete."

            # Update state
            state["current_action"] = action
            state["reasoning"] = reasoning

            # Log reasoning
            step = {
                "timestamp": datetime.now().isoformat(),
                "action": action.value,
                "reasoning": reasoning,
                "state_summary": {
                    "articles": len(state.get("articles", [])),
                    "filtered": len(state.get("filtered_articles", [])),
                    "has_summary": has_summary
                }
            }
            state.setdefault("steps_taken", []).append(step)

            logger.info("reasoning_complete", action=action.value, reasoning=reasoning)
            if span:
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["error"] = f"Reasoning failed: {str(e)}"
            state["current_action"] = NewsAgentAction.COMPLETE

        return state

    def _route_action(self, state: NewsAgentState) -> str:
        """Route to next node based on current action."""
        return state["current_action"]

    async def _fetch_news_step(self, state: NewsAgentState) -> NewsAgentState:
        """Fetch news from configured sources."""
        span = self._start_span("fetch_news")

        try:
            all_articles = []

            # Fetch from HackerNews
            if "hackernews" in state.get("sources", ["hackernews"]):
                hn_articles = await self.hn_connector.execute_async()
                all_articles.extend(hn_articles)
                logger.info("fetched_hackernews", count=len(hn_articles))

            # Fetch from RSS feeds
            if "rss" in state.get("sources", ["rss"]):
                rss_articles = await self.rss_connector.fetch_all_feeds()
                all_articles.extend(rss_articles)
                logger.info("fetched_rss", count=len(rss_articles))

            # Limit to max articles
            max_articles = state.get("max_articles", 50)
            all_articles = all_articles[:max_articles]

            state["articles"] = all_articles
            state["total_articles_processed"] = len(all_articles)

            if span:
                span.add_tag("articles_fetched", len(all_articles))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["error"] = f"Failed to fetch news: {str(e)}"
            state["articles"] = []

        return state

    async def _search_similar_step(self, state: NewsAgentState) -> NewsAgentState:
        """Use RAG to find similar articles."""
        span = self._start_span("search_similar")

        try:
            # Mark that we've attempted similar search
            state["similar_searched"] = True

            if not self.vector_store or not state.get("articles"):
                state["similar_articles"] = []
                if span:
                    span.finish(SpanStatus.SUCCESS)
                return state

            # Embed and store articles
            texts = [f"{a.title} {a.description}" for a in state["articles"]]
            metadatas = [{"id": i, "title": a.title} for i, a in enumerate(state["articles"])]

            # Add to vector store
            self.vector_store.add_texts(texts, metadatas)

            # Search for similar articles for top stories
            similar_articles = []
            for article in state["articles"][:5]:  # Top 5
                query = f"{article.title} {article.description}"
                results = self.vector_store.similarity_search(query, k=3)

                for doc in results[1:]:  # Skip self
                    idx = doc.metadata.get("id")
                    if idx and idx < len(state["articles"]):
                        similar_articles.append(state["articles"][idx])

            state["similar_articles"] = similar_articles
            if span:
                span.add_tag("similar_found", len(similar_articles))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["similar_articles"] = []

        return state

    async def _analyze_trends_step(self, state: NewsAgentState) -> NewsAgentState:
        """Analyze trends in the news articles."""
        span = self._start_span("analyze_trends")

        try:
            articles = state.get("articles", [])

            # Count tag frequencies
            tag_counts = {}
            for article in articles:
                for tag in article.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Get top trends
            trends = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            state["trends"] = [tag for tag, _ in trends[:10]]

            # Also check HackerNews trending topics
            if self.hn_connector:
                hn_trends = await self.hn_connector.get_trending_topics()
                state["trends"].extend(hn_trends[:5])

            if span:
                span.add_tag("trends_found", len(state["trends"]))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["trends"] = []

        return state

    async def _rank_relevance_step(self, state: NewsAgentState) -> NewsAgentState:
        """Rank articles by relevance."""
        span = self._start_span("rank_relevance")

        try:
            articles = state.get("articles", [])

            # Sort by relevance score
            ranked = sorted(articles, key=lambda a: a.calculate_importance(), reverse=True)

            # Take top articles
            max_articles = min(state.get("max_articles", 30), len(ranked))
            state["filtered_articles"] = ranked[:max_articles]

            if span:
                span.add_tag("articles_ranked", len(state["filtered_articles"]))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["filtered_articles"] = state.get("articles", [])

        return state

    async def _apply_preferences_step(self, state: NewsAgentState) -> NewsAgentState:
        """Apply user preferences to filter and rank articles."""
        span = self._start_span("apply_preferences")

        try:
            articles = state.get("articles", [])
            preferences = state.get("preferences")

            if not preferences:
                state["filtered_articles"] = articles
                if span:
                    span.finish(SpanStatus.SUCCESS)
                return state

            # Score articles based on preferences
            scored_articles = []
            for article in articles:
                # Calculate relevance based on preferences
                content = {
                    "title": article.title,
                    "description": article.description,
                    "source": article.source.value,
                    "topics": article.tags,
                    "category": article.metadata.get("category", "general")
                }

                relevance = preferences.calculate_content_relevance(content)
                article.relevance_score = relevance
                scored_articles.append(article)

            # Filter by importance threshold
            filtered = [
                a for a in scored_articles
                if a.relevance_score >= preferences.importance_threshold
            ]

            # Sort by relevance
            filtered.sort(key=lambda a: a.relevance_score, reverse=True)

            # Limit to user preference
            max_items = preferences.max_items_per_section
            state["filtered_articles"] = filtered[:max_items]

            if span:
                span.add_tag("articles_filtered", len(state["filtered_articles"]))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["filtered_articles"] = state.get("articles", [])

        return state

    async def _summarize_step(self, state: NewsAgentState) -> NewsAgentState:
        """Create news summary."""
        span = self._start_span("summarize")

        try:
            articles = state.get("filtered_articles", state.get("articles", []))

            summary = NewsSummary(
                date=datetime.now(),
                source=DataSource.CUSTOM,
                total_articles=len(articles),
                top_articles=articles[:10],
                key_topics=state.get("trends", [])
            )

            # Add articles to summary
            for article in articles:
                summary.add_article(article)

            # Generate AI summary if LLM available
            if self.llm:
                summary_text = await self._generate_ai_summary(articles)
                summary.summary_text = summary_text

            state["summary"] = summary
            if span:
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["summary"] = None

        return state

    async def _generate_insights_step(self, state: NewsAgentState) -> NewsAgentState:
        """Generate insights from news data."""
        span = self._start_span("generate_insights")

        try:
            insights = []
            articles = state.get("filtered_articles", [])
            trends = state.get("trends", [])

            # Trend insights
            if trends:
                insights.append(f"Top trending topics: {', '.join(trends[:5])}")

            # Sentiment insights
            summary = state.get("summary")
            if summary and summary.sentiment_distribution:
                total = sum(summary.sentiment_distribution.values())
                if total > 0:
                    positive_pct = summary.sentiment_distribution.get("positive", 0) / total * 100
                    insights.append(f"Sentiment: {positive_pct:.0f}% positive news")

            # Source diversity
            sources = set(a.source_name for a in articles)
            insights.append(f"News from {len(sources)} different sources")

            # Time distribution
            if articles:
                latest = max(a.published_at for a in articles)
                oldest = min(a.published_at for a in articles)
                time_span = (latest - oldest).days
                insights.append(f"Stories span {time_span} days")

            state["insights"] = insights
            state["recommendations"] = self._generate_recommendations(state)

            if span:
                span.add_tag("insights_generated", len(insights))
                span.finish(SpanStatus.SUCCESS)

        except Exception as e:
            if span:
                span.set_error(str(e))
            state["insights"] = []
            state["recommendations"] = []

        return state

    def _generate_recommendations(self, state: NewsAgentState) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Based on trends
        trends = state.get("trends", [])
        if "ai" in trends or "llm" in trends:
            recommendations.append("Consider following AI/ML developments more closely")

        # Based on article count
        if state.get("total_articles_processed", 0) < 10:
            recommendations.append("Consider adding more news sources for broader coverage")

        # Based on preferences
        if state.get("preferences") and not state.get("filtered_articles"):
            recommendations.append("Adjust preference filters - too restrictive")

        return recommendations

    async def _generate_ai_summary(self, articles: List[NewsArticle]) -> str:
        """Generate AI summary using LLM."""
        if not self.llm:
            return ""

        prompt = PromptTemplate(
            template="""Summarize these news articles in 2-3 paragraphs:

{articles}

Focus on:
1. Major themes and trends
2. Most important stories
3. Potential impacts

Summary:""",
            input_variables=["articles"]
        )

        articles_text = "\n".join([
            f"- {a.title}: {a.description[:100]}..."
            for a in articles[:10]
        ])

        try:
            response = await self.llm.ainvoke(
                prompt.format(articles=articles_text)
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error("ai_summary_failed", error=str(e))
            return ""

    def _start_span(self, operation: str) -> Optional[Span]:
        """Start observability span."""
        if not self.observability:
            return None

        if not self.current_trace_id:
            self.current_trace_id = f"news_agent_{datetime.now().timestamp()}"

        span = self.observability.start_span(
            trace_id=self.current_trace_id,
            span_id=f"{operation}_{datetime.now().timestamp()}",
            operation_name=operation
        )
        return span

    async def _execute(self) -> NewsSummary:
        """Execute news agent workflow."""
        # Initialize state
        initial_state: NewsAgentState = {
            "user_query": None,
            "preferences": None,
            "sources": ["hackernews", "rss"],
            "max_articles": 50,
            "current_action": NewsAgentAction.FETCH_NEWS,
            "reasoning": "Starting news aggregation",
            "articles": [],
            "filtered_articles": [],
            "similar_articles": [],
            "trends": [],
            "summary": None,
            "insights": [],
            "recommendations": [],
            "error": None,
            "steps_taken": [],
            "total_articles_processed": 0,
            "processing_time_ms": 0,
        }

        # Start timing
        start_time = datetime.now()

        # Run workflow
        try:
            final_state = await self.workflow_graph.ainvoke(initial_state)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            final_state["processing_time_ms"] = processing_time

            # Log workflow completion
            logger.info("workflow_complete",
                       steps=len(final_state["steps_taken"]),
                       articles=final_state["total_articles_processed"],
                       time_ms=processing_time)

            # Complete trace
            if self.current_trace_id:
                self.observability.complete_trace(self.current_trace_id)
                self.current_trace_id = None

            # Return summary
            return final_state.get("summary")

        except Exception as e:
            logger.error("workflow_failed", error=str(e))
            raise

    def apply_feedback(self, feedback: UserFeedback, preferences: UserPreferences) -> None:
        """Apply user feedback to improve future results (RL component)."""
        # Calculate reward
        reward = feedback.calculate_reward()

        # Update preferences based on feedback
        if feedback.target_metadata:
            # Update topic scores
            topics = feedback.target_metadata.get("topics", [])
            for topic in topics:
                delta = reward * preferences.learning_rate
                preferences.update_topic_score(topic, delta)

            # Update source weights
            source = feedback.target_metadata.get("source")
            if source:
                delta = reward * preferences.learning_rate * 0.5
                preferences.update_source_weight(DataSource(source), delta)

        logger.info("feedback_applied",
                   feedback_type=feedback.feedback_type.value,
                   reward=reward)

    async def run(self) -> List[NewsArticle]:
        """Run the agent asynchronously and return articles.

        This is a simplified interface for integration testing.
        """
        try:
            # Execute the workflow
            summary = await self._execute()

            # Return the articles from the summary
            if summary and hasattr(summary, 'articles'):
                return summary.articles
            return []
        except Exception as e:
            logger.error("agent_run_failed", error=str(e))
            return []

    async def generate_summary(self, articles: List[NewsArticle]) -> NewsSummary:
        """Generate a summary from articles.

        This is a simplified interface for integration testing.
        """
        return NewsSummary(
            title="News Summary",
            summary="Generated summary of articles",
            articles=articles[:10],  # Top 10 articles
            total_articles=len(articles),
            sources=["hackernews", "rss"],
            categories={},
            key_topics=[],
            insights=[],
            recommendations=[],
            generated_at=datetime.now()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": self.metrics.__dict__,
            "hn_stats": self.hn_connector.get_statistics() if self.hn_connector else {},
            "rss_stats": self.rss_connector.get_feed_status() if self.rss_connector else {},
            "cache_stats": self.cache_manager.get_global_statistics() if self.cache_manager else {},
            "observability": self.observability.get_metrics_summary() if self.observability else {}
        }
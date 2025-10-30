"""Enhanced News Agent with integrated AI services."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentResult, BaseAgent
from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.core.logging import get_logger
from src.models.news import NewsArticle, NewsSummary, DataSource, Priority
from src.services.ollama_service import get_ollama_service
from src.services.rag_service import get_rag_service
from src.services.mcp_server import get_mcp_server
from src.services.langgraph_orchestrator import get_orchestrator

logger = get_logger(__name__)


class NewsAgentAI(BaseAgent[NewsSummary]):
    """Enhanced news agent with full AI integration.

    This agent uses:
    - Ollama for summarization and analysis
    - RAG for semantic search and similar articles
    - MCP for standardized tool access
    - LangGraph for complex workflow orchestration
    """

    def __init__(self, **kwargs):
        """Initialize the AI-enhanced news agent."""
        super().__init__(name="NewsAgentAI", **kwargs)

        # Initialize AI services
        self.ollama = get_ollama_service()
        self.rag = get_rag_service()
        self.mcp = get_mcp_server()
        self.orchestrator = get_orchestrator()

        # Initialize connectors (for direct access if needed)
        self.hn_connector = HackerNewsConnector(max_stories=30)
        self.rss_connector = RSSConnector(max_articles_per_feed=10)

        logger.info("news_agent_ai_initialized")

    async def fetch_news_with_mcp(
        self,
        sources: List[str] = ["hackernews", "rss"],
        max_articles: int = 50
    ) -> List[NewsArticle]:
        """Fetch news using MCP tools."""
        all_articles = []

        try:
            # Fetch from HackerNews via MCP
            if "hackernews" in sources:
                response = await self.mcp.execute_tool(
                    "fetch_hackernews",
                    {"story_type": "top", "max_stories": 30}
                )

                if response.success and response.data:
                    # Convert dict data to NewsArticle objects
                    for article_data in response.data:
                        article = self._dict_to_article(article_data)
                        if article:
                            all_articles.append(article)

                    logger.info("fetched_hackernews_via_mcp", count=len(response.data))

            # Fetch from RSS via MCP
            if "rss" in sources:
                response = await self.mcp.execute_tool(
                    "fetch_rss_feeds",
                    {"max_articles": 20}
                )

                if response.success and response.data:
                    for article_data in response.data:
                        article = self._dict_to_article(article_data)
                        if article:
                            all_articles.append(article)

                    logger.info("fetched_rss_via_mcp", count=len(response.data))

            # Limit to max articles
            all_articles = all_articles[:max_articles]

            # Store in RAG for semantic search
            if all_articles:
                await self.rag.add_articles_batch(all_articles[:20])  # Store top 20
                logger.info("articles_stored_in_rag", count=min(20, len(all_articles)))

        except Exception as e:
            logger.error("mcp_fetch_failed", error=str(e))

        return all_articles

    def _dict_to_article(self, data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Convert dictionary data to NewsArticle."""
        try:
            # Parse published date
            published_at = None
            if data.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(data["published_at"])
                except:
                    published_at = datetime.now()

            # Handle priority
            priority = data.get("priority", "medium")
            if isinstance(priority, str):
                priority_value = priority
            else:
                priority_value = priority.value if hasattr(priority, 'value') else str(priority)

            return NewsArticle(
                title=data.get("title", "Untitled"),
                url=data.get("url", ""),
                source=DataSource.HACKERNEWS if "hackernews" in data.get("source", "").lower() else DataSource.RSS,
                source_name=data.get("source_name", "Unknown"),
                author=data.get("author"),
                published_at=published_at or datetime.now(),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                priority=priority_value,
                relevance_score=data.get("relevance_score", 0.5)
            )
        except Exception as e:
            logger.error("article_conversion_failed", error=str(e))
            return None

    async def generate_ai_summary(self, articles: List[NewsArticle], model: str = None) -> str:
        """Generate AI-powered summary using Ollama."""
        if not articles:
            return "No articles to summarize."

        try:
            # Check Ollama availability
            if not await self.ollama.check_availability():
                logger.warning("ollama_not_available")
                return "AI summarization unavailable - Ollama not running. Please run: ollama serve"

            # Prepare articles text with better formatting
            articles_text = []
            for i, article in enumerate(articles[:10], 1):  # Top 10 articles
                desc = article.description if article.description else "No description available"
                # Truncate long descriptions
                if len(desc) > 200:
                    desc = desc[:200] + "..."

                articles_text.append(
                    f"{i}. **{article.title}**\n"
                    f"   Source: {article.source_name}\n"
                    f"   {desc}"
                )

            prompt = f"""Analyze these top news articles and create a concise bullet-point summary:

{chr(10).join(articles_text)}

Create a summary in this format:
**Top Headlines:**
- [Most important headline with brief context]
- [Second most important headline with brief context]
- [Third headline if significant]

**Key Trends & Themes:**
- [Major trend or theme you identified]
- [Another significant trend]
- [Additional trend if present]

Keep each bullet point to 1-2 sentences max. Focus on what's most important and actionable for readers.

Summary:"""

            # Generate summary with the selected model or default
            response = await self.ollama.generate(
                prompt=prompt,
                model=model,  # Use the passed model if provided
                temperature=0.3,  # Lower temperature for factual summary
                max_tokens=500
            )

            # Ensure we have content
            if response and response.content:
                summary = response.content.strip()
                if summary:
                    return summary
                else:
                    logger.warning("empty_summary_generated")
                    return "AI generated an empty summary. Please try again."
            else:
                logger.warning("no_response_from_ollama")
                return "No response from Ollama. Please check if the service is running."

        except Exception as e:
            logger.error("ai_summary_generation_failed", error=str(e))
            return f"Summary generation failed: {str(e)}"

    async def extract_insights(self, articles: List[NewsArticle], model: str = None) -> List[str]:
        """Extract key insights using AI with meaningful explanations."""
        if not articles:
            return []

        try:
            # Check Ollama availability
            if not await self.ollama.check_availability():
                return ["AI insights unavailable - Ollama not running. Please run: ollama serve"]

            insights = []

            # 1. Generate topic analysis with explanation
            all_keywords = []
            for article in articles[:5]:  # Top 5 articles
                keywords = await self.ollama.extract_keywords(
                    f"{article.title} {article.description}",
                    max_keywords=5
                )
                all_keywords.extend(keywords)

            # Count keyword frequencies
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

            # Get top keywords as trends
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            if top_keywords:
                trends = [kw for kw, _ in top_keywords[:3]]
                insights.append(
                    f"🔥 **Trending Topics**: {', '.join(trends)} - "
                    f"These topics appear frequently across today's news, indicating major areas of focus"
                )

            # 2. Sentiment analysis with context
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for article in articles[:10]:
                sentiment_result = await self.ollama.analyze_sentiment(
                    f"{article.title} {article.description}"
                )
                # Extract sentiment from dict result
                if isinstance(sentiment_result, dict):
                    sentiment_type = sentiment_result.get("sentiment", "neutral")
                    if sentiment_type == "positive":
                        positive_count += 1
                    elif sentiment_type == "negative":
                        negative_count += 1
                    else:
                        neutral_count += 1
                else:
                    # Fallback for numeric result
                    if sentiment_result > 0.6:
                        positive_count += 1
                    elif sentiment_result < 0.4:
                        negative_count += 1
                    else:
                        neutral_count += 1

            total_analyzed = 10
            if total_analyzed > 0:
                positive_pct = (positive_count / total_analyzed) * 100
                negative_pct = (negative_count / total_analyzed) * 100

                if positive_pct > 60:
                    insights.append(
                        f"😊 **Positive Outlook**: {positive_pct:.0f}% positive news - "
                        f"Today's news cycle is predominantly optimistic, suggesting favorable developments"
                    )
                elif negative_pct > 50:
                    insights.append(
                        f"⚠️ **Challenging Times**: {negative_pct:.0f}% negative news - "
                        f"Current events show significant challenges requiring attention"
                    )
                else:
                    insights.append(
                        f"⚖️ **Balanced Coverage**: {positive_pct:.0f}% positive, {negative_pct:.0f}% negative - "
                        f"Mixed news sentiment reflects varied developments across different sectors"
                    )

            # 3. Source diversity analysis
            sources = set(a.source_name for a in articles)
            if len(sources) > 5:
                insights.append(
                    f"📰 **Diverse Perspectives**: {len(sources)} different sources - "
                    f"Wide coverage ensures multiple viewpoints and comprehensive news understanding"
                )
            elif len(sources) > 2:
                insights.append(
                    f"📊 **Moderate Coverage**: {len(sources)} sources reporting - "
                    f"Consider checking additional sources for broader perspective"
                )
            else:
                insights.append(
                    f"🎯 **Limited Sources**: Only {len(sources)} sources - "
                    f"Consider diversifying news sources for more balanced coverage"
                )

            # 4. Freshness and timing insights
            if articles:
                articles_with_dates = [a for a in articles if a.published_at]
                if articles_with_dates:
                    latest = max(a.published_at for a in articles_with_dates)
                    hours_ago = (datetime.now() - latest).total_seconds() / 3600

                    if hours_ago < 1:
                        insights.append(
                            "⚡ **Breaking News**: Updates from less than 1 hour ago - "
                            "You're seeing the latest developments as they happen"
                        )
                    elif hours_ago < 6:
                        insights.append(
                            f"🕐 **Recent Updates**: Latest story from {hours_ago:.1f} hours ago - "
                            f"News is current and relevant to today's events"
                        )
                    else:
                        insights.append(
                            f"📅 **Time to Refresh**: Latest story is {hours_ago:.0f} hours old - "
                            f"Consider refreshing for more recent updates"
                        )

            # 5. Add an actionable insight
            if len(articles) > 20:
                insights.append(
                    "💡 **Action Item**: With abundant news available, use the search feature "
                    "to focus on topics most relevant to your interests"
                )

            return insights if insights else ["No insights could be generated. Check if articles have content."]

        except Exception as e:
            logger.error("insights_extraction_failed", error=str(e))
            return [f"Insight extraction failed: {str(e)}"]

    async def find_similar_articles(
        self,
        query: str,
        max_results: int = 5
    ) -> List[NewsArticle]:
        """Find similar articles using RAG semantic search."""
        try:
            # Search in RAG
            search_results = await self.rag.search_articles(
                query=query,
                max_results=max_results
            )

            # Convert search results back to articles
            similar_articles = []
            for result in search_results:
                # Create article from metadata
                metadata = result.metadata
                article = NewsArticle(
                    title=metadata.get("title", "Untitled"),
                    url=metadata.get("url", ""),
                    source=metadata.get("source", DataSource.CUSTOM),
                    source_name=metadata.get("source_name", "Unknown"),
                    author=metadata.get("author"),
                    published_at=datetime.fromisoformat(metadata.get("published_at")) if metadata.get("published_at") else datetime.now(),
                    description=result.content[:200] if result.content else "",
                    tags=eval(metadata.get("tags", "[]")) if metadata.get("tags") else [],
                    priority=metadata.get("priority", "medium"),
                    relevance_score=result.similarity
                )
                similar_articles.append(article)

            return similar_articles

        except Exception as e:
            logger.error("similar_articles_search_failed", error=str(e))
            return []

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the news using RAG."""
        try:
            result = await self.rag.answer_with_context(
                question=question,
                system_prompt="You are a news analyst. Answer questions based on recent news articles."
            )
            return result
        except Exception as e:
            logger.error("question_answering_failed", error=str(e))
            return {
                "answer": f"Unable to answer: {str(e)}",
                "sources": [],
                "context_used": False
            }

    async def run_orchestrated_workflow(
        self,
        user_request: str = "Give me a summary of today's tech news",
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the full LangGraph orchestrated workflow."""
        try:
            result = await self.orchestrator.run(
                user_request=user_request,
                preferences=preferences or {
                    "sources": ["hackernews", "rss"],
                    "topics": ["technology", "programming", "AI"]
                }
            )
            return result
        except Exception as e:
            logger.error("orchestrated_workflow_failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute(self) -> NewsSummary:
        """Execute the news agent workflow with AI enhancements."""
        try:
            # Fetch news using MCP
            articles = await self.fetch_news_with_mcp()

            if not articles:
                logger.warning("no_articles_fetched")
                return NewsSummary(
                    date=datetime.now(),
                    source=DataSource.CUSTOM,
                    total_articles=0,
                    top_articles=[],
                    summary_text="No articles available"
                )

            # Sort by importance
            articles.sort(key=lambda a: a.calculate_importance(), reverse=True)

            # Generate AI summary
            summary_text = await self.generate_ai_summary(articles)

            # Extract insights
            insights = await self.extract_insights(articles)

            # Get trending topics via MCP
            trending_response = await self.mcp.execute_tool(
                "get_trending_topics",
                {"limit": 10}
            )

            trending_topics = []
            if trending_response.success and trending_response.data:
                trending_topics = trending_response.data

            # Create summary
            summary = NewsSummary(
                date=datetime.now(),
                source=DataSource.CUSTOM,
                total_articles=len(articles),
                top_articles=articles[:10],
                summary_text=summary_text,
                key_topics=trending_topics,
                insights=insights,
                recommendations=[
                    "Check back in a few hours for updates",
                    "Explore articles in your areas of interest",
                    "Use semantic search to find related content"
                ]
            )

            # Add all articles to summary
            for article in articles:
                summary.add_article(article)

            logger.info("news_summary_generated",
                       total_articles=len(articles),
                       insights_count=len(insights))

            return summary

        except Exception as e:
            logger.error("news_agent_execution_failed", error=str(e))
            raise

    async def run(self) -> List[NewsArticle]:
        """Run the agent and return articles."""
        try:
            summary = await self._execute()
            return summary.articles if summary else []
        except Exception as e:
            logger.error("agent_run_failed", error=str(e))
            return []

    async def generate_summary(self, articles: List[NewsArticle]) -> NewsSummary:
        """Generate an AI-enhanced summary from articles."""
        # Generate AI summary
        summary_text = await self.generate_ai_summary(articles)

        # Extract insights
        insights = await self.extract_insights(articles)

        return NewsSummary(
            title="AI-Enhanced News Summary",
            summary=summary_text,
            articles=articles[:10],
            total_articles=len(articles),
            sources=list(set(a.source_name for a in articles)),
            key_topics=[],
            insights=insights,
            recommendations=[],
            generated_at=datetime.now()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics including AI service stats."""
        stats = super().get_statistics()

        # Add AI service statistics
        stats["ai_services"] = {
            "ollama": self.ollama.get_statistics(),
            "rag": self.rag.get_statistics(),
            "mcp": self.mcp.get_statistics(),
            "orchestrator": self.orchestrator.get_statistics()
        }

        return stats


# Factory function to get AI-enhanced news agent
def get_news_agent_ai(**kwargs) -> NewsAgentAI:
    """Get or create AI-enhanced news agent instance."""
    return NewsAgentAI(**kwargs)
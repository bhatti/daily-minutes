"""
LangGraph Orchestrator for Daily Minutes Workflow

This module implements the main orchestration logic using LangGraph,
coordinating between MCP tools, Ollama LLM, and RAG system.
"""

import asyncio
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.services.ollama_service import get_ollama_service
from src.services.mcp_server import get_mcp_server
from src.services.rag_service import get_rag_service
from src.models.news import NewsArticle, NewsSummary

logger = get_logger(__name__)


class WorkflowStep(str, Enum):
    """Steps in the daily minutes workflow."""

    ANALYZE_REQUEST = "analyze_request"
    FETCH_DATA = "fetch_data"
    SEARCH_CONTEXT = "search_context"
    SUMMARIZE = "summarize"
    GENERATE_INSIGHTS = "generate_insights"
    FORMAT_OUTPUT = "format_output"
    COMPLETE = "complete"


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""

    # Input
    user_request: str
    preferences: Optional[Dict[str, Any]]

    # Processing
    current_step: WorkflowStep
    data_sources: List[str]
    fetched_data: Dict[str, List[Any]]
    search_results: List[Dict[str, Any]]
    context: str

    # Output
    summary: Optional[str]
    insights: List[str]
    recommendations: List[str]
    formatted_output: Optional[str]

    # Metadata
    error: Optional[str]
    execution_time_ms: int
    steps_completed: List[str]


class LangGraphOrchestrator:
    """
    Main orchestrator using LangGraph for workflow management.

    Coordinates between:
    - MCP tools for data fetching
    - Ollama for LLM operations
    - RAG for context retrieval
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.ollama = get_ollama_service()
        self.mcp_server = get_mcp_server()
        self.rag_service = get_rag_service()

        # Build the workflow graph
        self.workflow = self._build_workflow()

        logger.info("langgraph_orchestrator_initialized")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""

        workflow = StateGraph(WorkflowState)

        # Add nodes for each step
        workflow.add_node("analyze", self._analyze_request)
        workflow.add_node("fetch", self._fetch_data)
        workflow.add_node("search", self._search_context)
        workflow.add_node("summarize", self._summarize_data)
        workflow.add_node("insights", self._generate_insights)
        workflow.add_node("format", self._format_output)

        # Set entry point
        workflow.set_entry_point("analyze")

        # Add edges
        workflow.add_edge("analyze", "fetch")
        workflow.add_edge("fetch", "search")
        workflow.add_edge("search", "summarize")
        workflow.add_edge("summarize", "insights")
        workflow.add_edge("insights", "format")
        workflow.add_edge("format", END)

        return workflow.compile()

    async def _analyze_request(self, state: WorkflowState) -> WorkflowState:
        """Analyze user request to determine data sources needed."""

        logger.info("analyzing_request", request=state["user_request"][:100])

        # Use LLM to analyze the request
        prompt = f"""Analyze the following request and determine which data sources are needed:

Request: {state["user_request"]}

Available data sources:
- hackernews: Tech news from HackerNews
- rss: RSS feeds from various sources
- weather: Weather information (future)
- calendar: Calendar events (future)
- email: Email summaries (future)

Return a JSON list of data sources needed for this request.
Example: ["hackernews", "rss"]

Data sources needed:"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=100
            )

            # Parse response
            content = response.content.strip()
            if "[" in content and "]" in content:
                import json
                start = content.find("[")
                end = content.rfind("]") + 1
                sources = json.loads(content[start:end])
            else:
                # Default to news sources
                sources = ["hackernews", "rss"]

            state["data_sources"] = sources
            state["current_step"] = WorkflowStep.FETCH_DATA
            state["steps_completed"].append("analyze")

            logger.info("request_analyzed", sources=sources)

        except Exception as e:
            logger.error("request_analysis_failed", error=str(e))
            state["data_sources"] = ["hackernews"]  # Fallback
            state["error"] = str(e)

        return state

    async def _fetch_data(self, state: WorkflowState) -> WorkflowState:
        """Fetch data from identified sources using MCP tools."""

        logger.info("fetching_data", sources=state["data_sources"])

        fetched_data = {}

        for source in state["data_sources"]:
            try:
                if source == "hackernews":
                    # Fetch HackerNews articles
                    response = await self.mcp_server.execute_tool(
                        "fetch_hackernews",
                        {"story_type": "top", "max_stories": 10}
                    )
                    if response.success:
                        fetched_data["hackernews"] = response.data

                elif source == "rss":
                    # Fetch RSS articles
                    response = await self.mcp_server.execute_tool(
                        "fetch_rss_feeds",
                        {"max_articles": 10}
                    )
                    if response.success:
                        fetched_data["rss"] = response.data

                # Add more sources as they become available

            except Exception as e:
                logger.error("fetch_failed", source=source, error=str(e))

        state["fetched_data"] = fetched_data
        state["current_step"] = WorkflowStep.SEARCH_CONTEXT
        state["steps_completed"].append("fetch")

        # Store articles in RAG for future search
        await self._store_articles_in_rag(fetched_data)

        logger.info("data_fetched",
                   sources=list(fetched_data.keys()),
                   total_items=sum(len(v) for v in fetched_data.values()))

        return state

    async def _search_context(self, state: WorkflowState) -> WorkflowState:
        """Search for relevant context using RAG."""

        logger.info("searching_context")

        # Build search query from user request
        query = state["user_request"]

        # Search for relevant articles
        search_results = await self.rag_service.search_articles(
            query=query,
            max_results=5
        )

        # Generate context
        context, _ = await self.rag_service.generate_context(
            query=query,
            max_documents=3
        )

        state["search_results"] = [
            {
                "title": r.metadata.get("title"),
                "url": r.metadata.get("url"),
                "similarity": r.similarity
            }
            for r in search_results
        ]
        state["context"] = context
        state["current_step"] = WorkflowStep.SUMMARIZE
        state["steps_completed"].append("search")

        logger.info("context_found", results=len(search_results))

        return state

    async def _summarize_data(self, state: WorkflowState) -> WorkflowState:
        """Summarize the fetched data."""

        logger.info("summarizing_data")

        # Combine all fetched articles
        all_articles = []
        for source_data in state["fetched_data"].values():
            if isinstance(source_data, list):
                all_articles.extend(source_data[:5])  # Limit per source

        if not all_articles:
            state["summary"] = "No data available to summarize."
            state["current_step"] = WorkflowStep.GENERATE_INSIGHTS
            state["steps_completed"].append("summarize")
            return state

        # Build summary prompt
        articles_text = "\n".join([
            f"- {a.get('title', 'Untitled')} ({a.get('source_name', 'Unknown')})"
            for a in all_articles[:10]
        ])

        prompt = f"""Summarize the following news articles for a daily briefing:

{articles_text}

Provide a concise summary highlighting the most important news and trends.
Maximum 200 words."""

        response = await self.ollama.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300
        )

        state["summary"] = response.content
        state["current_step"] = WorkflowStep.GENERATE_INSIGHTS
        state["steps_completed"].append("summarize")

        logger.info("summary_generated", length=len(response.content))

        return state

    async def _generate_insights(self, state: WorkflowState) -> WorkflowState:
        """Generate insights from the data."""

        logger.info("generating_insights")

        # Use context and summary to generate insights
        prompt = f"""Based on the following summary and context, generate 3-5 key insights:

Summary: {state.get('summary', 'No summary available')}

Context: {state.get('context', 'No context available')[:1000]}

Generate actionable insights that would be valuable for someone staying informed about technology and business.

Insights:"""

        response = await self.ollama.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=300
        )

        # Parse insights
        insights_text = response.content
        insights = [
            line.strip().lstrip("- •").strip()
            for line in insights_text.split("\n")
            if line.strip() and not line.strip().startswith("Insights")
        ]

        state["insights"] = insights[:5]

        # Generate recommendations
        rec_prompt = f"""Based on these insights, suggest 2-3 recommendations for actions or areas to explore:

Insights:
{chr(10).join(insights[:3])}

Recommendations:"""

        rec_response = await self.ollama.generate(
            prompt=rec_prompt,
            temperature=0.5,
            max_tokens=200
        )

        recommendations = [
            line.strip().lstrip("- •").strip()
            for line in rec_response.content.split("\n")
            if line.strip() and not line.strip().startswith("Recommendations")
        ]

        state["recommendations"] = recommendations[:3]
        state["current_step"] = WorkflowStep.FORMAT_OUTPUT
        state["steps_completed"].append("insights")

        logger.info("insights_generated",
                   insights_count=len(state["insights"]),
                   recommendations_count=len(state["recommendations"]))

        return state

    async def _format_output(self, state: WorkflowState) -> WorkflowState:
        """Format the final output."""

        logger.info("formatting_output")

        # Build formatted output
        output_parts = []

        # Add header
        output_parts.append("# Daily Minutes Report")
        output_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        output_parts.append("")

        # Add summary
        if state.get("summary"):
            output_parts.append("## Summary")
            output_parts.append(state["summary"])
            output_parts.append("")

        # Add insights
        if state.get("insights"):
            output_parts.append("## Key Insights")
            for insight in state["insights"]:
                output_parts.append(f"• {insight}")
            output_parts.append("")

        # Add recommendations
        if state.get("recommendations"):
            output_parts.append("## Recommendations")
            for rec in state["recommendations"]:
                output_parts.append(f"• {rec}")
            output_parts.append("")

        # Add sources
        if state.get("search_results"):
            output_parts.append("## Related Articles")
            for result in state["search_results"][:5]:
                output_parts.append(f"• [{result['title']}]({result.get('url', '#')})")
            output_parts.append("")

        # Add data sources
        output_parts.append("## Data Sources")
        for source in state.get("data_sources", []):
            count = len(state.get("fetched_data", {}).get(source, []))
            output_parts.append(f"• {source.title()}: {count} items")

        state["formatted_output"] = "\n".join(output_parts)
        state["current_step"] = WorkflowStep.COMPLETE
        state["steps_completed"].append("format")

        logger.info("output_formatted", length=len(state["formatted_output"]))

        return state

    async def _store_articles_in_rag(self, fetched_data: Dict[str, List[Any]]):
        """Store fetched articles in RAG for future search."""

        for source, articles in fetched_data.items():
            if not articles:
                continue

            # Convert to NewsArticle format if needed
            news_articles = []
            for article_data in articles[:20]:  # Limit storage
                if isinstance(article_data, dict):
                    # Create NewsArticle from dict
                    try:
                        from src.models.news import Priority, DataSource

                        # Parse published date
                        published_at = None
                        if article_data.get("published_at"):
                            from datetime import datetime
                            try:
                                published_at = datetime.fromisoformat(article_data["published_at"])
                            except:
                                published_at = datetime.now()

                        news_article = NewsArticle(
                            title=article_data.get("title", "Untitled"),
                            url=article_data.get("url", f"http://example.com/{source}"),
                            source=DataSource.HACKERNEWS if source == "hackernews" else DataSource.RSS,
                            source_name=article_data.get("source_name", source),
                            author=article_data.get("author"),
                            published_at=published_at or datetime.now(),
                            description=article_data.get("description", ""),
                            tags=article_data.get("tags", []),
                            priority=article_data.get("priority", "medium"),
                            relevance_score=article_data.get("relevance_score", 0.5)
                        )
                        news_articles.append(news_article)
                    except Exception as e:
                        logger.error("article_conversion_failed", error=str(e))

            # Store in RAG
            if news_articles:
                try:
                    await self.rag_service.add_articles_batch(news_articles)
                    logger.info("articles_stored_in_rag",
                               source=source,
                               count=len(news_articles))
                except Exception as e:
                    logger.error("rag_storage_failed",
                                source=source,
                                error=str(e))

    async def run_orchestrated_workflow(
        self,
        user_request: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow for a user request.

        This is an alias for the run() method for compatibility with UI code.

        Args:
            user_request: The user's request or query
            preferences: Optional user preferences

        Returns:
            Dictionary with the workflow results
        """
        return await self.run(user_request, preferences)

    async def run(
        self,
        user_request: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow for a user request.

        Args:
            user_request: The user's request or query
            preferences: Optional user preferences

        Returns:
            Dictionary with the workflow results
        """

        start_time = datetime.now()

        # Initialize state
        initial_state: WorkflowState = {
            "user_request": user_request,
            "preferences": preferences,
            "current_step": WorkflowStep.ANALYZE_REQUEST,
            "data_sources": [],
            "fetched_data": {},
            "search_results": [],
            "context": "",
            "summary": None,
            "insights": [],
            "recommendations": [],
            "formatted_output": None,
            "error": None,
            "execution_time_ms": 0,
            "steps_completed": []
        }

        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            final_state["execution_time_ms"] = int(execution_time)

            logger.info("workflow_completed",
                       steps=final_state["steps_completed"],
                       time_ms=execution_time)

            return {
                "success": True,
                "output": final_state["formatted_output"],
                "summary": final_state["summary"],
                "insights": final_state["insights"],
                "recommendations": final_state["recommendations"],
                "sources": final_state["search_results"],
                "execution_time_ms": final_state["execution_time_ms"],
                "steps_completed": final_state["steps_completed"]
            }

        except Exception as e:
            logger.error("workflow_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "services": {
                "ollama": self.ollama.get_statistics(),
                "mcp": self.mcp_server.get_statistics(),
                "rag": self.rag_service.get_statistics()
            }
        }


# Singleton instance
_orchestrator = None

def get_orchestrator() -> LangGraphOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator()
    return _orchestrator
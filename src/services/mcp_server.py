"""
MCP (Model Context Protocol) Server Implementation

This module wraps our connectors as MCP servers, making them available
as tools for LLMs to use in a standardized way.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.connectors.weather import get_weather_service
from src.connectors.email.gmail_connector import GmailConnector
from src.connectors.calendar.google_calendar_connector import GoogleCalendarConnector
from src.core.logging import get_logger
from src.core.models import EmailMessage, CalendarEvent
from src.models.news import NewsArticle

logger = get_logger(__name__)


class ToolType(str, Enum):
    """Types of tools available in MCP."""

    FETCH_NEWS = "fetch_news"
    SEARCH_NEWS = "search_news"
    GET_TRENDING = "get_trending"
    FETCH_RSS = "fetch_rss"
    GET_WEATHER = "get_weather"
    FETCH_EMAILS = "fetch_emails"
    FETCH_CALENDAR = "fetch_calendar"


class MCPTool(BaseModel):
    """MCP Tool definition."""

    name: str = Field(..., description="Tool name")
    type: ToolType = Field(..., description="Tool type")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    required_params: List[str] = Field(default_factory=list, description="Required parameters")

    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP protocol format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }


class MCPResponse(BaseModel):
    """Standard MCP response format."""

    success: bool = Field(..., description="Whether the operation succeeded")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class MCPServer:
    """
    MCP Server that exposes our connectors as tools.

    This allows LLMs to discover and use our data sources
    through a standardized interface.
    """

    def __init__(self):
        """Initialize MCP server with available tools."""
        self.tools: Dict[str, MCPTool] = {}
        self.connectors: Dict[str, Any] = {}

        # Initialize connectors
        self._initialize_connectors()

        # Register tools
        self._register_tools()

        logger.info("mcp_server_initialized", tools_count=len(self.tools))

    def _init_gmail_connector(self) -> Optional[GmailConnector]:
        """Initialize Gmail connector with credentials.

        Returns:
            GmailConnector if credentials exist and authentication succeeds, None otherwise
        """
        try:
            connector = GmailConnector(user_id="me")

            # Check if credentials exist
            creds_path = "./credentials/gmail_token.json"
            client_secrets_path = "./credentials/gmail_client_secret.json"

            if not os.path.exists(client_secrets_path):
                logger.info("gmail_credentials_not_found",
                           path=client_secrets_path,
                           message="Gmail connector will not be available. Add credentials to enable.")
                return None

            # Authenticate
            connector.authenticate(
                credentials_path=creds_path,
                client_secrets_path=client_secrets_path
            )

            if not connector.is_authenticated:
                logger.warning("gmail_authentication_failed")
                return None

            logger.info("gmail_connector_initialized", user_id="me")
            return connector

        except Exception as e:
            logger.error("gmail_connector_init_error", error=str(e))
            return None

    def _init_google_calendar_connector(self) -> Optional[GoogleCalendarConnector]:
        """Initialize Google Calendar connector with credentials.

        Returns:
            GoogleCalendarConnector if credentials exist and authentication succeeds, None otherwise
        """
        try:
            creds_file = "./credentials/google_calendar_credentials.json"

            if not os.path.exists(creds_file):
                logger.info("google_calendar_credentials_not_found",
                           path=creds_file,
                           message="Google Calendar connector will not be available. Add credentials to enable.")
                return None

            connector = GoogleCalendarConnector(
                credentials_file=creds_file,
                token_file="./credentials/google_calendar_token.pickle"
            )

            # Authenticate
            connector.authenticate()

            if not connector.is_authenticated:
                logger.warning("google_calendar_authentication_failed")
                return None

            logger.info("google_calendar_connector_initialized")
            return connector

        except Exception as e:
            logger.error("google_calendar_connector_init_error", error=str(e))
            return None

    def _initialize_connectors(self):
        """Initialize data connectors."""
        # News connectors (always real)
        self.connectors["hackernews"] = HackerNewsConnector(max_stories=10)
        self.connectors["rss"] = RSSConnector(max_articles_per_feed=5)

        # Weather connector (always real)
        self.connectors["weather"] = get_weather_service()

        # Email connectors (real - OAuth required, graceful fallback if missing)
        self.connectors["gmail"] = self._init_gmail_connector()

        # Calendar connectors (real - OAuth required, graceful fallback if missing)
        self.connectors["google_calendar"] = self._init_google_calendar_connector()

    def _register_tools(self):
        """Register available tools."""

        # HackerNews tools
        self.register_tool(MCPTool(
            name="fetch_hackernews",
            type=ToolType.FETCH_NEWS,
            description="Fetch top stories from HackerNews",
            parameters={
                "story_type": {
                    "type": "string",
                    "enum": ["top", "new", "best", "ask", "show", "job"],
                    "description": "Type of stories to fetch",
                    "default": "top"
                },
                "max_stories": {
                    "type": "integer",
                    "description": "Maximum number of stories to fetch",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                },
                "min_score": {
                    "type": "integer",
                    "description": "Minimum score filter",
                    "default": 0
                }
            },
            required_params=[]
        ))

        self.register_tool(MCPTool(
            name="search_hackernews",
            type=ToolType.SEARCH_NEWS,
            description="Search HackerNews stories by query",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10
                }
            },
            required_params=["query"]
        ))

        self.register_tool(MCPTool(
            name="get_trending_topics",
            type=ToolType.GET_TRENDING,
            description="Get trending topics from HackerNews",
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Number of topics",
                    "default": 10
                }
            },
            required_params=[]
        ))

        # RSS tools
        self.register_tool(MCPTool(
            name="fetch_rss_feeds",
            type=ToolType.FETCH_RSS,
            description="Fetch articles from RSS feeds",
            parameters={
                "feed_url": {
                    "type": "string",
                    "description": "Specific RSS feed URL (optional)"
                },
                "category": {
                    "type": "string",
                    "description": "Feed category filter",
                    "enum": ["technology", "ai", "programming", "news"]
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Maximum articles per feed",
                    "default": 10
                }
            },
            required_params=[]
        ))

        # Weather tools
        self.register_tool(MCPTool(
            name="get_current_weather",
            type=ToolType.GET_WEATHER,
            description="Get current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates (lat,lon)"
                }
            },
            required_params=["location"]
        ))

        self.register_tool(MCPTool(
            name="get_weather_forecast",
            type=ToolType.GET_WEATHER,
            description="Get weather forecast for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates (lat,lon)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to forecast",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 7
                }
            },
            required_params=["location"]
        ))

        # Email tools (register only if Gmail connector is available)
        if self.connectors.get("gmail"):
            self.register_tool(MCPTool(
                name="fetch_emails",
                type=ToolType.FETCH_EMAILS,
                description="Fetch recent emails from Gmail with action items and importance scores",
                parameters={
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to fetch",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500
                    },
                    "query": {
                        "type": "string",
                        "description": "Gmail search query (e.g., 'is:unread', 'from:boss@company.com')",
                        "default": "is:unread"
                    }
                },
                required_params=[]
            ))
            logger.info("email_tool_registered", tool="fetch_emails")

        # Calendar tools (register only if Google Calendar connector is available)
        if self.connectors.get("google_calendar"):
            self.register_tool(MCPTool(
                name="fetch_calendar_events",
                type=ToolType.FETCH_CALENDAR,
                description="Fetch upcoming calendar events with preparation requirements",
                parameters={
                    "days_ahead": {
                        "type": "integer",
                        "description": "Number of days ahead to fetch events",
                        "default": 7,
                        "minimum": 1,
                        "maximum": 30
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to fetch",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 250
                    }
                },
                required_params=[]
            ))
            logger.info("calendar_tool_registered", tool="fetch_calendar_events")

    def register_tool(self, tool: MCPTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool_name=tool.name, tool_type=tool.type)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools in MCP format."""
        return [tool.to_mcp_format() for tool in self.tools.values()]

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPResponse:
        """Execute a tool with given parameters."""

        if tool_name not in self.tools:
            return MCPResponse(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        tool = self.tools[tool_name]

        # Validate required parameters
        missing_params = [p for p in tool.required_params if p not in parameters]
        if missing_params:
            return MCPResponse(
                success=False,
                error=f"Missing required parameters: {missing_params}"
            )

        try:
            # Route to appropriate handler
            if tool_name == "fetch_hackernews":
                result = await self._fetch_hackernews(parameters)
            elif tool_name == "search_hackernews":
                result = await self._search_hackernews(parameters)
            elif tool_name == "get_trending_topics":
                result = await self._get_trending_topics(parameters)
            elif tool_name == "fetch_rss_feeds":
                result = await self._fetch_rss_feeds(parameters)
            elif tool_name == "get_current_weather":
                result = await self._get_current_weather(parameters)
            elif tool_name == "get_weather_forecast":
                result = await self._get_weather_forecast(parameters)
            elif tool_name == "fetch_emails":
                result = await self._fetch_emails(parameters)
            elif tool_name == "fetch_calendar_events":
                result = await self._fetch_calendar_events(parameters)
            else:
                return MCPResponse(
                    success=False,
                    error=f"Handler not implemented for tool '{tool_name}'"
                )

            return MCPResponse(
                success=True,
                data=result,
                metadata={
                    "tool": tool_name,
                    "parameters": parameters,
                    "execution_time_ms": 0  # TODO: Add timing
                }
            )

        except Exception as e:
            logger.error("tool_execution_failed",
                        tool=tool_name,
                        error=str(e))
            return MCPResponse(
                success=False,
                error=str(e)
            )

    async def _fetch_hackernews(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch HackerNews stories."""
        connector = self.connectors["hackernews"]

        # Update connector settings
        connector.story_type = params.get("story_type", "top")
        connector.max_stories = params.get("max_stories", 10)
        connector.min_score = params.get("min_score", 0)

        # Fetch articles
        articles = await connector.execute_async()

        # Convert to dictionary format for JSON serialization
        return [self._article_to_dict(article) for article in articles]

    async def _search_hackernews(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search HackerNews stories."""
        connector = self.connectors["hackernews"]

        query = params["query"]
        limit = params.get("limit", 10)

        articles = await connector.search_stories(query, limit)

        return [self._article_to_dict(article) for article in articles]

    async def _get_trending_topics(self, params: Dict[str, Any]) -> List[str]:
        """Get trending topics from HackerNews."""
        connector = self.connectors["hackernews"]

        limit = params.get("limit", 10)

        # First fetch some articles to analyze
        connector.max_stories = 50  # Analyze more stories for trends
        await connector.execute_async()

        topics = await connector.get_trending_topics()

        return topics[:limit]

    async def _fetch_rss_feeds(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch RSS feed articles."""
        connector = self.connectors["rss"]

        feed_url = params.get("feed_url")
        category = params.get("category")
        max_articles = params.get("max_articles", 10)

        connector.max_articles_per_feed = max_articles

        if feed_url:
            # Fetch from specific feed
            articles = await connector.fetch_from_source(feed_url)
        else:
            # Fetch from all configured feeds
            articles = await connector.fetch_all_feeds()

            # Filter by category if specified
            if category:
                articles = [a for a in articles
                          if category in a.tags or
                          (hasattr(a, 'metadata') and
                           a.metadata.get('feed_category') == category)]

        return [self._article_to_dict(article) for article in articles]

    async def _get_current_weather(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current weather for a location."""
        service = self.connectors["weather"]
        location = params["location"]

        weather = await service.get_current_weather(location)

        if weather:
            return {
                "location": weather.location,
                "temperature": weather.temperature,
                "feels_like": weather.feels_like,
                "description": weather.description,
                "humidity": weather.humidity,
                "pressure": weather.pressure,
                "wind_speed": weather.wind_speed,
                "wind_direction": weather.wind_direction,
                "icon": weather.icon,
                "timestamp": weather.timestamp.isoformat(),
                "sunrise": weather.sunrise.isoformat() if weather.sunrise else None,
                "sunset": weather.sunset.isoformat() if weather.sunset else None,
                "visibility": weather.visibility,
                "uv_index": weather.uv_index
            }
        else:
            return None

    async def _get_weather_forecast(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get weather forecast for a location."""
        service = self.connectors["weather"]
        location = params["location"]
        days = params.get("days", 5)

        forecast = await service.get_forecast(location, days)

        if forecast:
            return [
                {
                    "location": w.location,
                    "date": w.timestamp.isoformat(),
                    "temperature": w.temperature,
                    "feels_like": w.feels_like,
                    "description": w.description,
                    "humidity": w.humidity,
                    "pressure": w.pressure,
                    "wind_speed": w.wind_speed,
                    "wind_direction": w.wind_direction,
                    "icon": w.icon,
                    "forecast_details": w.forecast
                }
                for w in forecast
            ]
        else:
            return []

    async def _fetch_emails(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail using real Gmail API."""
        connector = self.connectors.get("gmail")
        if not connector:
            raise Exception(
                "Gmail connector not available. "
                "Please configure Gmail credentials in ./credentials/gmail_client_secret.json"
            )

        max_results = params.get("max_results", 50)
        query = params.get("query", "is:unread")

        # Fetch emails using real Gmail API
        emails = await connector.fetch_unread_emails(max_results=max_results)

        logger.info("emails_fetched", count=len(emails), source="gmail")

        # Convert to dictionary format for JSON serialization
        return [self._email_to_dict(email) for email in emails]

    async def _fetch_calendar_events(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch calendar events from Google Calendar using real API."""
        connector = self.connectors.get("google_calendar")
        if not connector:
            raise Exception(
                "Google Calendar connector not available. "
                "Please configure Google Calendar credentials in ./credentials/google_calendar_credentials.json"
            )

        days_ahead = params.get("days_ahead", 7)
        max_results = params.get("max_results", 50)

        # Calculate time range
        time_min = datetime.now()
        time_max = time_min + timedelta(days=days_ahead)

        # Fetch events using real Google Calendar API
        events = await connector.fetch_events(
            time_min=time_min,
            time_max=time_max,
            max_results=max_results
        )

        logger.info("calendar_events_fetched", count=len(events), source="google_calendar")

        # Convert to dictionary format for JSON serialization
        return [self._calendar_event_to_dict(event) for event in events]

    def _email_to_dict(self, email: EmailMessage) -> Dict[str, Any]:
        """Convert EmailMessage to dictionary for JSON serialization."""
        return {
            "id": email.id,
            "subject": email.subject,
            "sender": email.sender,
            "body": email.body,
            "received_at": email.received_at.isoformat(),
            "importance_score": email.importance_score,
            "has_action_items": email.has_action_items,
            "action_items": email.action_items,
            "is_read": email.is_read
        }

    def _calendar_event_to_dict(self, event: CalendarEvent) -> Dict[str, Any]:
        """Convert CalendarEvent to dictionary for JSON serialization."""
        return {
            "id": event.id,
            "summary": event.summary,
            "description": event.description,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat(),
            "location": event.location,
            "attendees": event.attendees,
            "importance_score": event.importance_score,
            "requires_preparation": event.requires_preparation,
            "preparation_notes": event.preparation_notes,
            "is_focus_time": event.is_focus_time
        }

    def _article_to_dict(self, article: NewsArticle) -> Dict[str, Any]:
        """Convert NewsArticle to dictionary for JSON serialization."""
        priority = article.priority
        if isinstance(priority, str):
            priority_str = priority
        else:
            priority_str = priority.value if hasattr(priority, 'value') else str(priority)

        return {
            "title": article.title,
            "url": article.url,
            "source": str(article.source),
            "source_name": article.source_name,
            "author": article.author,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "description": article.description,
            "tags": article.tags,
            "priority": priority_str,
            "relevance_score": article.relevance_score,
            "sentiment_score": article.sentiment_score,
            "importance": article.calculate_importance(),
            "metadata": article.metadata
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP protocol request.

        This is the main entry point for MCP clients.
        """

        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            # List available tools
            tools = await self.list_tools()
            return {
                "tools": tools
            }

        elif method == "tools/call":
            # Execute a tool
            tool_name = params.get("name")
            tool_params = params.get("arguments", {})

            response = await self.execute_tool(tool_name, tool_params)

            return {
                "success": response.success,
                "result": response.data if response.success else None,
                "error": response.error if not response.success else None,
                "metadata": response.metadata
            }

        else:
            return {
                "error": f"Unknown method: {method}"
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "tools_count": len(self.tools),
            "tools": list(self.tools.keys()),
            "connectors": list(self.connectors.keys()),
            "connector_stats": {
                name: connector.get_statistics() if hasattr(connector, 'get_statistics') else {}
                for name, connector in self.connectors.items()
            }
        }


# Singleton instance
_mcp_server = None

def get_mcp_server() -> MCPServer:
    """Get or create MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server
"""Unit tests for ReAct Agent

Tests the ReAct agent's reasoning, acting, and observation loop using mocks.
Follows TDD principles with comprehensive test coverage.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.react_agent import (
    ReActAgent,
    ReActStep,
    DailyBrief,
)
from src.agents.base_agent import AgentConfig
from tests.load_mock_data import load_mock_emails
from tests.mock_calendar_data import generate_mock_calendar_events


class TestReActAgentInitialization:
    """Test ReAct agent initialization."""

    def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default configuration."""
        agent = ReActAgent()

        assert agent.name == "ReActAgent"
        assert agent.max_steps == 15
        assert agent.temperature == 0.7
        assert agent.model_name == "llama3.2:3b"
        assert agent.mcp_server is not None
        assert agent.ollama_service is not None
        assert agent.steps == []
        assert agent.collected_data == {
            "emails": [],
            "calendar_events": [],
            "news": [],
            "weather": None,
        }

    def test_agent_initialization_with_custom_config(self):
        """Test agent initializes with custom configuration."""
        config = AgentConfig(timeout=60.0, max_retries=5)
        agent = ReActAgent(
            name="CustomAgent",
            config=config,
            max_steps=10,
            temperature=0.5,
            model_name="llama3.2:1b",
        )

        assert agent.name == "CustomAgent"
        assert agent.config.timeout == 60.0
        assert agent.config.max_retries == 5
        assert agent.max_steps == 10
        assert agent.temperature == 0.5
        assert agent.model_name == "llama3.2:1b"

    def test_agent_initialization_with_mocked_services(self):
        """Test agent initializes with mocked MCP and Ollama services."""
        mock_mcp = MagicMock()
        mock_ollama = MagicMock()

        agent = ReActAgent(
            mcp_server=mock_mcp,
            ollama_service=mock_ollama,
        )

        assert agent.mcp_server is mock_mcp
        assert agent.ollama_service is mock_ollama


class TestReActAgentBeforeExecute:
    """Test before_execute hook."""

    def test_before_execute_resets_state(self):
        """Test that before_execute resets execution state."""
        agent = ReActAgent()

        # Simulate some previous state
        agent.steps = [ReActStep(1, "test", "action", {})]
        agent.collected_data = {"emails": [{"test": "data"}]}

        # Call before_execute
        agent.before_execute()

        # Verify state is reset
        assert agent.steps == []
        assert agent.collected_data == {
            "emails": [],
            "calendar_events": [],
            "news": [],
            "weather": None,
        }


class TestReActAgentReasoning:
    """Test ReAct agent reasoning methods."""

    def test_parse_reasoning_response_with_valid_format(self):
        """Test parsing valid LLM reasoning response."""
        agent = ReActAgent()

        response = """Thought: I need to fetch emails
Action: fetch_emails
Action Input: {"max_results": 20}"""

        thought, action, action_input = agent._parse_reasoning_response(response)

        assert thought == "I need to fetch emails"
        assert action == "fetch_emails"
        assert action_input == {"max_results": 20}

    def test_parse_reasoning_response_with_finish_action(self):
        """Test parsing response with finish action."""
        agent = ReActAgent()

        response = """Thought: All data collected
Action: finish
Action Input: null"""

        thought, action, action_input = agent._parse_reasoning_response(response)

        assert thought == "All data collected"
        assert action == "finish"
        assert action_input is None

    def test_parse_reasoning_response_with_missing_fields(self):
        """Test parsing response with missing fields uses defaults."""
        agent = ReActAgent()

        response = "Some unstructured text"

        thought, action, action_input = agent._parse_reasoning_response(response)

        assert thought == "Proceeding with next step"
        assert action == "finish"
        assert action_input is None

    def test_fallback_reasoning_step_1(self):
        """Test fallback reasoning returns news fetch for step 1."""
        agent = ReActAgent()

        thought, action, action_input = agent._fallback_reasoning(1)

        assert "news" in thought.lower()
        assert action == "fetch_hackernews"
        assert action_input == {"max_stories": 10}

    def test_fallback_reasoning_step_2(self):
        """Test fallback reasoning returns email fetch for step 2."""
        agent = ReActAgent()

        thought, action, action_input = agent._fallback_reasoning(2)

        assert "email" in thought.lower()
        assert action == "fetch_emails"
        assert action_input == {"max_results": 20}

    def test_fallback_reasoning_step_5(self):
        """Test fallback reasoning returns finish for step 5+."""
        agent = ReActAgent()

        thought, action, action_input = agent._fallback_reasoning(5)

        assert action == "finish"


class TestReActAgentDataStorage:
    """Test data storage methods."""

    def test_store_collected_data_emails(self):
        """Test storing email data."""
        agent = ReActAgent()
        mock_emails = load_mock_emails(count=5)

        agent._store_collected_data("fetch_emails", mock_emails)

        assert agent.collected_data["emails"] == mock_emails

    def test_store_collected_data_calendar(self):
        """Test storing calendar data."""
        agent = ReActAgent()
        mock_events = generate_mock_calendar_events(count=5)

        agent._store_collected_data("fetch_calendar_events", mock_events)

        assert agent.collected_data["calendar_events"] == mock_events

    def test_store_collected_data_news(self):
        """Test storing news data."""
        agent = ReActAgent()
        mock_news = [{"title": "News 1"}, {"title": "News 2"}]

        agent._store_collected_data("fetch_hackernews", mock_news)

        assert agent.collected_data["news"] == mock_news

    def test_store_collected_data_weather(self):
        """Test storing weather data."""
        agent = ReActAgent()
        mock_weather = {"temperature": 72, "condition": "sunny"}

        agent._store_collected_data("get_current_weather", mock_weather)

        assert agent.collected_data["weather"] == mock_weather


class TestReActAgentObservationSummary:
    """Test observation summary generation."""

    def test_summarize_observation_emails(self):
        """Test email observation summary."""
        agent = ReActAgent()
        mock_emails = load_mock_emails(count=10)

        summary = agent._summarize_observation("fetch_emails", mock_emails)

        assert "10 emails" in summary.lower()
        assert "high importance" in summary.lower()

    def test_summarize_observation_calendar(self):
        """Test calendar observation summary."""
        agent = ReActAgent()
        mock_events = generate_mock_calendar_events(count=8)

        summary = agent._summarize_observation("fetch_calendar_events", mock_events)

        assert "8" in summary
        assert "events" in summary.lower()

    def test_summarize_observation_news(self):
        """Test news observation summary."""
        agent = ReActAgent()
        mock_news = [{"title": f"News {i}"} for i in range(15)]

        summary = agent._summarize_observation("fetch_hackernews", mock_news)

        assert "15" in summary
        assert "news" in summary.lower()

    def test_summarize_observation_weather(self):
        """Test weather observation summary."""
        agent = ReActAgent()
        mock_weather = {
            "temperature": 72,
            "description": "partly cloudy",
        }

        summary = agent._summarize_observation("get_current_weather", mock_weather)

        assert "72" in summary
        assert "partly cloudy" in summary.lower()


class TestReActAgentActionItemExtraction:
    """Test action item extraction."""

    def test_extract_action_items_from_emails(self):
        """Test extracting action items from emails."""
        agent = ReActAgent()

        # Create mock emails with action items
        agent.collected_data["emails"] = [
            {
                "subject": "Email 1",
                "has_action_items": True,
                "action_items": ["Review document", "Send feedback"],
            },
            {
                "subject": "Email 2",
                "has_action_items": False,
                "action_items": [],
            },
            {
                "subject": "Email 3",
                "has_action_items": True,
                "action_items": ["Schedule meeting"],
            },
        ]

        action_items = agent._extract_action_items()

        assert len(action_items) == 3
        assert "Review document" in action_items
        assert "Send feedback" in action_items
        assert "Schedule meeting" in action_items

    def test_extract_action_items_from_calendar(self):
        """Test extracting action items from calendar events."""
        agent = ReActAgent()

        agent.collected_data["calendar_events"] = [
            {
                "summary": "Meeting 1",
                "requires_preparation": True,
                "preparation_notes": ["Prepare slides", "Review agenda"],
            },
            {
                "summary": "Meeting 2",
                "requires_preparation": False,
                "preparation_notes": [],
            },
        ]

        action_items = agent._extract_action_items()

        assert len(action_items) == 2
        assert "[Meeting 1] Prepare slides" in action_items
        assert "[Meeting 1] Review agenda" in action_items

    def test_extract_action_items_from_both_sources(self):
        """Test extracting action items from both emails and calendar."""
        agent = ReActAgent()

        agent.collected_data["emails"] = [
            {
                "has_action_items": True,
                "action_items": ["Email action"],
            }
        ]

        agent.collected_data["calendar_events"] = [
            {
                "summary": "Event",
                "requires_preparation": True,
                "preparation_notes": ["Event prep"],
            }
        ]

        action_items = agent._extract_action_items()

        assert len(action_items) == 2
        assert "Email action" in action_items
        assert "[Event] Event prep" in action_items


class TestReActAgentSynthesisPromptBuilding:
    """Test synthesis prompt building."""

    def test_build_synthesis_prompt_with_all_data(self):
        """Test building synthesis prompt with all data types."""
        agent = ReActAgent()

        agent.collected_data = {
            "emails": [{"subject": "Test Email", "importance_score": 0.8}],
            "calendar_events": [{"summary": "Test Meeting", "start_time": "2025-01-28T10:00:00"}],
            "news": [{"title": "Breaking News"}],
            "weather": {"temperature": 72, "condition": "sunny"},
        }

        prompt = agent._build_synthesis_prompt()

        assert "Emails" in prompt
        assert "Calendar Events" in prompt
        assert "News" in prompt
        assert "Weather" in prompt
        assert "Test Email" in prompt
        assert "Test Meeting" in prompt
        assert "Breaking News" in prompt

    def test_build_synthesis_prompt_with_empty_data(self):
        """Test building synthesis prompt with no data."""
        agent = ReActAgent()

        prompt = agent._build_synthesis_prompt()

        assert "No emails" in prompt
        assert "No events" in prompt
        assert "No news" in prompt


class TestReActAgentSynthesisResponseParsing:
    """Test synthesis response parsing."""

    def test_parse_synthesis_response_with_valid_format(self):
        """Test parsing valid synthesis response."""
        agent = ReActAgent()

        response = """Summary:
Today you have important emails to review and meetings to attend. Weather is good.

Key Points:
- Review 5 important emails
- Attend 3 meetings
- Complete project report
- Weather forecast: Sunny, 72°F"""

        summary, key_points = agent._parse_synthesis_response(response)

        assert "important emails" in summary.lower()
        assert len(key_points) == 4
        assert "Review 5 important emails" in key_points
        assert "Attend 3 meetings" in key_points

    def test_parse_synthesis_response_with_missing_summary(self):
        """Test parsing response with missing summary uses fallback."""
        agent = ReActAgent()

        response = """Key Points:
- Point 1
- Point 2"""

        summary, key_points = agent._parse_synthesis_response(response)

        assert "Daily brief generated" in summary
        assert len(key_points) == 2

    def test_parse_synthesis_response_with_no_structure(self):
        """Test parsing unstructured response uses fallbacks."""
        agent = ReActAgent()

        response = "Some random text"

        summary, key_points = agent._parse_synthesis_response(response)

        assert "Daily brief generated" in summary
        assert len(key_points) == 3  # Default fallback key points

    def test_fallback_summary_generation(self):
        """Test fallback summary with data counts."""
        agent = ReActAgent()

        agent.collected_data = {
            "emails": [1, 2, 3],
            "calendar_events": [1, 2],
            "news": [1, 2, 3, 4],
            "weather": None,
        }

        summary, key_points = agent._fallback_summary()

        assert "3 emails" in summary
        assert "2" in summary  # calendar events
        assert "4 news" in summary
        assert len(key_points) == 3


class TestReActAgentDailyBriefGeneration:
    """Test daily brief generation."""

    @patch('asyncio.run')
    def test_generate_daily_brief_with_mocked_llm(self, mock_asyncio_run):
        """Test daily brief generation with mocked LLM."""
        agent = ReActAgent()

        # Setup mock data
        agent.collected_data = {
            "emails": load_mock_emails(count=5),
            "calendar_events": generate_mock_calendar_events(count=3),
            "news": [{"title": f"News {i}"} for i in range(10)],
            "weather": {"temperature": 72},
        }

        # Mock LLM response
        mock_ollama_response = MagicMock()
        mock_ollama_response.text = """Summary:
Daily brief with 5 emails and 3 events.

Key Points:
- Check emails
- Attend meetings
- Review news"""
        mock_asyncio_run.return_value = mock_ollama_response

        # Generate brief
        brief = agent._generate_daily_brief()

        assert isinstance(brief, DailyBrief)
        assert brief.emails_count == 5
        assert brief.calendar_events_count == 3
        assert brief.news_items_count == 10
        assert brief.weather_info == {"temperature": 72}
        assert len(brief.key_points) >= 1
        assert len(brief.summary) > 0


class TestReActAgentActMethod:
    """Test _act method for executing MCP tools."""

    @patch('asyncio.run')
    def test_act_with_successful_email_fetch(self, mock_asyncio_run):
        """Test acting with successful email fetch."""
        agent = ReActAgent()

        # Mock MCP response
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = load_mock_emails(count=5)
        mock_asyncio_run.return_value = mock_response

        observation = agent._act("fetch_emails", {"max_results": 5})

        assert "5 emails" in observation.lower()
        assert agent.collected_data["emails"] == mock_response.data

    @patch('asyncio.run')
    def test_act_with_failed_tool_execution(self, mock_asyncio_run):
        """Test acting with failed tool execution."""
        agent = ReActAgent()

        # Mock MCP error response
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error = "Tool execution failed"
        mock_asyncio_run.return_value = mock_response

        observation = agent._act("fetch_emails", {})

        assert "error" in observation.lower()
        assert "Tool execution failed" in observation

    @patch('asyncio.run')
    def test_act_with_exception(self, mock_asyncio_run):
        """Test acting when exception occurs."""
        agent = ReActAgent()

        # Mock exception
        mock_asyncio_run.side_effect = Exception("MCP server unavailable")

        observation = agent._act("fetch_emails", {})

        assert "failed" in observation.lower()
        assert "MCP server unavailable" in observation


class TestReActAgentFullExecution:
    """Test full ReAct agent execution."""

    @patch('asyncio.run')
    def test_execute_with_fallback_reasoning(self, mock_asyncio_run):
        """Test full execution using fallback reasoning (no LLM)."""
        agent = ReActAgent(max_steps=5)

        # Mock all MCP tool calls to return success
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = []

        # Mock Ollama calls to raise exception (triggers fallback)
        def mock_run(coro):
            # If it's an Ollama call, raise exception
            if hasattr(coro, '__name__') and 'generate' in str(coro):
                raise Exception("Ollama unavailable")
            # MCP calls return success
            return mock_response

        mock_asyncio_run.side_effect = mock_run

        # Execute
        brief = agent._execute()

        # Verify
        assert isinstance(brief, DailyBrief)
        assert len(agent.steps) > 0
        assert brief.summary is not None
        assert len(brief.key_points) > 0

    @patch('asyncio.run')
    def test_execute_stops_at_max_steps(self, mock_asyncio_run):
        """Test execution stops at max_steps."""
        agent = ReActAgent(max_steps=3)

        # Mock to never return "finish"
        mock_ollama_response = MagicMock()
        mock_ollama_response.text = """Thought: Keep going
Action: fetch_emails
Action Input: {}"""

        mock_mcp_response = MagicMock()
        mock_mcp_response.success = True
        mock_mcp_response.data = []

        def mock_run(coro):
            if 'generate' in str(type(coro)):
                return mock_ollama_response
            return mock_mcp_response

        mock_asyncio_run.side_effect = mock_run

        # Execute
        brief = agent._execute()

        # Should stop at max_steps
        assert len(agent.steps) <= 3


class TestReActAgentIntegration:
    """Integration tests for ReAct agent."""

    @patch('asyncio.run')
    def test_agent_run_returns_agent_result(self, mock_asyncio_run):
        """Test that agent.run() returns AgentResult."""
        agent = ReActAgent(max_steps=2)

        # Mock responses
        mock_mcp_response = MagicMock()
        mock_mcp_response.success = True
        mock_mcp_response.data = []

        mock_ollama_response = MagicMock()
        mock_ollama_response.text = """Thought: Finish
Action: finish
Action Input: null"""

        def mock_run(coro):
            if 'generate' in str(type(coro)):
                return mock_ollama_response
            return mock_mcp_response

        mock_asyncio_run.side_effect = mock_run

        # Run agent
        result = agent.run()

        # Verify
        assert result.success is True
        assert isinstance(result.data, DailyBrief)
        assert result.execution_time > 0
        assert result.metadata["agent_name"] == "ReActAgent"

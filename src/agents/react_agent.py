"""ReAct Agent - Reasoning and Acting for Daily Minutes Generation

This agent implements the ReAct (Reasoning + Acting) pattern to generate
intelligent daily briefs by:
1. Thinking (reasoning about what to do)
2. Acting (calling MCP tools to fetch data)
3. Observing (analyzing results and deciding next steps)

The agent uses:
- MCP Server for tool execution (email, calendar, news, weather)
- Ollama LLM for reasoning and synthesis
- BaseAgent framework for lifecycle management
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio

import structlog

from .base_agent import BaseAgent, AgentConfig, AgentResult
from src.services.mcp_server import get_mcp_server, MCPServer
from src.services.ollama_service import get_ollama_service, OllamaService


logger = structlog.get_logger(__name__)


@dataclass
class ReActStep:
    """Single step in ReAct loop."""

    step_num: int
    thought: str  # What the agent is thinking
    action: Optional[str] = None  # Tool to call (or None if final)
    action_input: Optional[Dict[str, Any]] = None  # Tool parameters
    observation: Optional[str] = None  # Result of action
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DailyBrief:
    """Generated daily brief with all data."""

    summary: str
    key_points: List[str]
    action_items: List[str]
    emails_count: int
    calendar_events_count: int
    news_items_count: int
    weather_info: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


class ReActAgent(BaseAgent[DailyBrief]):
    """ReAct Agent for generating Daily Minutes.

    This agent implements the ReAct pattern to intelligently fetch and synthesize
    information from multiple sources (email, calendar, news, weather) using
    the MCP server and Ollama LLM.

    Example:
        >>> agent = ReActAgent(max_steps=10)
        >>> result = agent.run()
        >>> if result.success:
        ...     print(result.data.summary)
    """

    def __init__(
        self,
        name: str = "ReActAgent",
        config: Optional[AgentConfig] = None,
        mcp_server: Optional[MCPServer] = None,
        ollama_service: Optional[OllamaService] = None,
        max_steps: int = 15,
        temperature: float = 0.7,
        model_name: str = "llama3.2:3b",
        **kwargs
    ):
        """Initialize ReAct Agent.

        Args:
            name: Agent name
            config: Agent configuration
            mcp_server: MCP server instance (or None to create)
            ollama_service: Ollama service instance (or None to create)
            max_steps: Maximum ReAct loop iterations
            temperature: LLM temperature for reasoning
            model_name: Ollama model to use
            **kwargs: Additional args for BaseAgent
        """
        super().__init__(name=name, config=config or AgentConfig(), **kwargs)

        # Initialize services
        self.mcp_server = mcp_server or get_mcp_server()
        self.ollama_service = ollama_service or get_ollama_service()

        # ReAct configuration
        self.max_steps = max_steps
        self.temperature = temperature
        self.model_name = model_name

        # Execution state
        self.steps: List[ReActStep] = []
        self.collected_data: Dict[str, Any] = {
            "emails": [],
            "calendar_events": [],
            "news": [],
            "weather": None,
        }

        self.logger.info(
            "react_agent_initialized",
            max_steps=max_steps,
            model=model_name,
            temperature=temperature,
        )

    def before_execute(self) -> None:
        """Reset execution state before starting."""
        self.steps = []
        self.collected_data = {
            "emails": [],
            "calendar_events": [],
            "news": [],
            "weather": None,
        }
        self.logger.info("react_agent_starting", goal="Generate daily brief")

    def _execute(self) -> DailyBrief:
        """Execute ReAct loop to generate daily brief.

        Returns:
            DailyBrief with all collected data and generated summary
        """
        goal = "Generate a comprehensive daily brief with emails, calendar, news, and weather"

        self.logger.info("react_loop_starting", goal=goal, max_steps=self.max_steps)

        # ReAct loop
        for step_num in range(1, self.max_steps + 1):
            self.logger.debug("react_step_starting", step=step_num)

            # Think: What should we do next?
            thought, action, action_input = self._reason(goal, step_num)

            # Create step record
            step = ReActStep(
                step_num=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
            )

            # Act: Execute the chosen action (or finish)
            if action is None or action.lower() == "finish":
                self.logger.info("react_loop_finishing", reason="Agent decided to finish")
                step.observation = "Task complete - generating final summary"
                self.steps.append(step)
                break

            # Execute action via MCP
            observation = self._act(action, action_input or {})
            step.observation = observation
            self.steps.append(step)

            self.logger.info(
                "react_step_completed",
                step=step_num,
                action=action,
                observation_length=len(observation) if observation else 0,
            )

        # Generate final daily brief
        daily_brief = self._generate_daily_brief()

        self.logger.info(
            "react_loop_completed",
            total_steps=len(self.steps),
            action_items=len(daily_brief.action_items),
            key_points=len(daily_brief.key_points),
        )

        return daily_brief

    def _reason(
        self,
        goal: str,
        step_num: int
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """Use LLM to reason about next action.

        Args:
            goal: Overall goal to achieve
            step_num: Current step number

        Returns:
            (thought, action, action_input) tuple
        """
        # Build prompt with current state
        prompt = self._build_reasoning_prompt(goal, step_num)

        # Query LLM (using asyncio.run for async call)
        try:
            response_obj = asyncio.run(self.ollama_service.generate(
                prompt=prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=500,
            ))

            # Extract text from OllamaResponse
            response = response_obj.text if hasattr(response_obj, 'text') else str(response_obj)

            # Parse LLM response
            thought, action, action_input = self._parse_reasoning_response(response)

            return thought, action, action_input

        except Exception as e:
            self.logger.error("reasoning_failed", error=str(e))
            # Fallback: simple sequential plan
            return self._fallback_reasoning(step_num)

    def _build_reasoning_prompt(self, goal: str, step_num: int) -> str:
        """Build prompt for LLM reasoning.

        Args:
            goal: Overall goal
            step_num: Current step number

        Returns:
            Prompt string for LLM
        """
        # Get available tools from MCP
        available_tools = list(self.mcp_server.tools.keys())

        # Build execution history
        history = "\n".join([
            f"Step {s.step_num}: {s.thought}\n"
            f"  Action: {s.action or 'None'}\n"
            f"  Observation: {s.observation[:100] if s.observation else 'None'}..."
            for s in self.steps[-3:]  # Last 3 steps for context
        ])

        # Current data status
        data_status = {
            "emails_fetched": len(self.collected_data.get("emails", [])) > 0,
            "calendar_fetched": len(self.collected_data.get("calendar_events", [])) > 0,
            "news_fetched": len(self.collected_data.get("news", [])) > 0,
            "weather_fetched": self.collected_data.get("weather") is not None,
        }

        prompt = f"""You are a ReAct agent helping generate a daily brief.

**Goal**: {goal}

**Available Tools** (call via MCP):
{', '.join(available_tools)}

**Current Step**: {step_num}/{self.max_steps}

**Data Collection Status**:
- Emails: {'✓' if data_status['emails_fetched'] else '✗'}
- Calendar: {'✓' if data_status['calendar_fetched'] else '✗'}
- News: {'✓' if data_status['news_fetched'] else '✗'}
- Weather: {'✓' if data_status['weather_fetched'] else '✗'}

**Recent History**:
{history if history else "No steps yet"}

**Instructions**:
Think about what to do next. You should:
1. Fetch data from all sources (emails, calendar, news, weather)
2. Once all data is collected, call "finish" to generate the summary

Respond in this format:
Thought: <your reasoning>
Action: <tool_name or "finish">
Action Input: <JSON parameters or null>

Example:
Thought: I need to fetch news to include in the daily brief
Action: fetch_hackernews
Action Input: {{"max_stories": 10}}

OR when done:
Thought: All data collected, ready to generate summary
Action: finish
Action Input: null

Your response:"""

        return prompt

    def _parse_reasoning_response(
        self,
        response: str
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """Parse LLM reasoning response.

        Args:
            response: LLM response text

        Returns:
            (thought, action, action_input) tuple
        """
        lines = response.strip().split("\n")

        thought = None
        action = None
        action_input = None

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                if input_str and input_str.lower() not in ["null", "none", "{}"]:
                    try:
                        action_input = json.loads(input_str)
                    except json.JSONDecodeError:
                        action_input = {}

        # Defaults
        if not thought:
            thought = "Proceeding with next step"
        if not action:
            action = "finish"

        return thought, action, action_input

    def _fallback_reasoning(
        self,
        step_num: int
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """Fallback reasoning if LLM fails.

        Simple sequential plan:
        1. Fetch news
        2. Fetch emails
        3. Fetch calendar
        4. Fetch weather
        5. Finish

        Args:
            step_num: Current step number

        Returns:
            (thought, action, action_input) tuple
        """
        if step_num == 1:
            return ("Fetching news headlines", "fetch_hackernews", {"max_stories": 10})
        elif step_num == 2:
            return ("Checking emails", "fetch_emails", {"max_results": 20})
        elif step_num == 3:
            return ("Reviewing calendar events", "fetch_calendar_events", {"days_ahead": 7})
        elif step_num == 4:
            return ("Getting weather info", "get_current_weather", {})
        else:
            return ("All data collected", "finish", None)

    def _act(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute action via MCP server.

        Args:
            action: Tool name to call
            action_input: Tool parameters

        Returns:
            Observation string describing result
        """
        try:
            self.logger.debug("executing_action", action=action, input=action_input)

            # Execute via MCP (using asyncio.run for async call)
            response = asyncio.run(self.mcp_server.execute_tool(action, action_input))

            if response.success:
                # Store data by category
                self._store_collected_data(action, response.data)

                # Return observation summary
                observation = self._summarize_observation(action, response.data)
                return observation
            else:
                return f"Error executing {action}: {response.error}"

        except Exception as e:
            self.logger.error("action_execution_failed", action=action, error=str(e))
            return f"Failed to execute {action}: {str(e)}"

    def _store_collected_data(self, action: str, data: Any):
        """Store fetched data in collected_data dict.

        Args:
            action: Tool that was called
            data: Data returned from tool
        """
        if action == "fetch_emails":
            self.collected_data["emails"] = data
        elif action == "fetch_calendar_events":
            self.collected_data["calendar_events"] = data
        elif action in ["fetch_hackernews", "search_hackernews", "fetch_rss_feeds"]:
            if "news" not in self.collected_data or not isinstance(self.collected_data["news"], list):
                self.collected_data["news"] = []
            if isinstance(data, list):
                self.collected_data["news"].extend(data)
            else:
                self.collected_data["news"].append(data)
        elif action in ["get_current_weather", "get_weather_forecast"]:
            self.collected_data["weather"] = data

    def _summarize_observation(self, action: str, data: Any) -> str:
        """Create human-readable observation summary.

        Args:
            action: Tool that was called
            data: Data returned

        Returns:
            Summary string
        """
        def safe_get(obj, key, default=None):
            """Safely get attribute from dict or Pydantic model."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return getattr(obj, key, default)

        if action == "fetch_emails":
            count = len(data) if isinstance(data, list) else 0
            high_importance = sum(1 for e in data if safe_get(e, "importance_score", 0) > 0.7) if isinstance(data, list) else 0
            return f"Fetched {count} emails ({high_importance} high importance)"

        elif action == "fetch_calendar_events":
            count = len(data) if isinstance(data, list) else 0
            requires_prep = sum(1 for e in data if safe_get(e, "requires_preparation", False)) if isinstance(data, list) else 0
            return f"Found {count} calendar events ({requires_prep} require preparation)"

        elif action in ["fetch_hackernews", "search_hackernews"]:
            count = len(data) if isinstance(data, list) else 0
            return f"Retrieved {count} news items from HackerNews"

        elif action == "get_current_weather":
            if isinstance(data, dict):
                temp = data.get("temperature", "unknown")
                condition = data.get("description", "unknown")
                return f"Current weather: {temp}°F, {condition}"
            return "Weather data retrieved"

        else:
            return f"Executed {action} successfully"

    def _generate_daily_brief(self) -> DailyBrief:
        """Generate final daily brief from collected data.

        Returns:
            DailyBrief object with summary and metadata
        """
        self.logger.info("generating_daily_brief")

        # Extract action items from emails and calendar
        action_items = self._extract_action_items()

        # Generate summary using LLM
        summary, key_points = self._synthesize_summary()

        # Build daily brief
        brief = DailyBrief(
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            emails_count=len(self.collected_data.get("emails", [])),
            calendar_events_count=len(self.collected_data.get("calendar_events", [])),
            news_items_count=len(self.collected_data.get("news", [])),
            weather_info=self.collected_data.get("weather"),
            metadata={
                "total_steps": len(self.steps),
                "model_used": self.model_name,
            }
        )

        return brief

    def _extract_action_items(self) -> List[str]:
        """Extract action items from emails and calendar.

        Returns:
            List of action item strings
        """
        def safe_get(obj, key, default=None):
            """Safely get attribute from dict or Pydantic model."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return getattr(obj, key, default)

        action_items = []

        # From emails
        emails = self.collected_data.get("emails", [])
        for email in emails:
            if safe_get(email, "has_action_items") and safe_get(email, "action_items"):
                items = safe_get(email, "action_items", [])
                action_items.extend(items)

        # From calendar events
        events = self.collected_data.get("calendar_events", [])
        for event in events:
            if safe_get(event, "requires_preparation") and safe_get(event, "preparation_notes"):
                notes = safe_get(event, "preparation_notes", [])
                summary = safe_get(event, "summary", "Event")
                action_items.extend([
                    f"[{summary}] {note}"
                    for note in notes
                ])

        return action_items

    def _synthesize_summary(self) -> tuple[str, List[str]]:
        """Use LLM to synthesize all data into a coherent summary.

        Returns:
            (summary_text, key_points_list) tuple
        """
        try:
            prompt = self._build_synthesis_prompt()

            response_obj = asyncio.run(self.ollama_service.generate(
                prompt=prompt,
                model=self.model_name,
                temperature=0.5,  # Lower temp for more focused summary
                max_tokens=1000,
            ))

            # Extract text from OllamaResponse
            response = response_obj.text if hasattr(response_obj, 'text') else str(response_obj)

            # Parse response into summary and key points
            summary, key_points = self._parse_synthesis_response(response)

            return summary, key_points

        except Exception as e:
            self.logger.error("synthesis_failed", error=str(e))
            # Fallback: basic summary
            return self._fallback_summary()

    def _build_synthesis_prompt(self) -> str:
        """Build prompt for LLM synthesis.

        Returns:
            Prompt string
        """
        emails = self.collected_data.get("emails", [])
        events = self.collected_data.get("calendar_events", [])
        news = self.collected_data.get("news", [])
        weather = self.collected_data.get("weather")

        prompt = f"""Generate a comprehensive daily brief from this data:

**Emails** ({len(emails)}):
{self._format_emails_for_prompt(emails[:5])}

**Calendar Events** ({len(events)}):
{self._format_events_for_prompt(events[:5])}

**News** ({len(news)}):
{self._format_news_for_prompt(news[:5])}

**Weather**:
{weather if weather else 'Not available'}

Create a daily brief with:
1. A concise summary paragraph (2-3 sentences)
2. 3-5 key points for today

Format:
Summary:
<your summary paragraph>

Key Points:
- <point 1>
- <point 2>
- <point 3>
...

Your response:"""

        return prompt

    def _format_emails_for_prompt(self, emails: List[Dict]) -> str:
        """Format emails for inclusion in prompt."""
        if not emails:
            return "No emails"

        return "\n".join([
            f"- {e.get('subject', 'No subject')} (importance: {e.get('importance_score', 0):.2f})"
            for e in emails
        ])

    def _format_events_for_prompt(self, events: List[Dict]) -> str:
        """Format calendar events for inclusion in prompt."""
        if not events:
            return "No events"

        return "\n".join([
            f"- {e.get('summary', 'No title')} at {e.get('start_time', 'unknown')}"
            for e in events
        ])

    def _format_news_for_prompt(self, news: List[Dict]) -> str:
        """Format news items for inclusion in prompt."""
        if not news:
            return "No news"

        return "\n".join([
            f"- {n.get('title', 'No title')}"
            for n in news
        ])

    def _parse_synthesis_response(self, response: str) -> tuple[str, List[str]]:
        """Parse LLM synthesis response.

        Args:
            response: LLM response

        Returns:
            (summary, key_points) tuple
        """
        lines = response.strip().split("\n")

        summary = ""
        key_points = []

        in_summary = False
        in_key_points = False

        for line in lines:
            line = line.strip()

            if "Summary:" in line or "summary:" in line.lower():
                in_summary = True
                in_key_points = False
                # Get summary on same line if present
                after_colon = line.split(":", 1)[-1].strip()
                if after_colon:
                    summary = after_colon
                continue

            if "Key Points:" in line or "key points:" in line.lower():
                in_summary = False
                in_key_points = True
                continue

            if in_summary and line:
                summary += " " + line if summary else line
            elif in_key_points and line.startswith("-"):
                point = line.lstrip("- ").strip()
                if point:
                    key_points.append(point)

        # Fallback
        if not summary:
            summary = "Daily brief generated from emails, calendar, news, and weather data."
        if not key_points:
            key_points = ["Check emails for updates", "Review calendar events", "Stay informed on news"]

        return summary.strip(), key_points

    def _fallback_summary(self) -> tuple[str, List[str]]:
        """Generate fallback summary if LLM fails.

        Returns:
            (summary, key_points) tuple
        """
        emails = self.collected_data.get("emails", [])
        events = self.collected_data.get("calendar_events", [])
        news = self.collected_data.get("news", [])

        summary = (
            f"Daily brief: {len(emails)} emails, {len(events)} calendar events, "
            f"and {len(news)} news items reviewed."
        )

        key_points = [
            f"{len(emails)} emails to review",
            f"{len(events)} upcoming calendar events",
            f"{len(news)} news items",
        ]

        return summary, key_points

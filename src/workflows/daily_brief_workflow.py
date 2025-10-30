"""LangGraph Workflow for Daily Brief Generation.

This workflow orchestrates all components to generate a personalized daily brief:
- RAG Memory for context
- MCP connectors for data
- User preferences for personalization
- ReAct agent for reasoning
- Feedback collection for learning
"""

from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.retrieval import MemoryRetriever
from src.memory.models import DailyBriefMemory
from src.agents.react_agent import ReActAgent
from src.services.preference_tracker import PreferenceTracker
from src.services.feedback_collector import FeedbackCollector


class WorkflowStep(Enum):
    """Steps in the daily brief workflow."""
    START = "start"
    RETRIEVE_CONTEXT = "retrieve_context"
    FETCH_DATA = "fetch_data"
    LOAD_PREFERENCES = "load_preferences"
    AGENT_REASONING = "agent_reasoning"
    STORE_MEMORY = "store_memory"
    END = "end"


class WorkflowState(TypedDict):
    """State maintained throughout the workflow."""
    user_query: str
    context: List[Any]
    emails: List[Dict[str, Any]]
    calendar_events: List[Dict[str, Any]]
    news_items: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    agent_response: str
    step: WorkflowStep


class DailyBriefWorkflow:
    """LangGraph workflow for generating daily briefs.

    This workflow coordinates all components to generate a personalized
    daily brief that learns from user feedback.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        user_id: str = "default"
    ):
        """Initialize the workflow.

        Args:
            persist_directory: ChromaDB persistence directory
            user_id: User identifier
        """
        # Initialize all components
        self.memory_retriever = MemoryRetriever(persist_directory=persist_directory)
        self.react_agent = ReActAgent()
        self.preference_tracker = PreferenceTracker(
            persist_directory=persist_directory,
            user_id=user_id
        )
        self.feedback_collector = FeedbackCollector(
            persist_directory=persist_directory,
            user_id=user_id
        )

        # MCP clients (will be injected or mocked in tests)
        self.email_client: Optional[Any] = None
        self.calendar_client: Optional[Any] = None
        self.news_client: Optional[Any] = None

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine.

        Returns:
            Compiled state graph
        """
        # Create the graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("fetch_data", self._fetch_data)
        workflow.add_node("load_preferences", self._load_preferences)
        workflow.add_node("agent_reasoning", self._agent_reasoning)
        workflow.add_node("store_memory", self._store_memory)

        # Define edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "fetch_data")
        workflow.add_edge("fetch_data", "load_preferences")
        workflow.add_edge("load_preferences", "agent_reasoning")
        workflow.add_edge("agent_reasoning", "store_memory")
        workflow.add_edge("store_memory", END)

        # Compile the graph
        return workflow.compile()

    async def _retrieve_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Retrieve relevant context from memory.

        Args:
            state: Current workflow state

        Returns:
            Updated state with context
        """
        try:
            # Get relevant context from RAG memory
            context = await self.memory_retriever.get_relevant_context(
                query=state["user_query"],
                context_window=5
            )

            return {
                "context": context,
                "step": WorkflowStep.FETCH_DATA,
            }
        except Exception as e:
            # Graceful degradation: continue without context
            return {
                "context": [],
                "step": WorkflowStep.FETCH_DATA,
            }

    async def _fetch_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Fetch data from MCP connectors.

        Args:
            state: Current workflow state

        Returns:
            Updated state with fetched data
        """
        emails = []
        calendar_events = []
        news_items = []

        # Fetch emails
        try:
            if self.email_client:
                emails = await self.email_client.get_recent_emails()
        except Exception:
            # Continue without emails on error
            pass

        # Fetch calendar events
        try:
            if self.calendar_client:
                calendar_events = await self.calendar_client.get_today_events()
        except Exception:
            # Continue without calendar on error
            pass

        # Fetch news
        try:
            if self.news_client:
                news_items = await self.news_client.get_top_stories()
        except Exception:
            # Continue without news on error
            pass

        return {
            "emails": emails,
            "calendar_events": calendar_events,
            "news_items": news_items,
            "step": WorkflowStep.LOAD_PREFERENCES,
        }

    async def _load_preferences(self, state: WorkflowState) -> Dict[str, Any]:
        """Load user preferences.

        Args:
            state: Current workflow state

        Returns:
            Updated state with preferences
        """
        try:
            # Export preferences as dictionary
            preferences = await self.preference_tracker.export_preferences()

            return {
                "preferences": preferences,
                "step": WorkflowStep.AGENT_REASONING,
            }
        except Exception:
            # Continue without preferences on error
            return {
                "preferences": {},
                "step": WorkflowStep.AGENT_REASONING,
            }

    async def _agent_reasoning(self, state: WorkflowState) -> Dict[str, Any]:
        """Use ReAct agent to generate response.

        Args:
            state: Current workflow state

        Returns:
            Updated state with agent response
        """
        try:
            # Prepare context for agent
            context_summary = self._prepare_agent_context(state)

            # Run ReAct agent
            agent_response = await self.react_agent.run(
                query=state["user_query"],
                context=context_summary
            )

            return {
                "agent_response": agent_response,
                "step": WorkflowStep.STORE_MEMORY,
            }
        except Exception as e:
            # Provide fallback response
            return {
                "agent_response": f"Unable to generate brief: {str(e)}",
                "step": WorkflowStep.STORE_MEMORY,
            }

    def _prepare_agent_context(self, state: WorkflowState) -> str:
        """Prepare context string for agent.

        Args:
            state: Current workflow state

        Returns:
            Formatted context string
        """
        parts = []

        # Add memory context
        if state.get("context"):
            parts.append("# Previous Context")
            for ctx in state["context"][:3]:  # Limit to 3 most relevant
                parts.append(f"- {ctx.memory.content[:200]}")

        # Add email summary
        if state.get("emails"):
            parts.append(f"\n# Emails ({len(state['emails'])} total)")
            for email in state["emails"][:5]:  # Show top 5
                parts.append(f"- {email.get('subject', 'No subject')}")

        # Add calendar summary
        if state.get("calendar_events"):
            parts.append(f"\n# Calendar ({len(state['calendar_events'])} events)")
            for event in state["calendar_events"][:5]:
                parts.append(f"- {event.get('title', 'No title')}")

        # Add news summary
        if state.get("news_items"):
            parts.append(f"\n# News ({len(state['news_items'])} items)")
            for item in state["news_items"][:5]:
                parts.append(f"- {item.get('title', 'No title')}")

        # Add preferences
        if state.get("preferences"):
            parts.append("\n# User Preferences")
            for category, prefs in state["preferences"].items():
                parts.append(f"- {category}: {prefs}")

        return "\n".join(parts)

    async def _store_memory(self, state: WorkflowState) -> Dict[str, Any]:
        """Store generated brief in memory.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        try:
            # Create daily brief memory
            brief = DailyBriefMemory(
                id=f"brief-{datetime.now().timestamp()}",
                summary=state["agent_response"],
                key_points=[],  # Could extract from response
                action_items=[],  # Could extract from response
                emails_count=len(state.get("emails", [])),
                calendar_events_count=len(state.get("calendar_events", [])),
                news_items_count=len(state.get("news_items", [])),
            )

            # Store in memory
            await self.memory_retriever.store_memory(brief)

        except Exception:
            # Continue even if storage fails
            pass

        return {
            "step": WorkflowStep.END,
        }

    async def run(self, user_query: str) -> Dict[str, Any]:
        """Execute the workflow.

        Args:
            user_query: User's query or request

        Returns:
            Result dictionary with response
        """
        # Initialize state
        initial_state: WorkflowState = {
            "user_query": user_query,
            "context": [],
            "emails": [],
            "calendar_events": [],
            "news_items": [],
            "preferences": {},
            "agent_response": "",
            "step": WorkflowStep.START,
        }

        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state.get("agent_response", ""),
            "emails_count": len(final_state.get("emails", [])),
            "calendar_count": len(final_state.get("calendar_events", [])),
            "news_count": len(final_state.get("news_items", [])),
        }

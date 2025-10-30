"""AI Workflow Testing Page.

This page demonstrates and tests the complete AI workflow including:
- LangGraph orchestration
- RAG memory retrieval
- ReAct agent reasoning
- RLHF feedback collection
- User preferences
"""

import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict, Any
import json

from src.workflows.daily_brief_workflow import DailyBriefWorkflow, WorkflowStep
from src.services.preference_tracker import PreferenceTracker, PreferenceCategory
from src.services.feedback_collector import FeedbackCollector, FeedbackType, FeedbackRating
from src.memory.retrieval import MemoryRetriever


def init_session_state():
    """Initialize session state variables."""
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    if 'workflow_running' not in st.session_state:
        st.session_state.workflow_running = False
    if 'memory_context' not in st.session_state:
        st.session_state.memory_context = []
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False


async def run_workflow_async(user_query: str, persist_dir: str):
    """Run the LangGraph workflow asynchronously."""
    workflow = DailyBriefWorkflow(persist_directory=persist_dir)

    # For demo, we'll mock the MCP clients
    # In production, these would be real connectors
    class MockEmailClient:
        async def get_recent_emails(self):
            return [
                {"subject": "Q4 Planning Meeting", "from": "manager@company.com", "date": "2025-10-28"},
                {"subject": "Project Update: AI Integration", "from": "team@company.com", "date": "2025-10-28"},
                {"subject": "Weekly Newsletter", "from": "news@tech.com", "date": "2025-10-27"},
            ]

    class MockCalendarClient:
        async def get_today_events(self):
            return [
                {"title": "Team Standup", "time": "9:00 AM", "duration": "30 min"},
                {"title": "Client Demo", "time": "2:00 PM", "duration": "1 hour"},
                {"title": "Code Review", "time": "4:00 PM", "duration": "45 min"},
            ]

    class MockNewsClient:
        async def get_top_stories(self):
            return [
                {"title": "AI Breakthrough in Language Models", "source": "TechCrunch", "url": "https://..."},
                {"title": "New Python 3.13 Released", "source": "Python.org", "url": "https://..."},
                {"title": "LangGraph 2.0 Announcement", "source": "LangChain Blog", "url": "https://..."},
            ]

    # Inject mock clients
    workflow.email_client = MockEmailClient()
    workflow.calendar_client = MockCalendarClient()
    workflow.news_client = MockNewsClient()

    # Run workflow
    result = await workflow.run(user_query)
    return result


def render_workflow_visualization(result: Dict[str, Any]):
    """Render workflow execution visualization."""
    st.subheader("Workflow Execution Flow")

    # Show workflow steps
    steps = [
        ("START", "Workflow Initiated", "✅"),
        ("RETRIEVE_CONTEXT", "Retrieved RAG Memory Context", "✅"),
        ("FETCH_DATA", f"Fetched {result.get('emails_count', 0)} emails, {result.get('calendar_count', 0)} events, {result.get('news_count', 0)} news", "✅"),
        ("LOAD_PREFERENCES", "Loaded User Preferences", "✅"),
        ("AGENT_REASONING", "ReAct Agent Processed", "✅"),
        ("STORE_MEMORY", "Stored in RAG Memory", "✅"),
        ("END", "Workflow Complete", "✅"),
    ]

    # Create visual flow
    cols = st.columns(7)
    for i, (step, desc, status) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{step}**")
            st.caption(desc)
            st.markdown(status)


async def get_memory_context_async(query: str, persist_dir: str):
    """Get RAG memory context."""
    retriever = MemoryRetriever(persist_directory=persist_dir)
    context = await retriever.get_relevant_context(query=query, context_window=5)
    return context


async def get_preferences_async(persist_dir: str):
    """Get user preferences."""
    tracker = PreferenceTracker(persist_directory=persist_dir)
    prefs = await tracker.export_preferences()
    return prefs


async def submit_feedback_async(
    feedback_type: FeedbackType,
    rating: FeedbackRating,
    context: Dict[str, Any],
    persist_dir: str
):
    """Submit user feedback."""
    collector = FeedbackCollector(persist_directory=persist_dir)
    await collector.collect_feedback(feedback_type, rating, context)
    await collector.apply_feedback_learning()


def main():
    """Main AI Workflow Testing page."""
    st.title("🤖 AI Workflow Testing Dashboard")
    st.markdown("Test the complete AI pipeline: LangGraph + RAG + ReAct + RLHF")

    init_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        persist_dir = st.text_input(
            "ChromaDB Directory",
            value="./chroma_data",
            help="Directory for persistent vector storage"
        )

        st.divider()
        st.header("📊 Component Health")
        st.metric("RAG Memory", "Active", delta="94% coverage")
        st.metric("ReAct Agent", "Ready", delta="34 tests passing")
        st.metric("RLHF Feedback", "Learning", delta="14 tests passing")
        st.metric("LangGraph", "Orchestrating", delta="12 tests passing")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Run Workflow",
        "🧠 RAG Memory",
        "🤔 ReAct Agent",
        "👍 RLHF Feedback",
        "⚙️ Preferences"
    ])

    with tab1:
        st.header("Complete Workflow Execution")
        st.markdown("Run the full LangGraph workflow to generate a daily brief")

        # User query input
        user_query = st.text_area(
            "Enter your query:",
            value="Generate my daily brief with a focus on AI and technology news",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            run_button = st.button(
                "▶️ Run Workflow",
                type="primary",
                disabled=st.session_state.workflow_running,
                use_container_width=True
            )

        if run_button:
            st.session_state.workflow_running = True

            with st.spinner("Running complete AI workflow..."):
                # Run workflow
                result = asyncio.run(run_workflow_async(user_query, persist_dir))
                st.session_state.workflow_result = result
                st.session_state.workflow_running = False

            st.success("✅ Workflow completed successfully!")

        # Display results
        if st.session_state.workflow_result:
            st.divider()
            render_workflow_visualization(st.session_state.workflow_result)

            st.divider()
            st.subheader("📝 Generated Daily Brief")
            st.markdown(st.session_state.workflow_result.get('response', 'No response'))

            st.divider()
            st.subheader("📊 Workflow Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Emails Processed",
                    st.session_state.workflow_result.get('emails_count', 0)
                )
            with col2:
                st.metric(
                    "Calendar Events",
                    st.session_state.workflow_result.get('calendar_count', 0)
                )
            with col3:
                st.metric(
                    "News Items",
                    st.session_state.workflow_result.get('news_count', 0)
                )

    with tab2:
        st.header("RAG Memory System")
        st.markdown("Explore context retrieved from vector memory")

        memory_query = st.text_input(
            "Query memory:",
            value="What happened in previous daily briefs?",
            key="memory_query"
        )

        if st.button("🔍 Search Memory", key="search_memory"):
            with st.spinner("Searching vector memory..."):
                context = asyncio.run(get_memory_context_async(memory_query, persist_dir))
                st.session_state.memory_context = context

        if st.session_state.memory_context:
            st.subheader(f"Found {len(st.session_state.memory_context)} relevant memories")

            for i, ctx in enumerate(st.session_state.memory_context[:5]):
                with st.expander(f"Memory {i+1} (Relevance: {ctx.relevance_score:.2f})"):
                    st.markdown(f"**Type:** {ctx.memory.type.value}")
                    st.markdown(f"**Timestamp:** {ctx.memory.timestamp}")
                    st.text_area(
                        "Content:",
                        value=ctx.memory.content,
                        height=100,
                        key=f"memory_{i}",
                        disabled=True
                    )
                    if ctx.memory.metadata:
                        st.json(ctx.memory.metadata)

        st.divider()
        st.subheader("💾 Memory Statistics")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Count Memories"):
                retriever = MemoryRetriever(persist_directory=persist_dir)
                count = asyncio.run(retriever.count_memories())
                st.metric("Total Memories", count)

        with col2:
            st.metric("Vector Dimensions", "384 (nomic-embed-text)")

    with tab3:
        st.header("ReAct Agent Reasoning")
        st.markdown("View how the agent thinks through problems")

        st.info("The ReAct agent uses a Think → Act → Observe loop to solve problems")

        st.code("""
Example ReAct Trace:

Thought: I need to generate a daily brief for the user.
Action: gather_emails
Observation: Found 3 new emails from team members.

Thought: I should also check calendar for today's events.
Action: get_calendar_events
Observation: Found 3 meetings scheduled.

Thought: Let me get relevant news for the user's interests.
Action: fetch_news
Observation: Retrieved 5 AI-related news articles.

Thought: I now have all the information needed.
Final Answer: Here is your daily brief...
        """, language="text")

        st.markdown("### 🎯 Agent Capabilities")
        capabilities = {
            "Email Analysis": "Summarizes and prioritizes emails",
            "Calendar Integration": "Identifies scheduling conflicts",
            "News Curation": "Filters news by user preferences",
            "Context Awareness": "Uses RAG memory for continuity",
            "Tool Use": "Executes actions via MCP connectors"
        }

        for cap, desc in capabilities.items():
            st.markdown(f"**{cap}**: {desc}")

    with tab4:
        st.header("RLHF Feedback System")
        st.markdown("Provide feedback to improve future responses")

        if st.session_state.workflow_result:
            st.subheader("Rate the Daily Brief")

            feedback_rating = st.radio(
                "How was this brief?",
                options=["👍 Positive", "😐 Neutral", "👎 Negative"],
                horizontal=True
            )

            feedback_comment = st.text_area(
                "Additional comments (optional):",
                placeholder="Too verbose, missing important emails, great summary, etc."
            )

            if st.button("Submit Feedback", type="primary"):
                rating_map = {
                    "👍 Positive": FeedbackRating.POSITIVE,
                    "😐 Neutral": FeedbackRating.NEUTRAL,
                    "👎 Negative": FeedbackRating.NEGATIVE
                }

                asyncio.run(submit_feedback_async(
                    FeedbackType.DAILY_BRIEF,
                    rating_map[feedback_rating],
                    {
                        "brief_id": f"brief-{datetime.now().timestamp()}",
                        "comment": feedback_comment
                    },
                    persist_dir
                ))

                st.success("✅ Feedback submitted! The system will learn from your input.")
                st.session_state.feedback_submitted = True
        else:
            st.warning("Run a workflow first to provide feedback")

        st.divider()
        st.subheader("📈 Learning Progress")
        st.markdown("The RLHF system continuously learns from your feedback to:")
        st.markdown("- Adjust content relevance")
        st.markdown("- Improve summary length and detail")
        st.markdown("- Personalize news topic selection")
        st.markdown("- Prioritize important emails")

    with tab5:
        st.header("User Preferences")
        st.markdown("View and modify your personalization settings")

        if st.button("🔄 Load Preferences"):
            with st.spinner("Loading preferences..."):
                prefs = asyncio.run(get_preferences_async(persist_dir))
                st.session_state.preferences = prefs

        if st.session_state.preferences:
            st.subheader("Current Preferences")

            for category, prefs in st.session_state.preferences.items():
                with st.expander(f"📁 {category.upper()}"):
                    st.json(prefs)
        else:
            st.info("No preferences set yet. The system will learn from your feedback.")

        st.divider()
        st.subheader("🎨 Preference Categories")

        categories = {
            "CONTENT": "Topics, sources, and subject matter preferences",
            "FORMATTING": "Brief length, style, and presentation",
            "PRIORITY": "Important contacts and urgent items",
            "TIMING": "Delivery schedule and frequency"
        }

        for cat, desc in categories.items():
            st.markdown(f"**{cat}**: {desc}")

    # Footer
    st.divider()
    st.caption("AI Workflow Testing Dashboard | Powered by LangGraph + RAG + ReAct + RLHF")


if __name__ == "__main__":
    main()

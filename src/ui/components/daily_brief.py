"""Daily Brief UI Component - ReAct Agent Integration

This component provides the UI for generating AI-powered daily briefs using the ReAct Agent.
"""

import streamlit as st
from datetime import datetime
from typing import Optional

from src.agents.react_agent import ReActAgent, DailyBrief


def render_daily_brief_section():
    """Render the Daily Brief section with persisted brief from database."""
    st.subheader("🤖 AI-Powered Daily Brief")

    # Check if we have a persisted brief from startup_service
    if 'data' in st.session_state and st.session_state.data.get('daily_brief'):
        brief_data = st.session_state.data['daily_brief']
        _display_persisted_brief(brief_data)
    else:
        st.info(
            "📋 Daily briefs are automatically generated in the background by the brief scheduler. "
            "Run `make preload` to generate your first brief."
        )

        st.caption("Briefs are generated when:")
        st.markdown("""
        - Sufficient data is available (emails, calendar, news)
        - Configured time interval has elapsed
        - MCP data has changed since last brief
        """)


def _display_persisted_brief(brief_data: dict):
    """Display a persisted brief from database cache.

    Args:
        brief_data: Dict with brief fields (summary, key_points, action_items, etc.)
    """
    # Condensed banner: timestamp + stats on same line
    weather_info = brief_data.get('weather_info')
    temp_str = 'N/A'
    if weather_info and isinstance(weather_info, dict):
        temp_str = weather_info.get('temperature', 'N/A')

    # Calculate time ago for brief generation
    time_ago_str = ""
    generated_at = brief_data.get('generated_at')
    if generated_at:
        try:
            from datetime import datetime
            gen_time = datetime.fromisoformat(generated_at)
            time_ago = datetime.now() - gen_time
            hours_ago = time_ago.total_seconds() / 3600
            time_ago_str = f" · ✅ Brief: {hours_ago:.1f}h ago"
        except Exception:
            time_ago_str = " · ✅ Brief ready"

    # Compact stats + timestamp on one line
    sources_text = (
        f"📧 {brief_data.get('emails_count', 0)} emails · "
        f"📅 {brief_data.get('calendar_events_count', 0)} events · "
        f"📰 {brief_data.get('news_items_count', 0)} articles · "
        f"🌤️ {temp_str}°F"
        f"{time_ago_str}"
    )
    st.markdown(f"<small style='color: gray;'>{sources_text}</small>", unsafe_allow_html=True)
    st.divider()

    # TLDR (if available) - Show first for quick scanning
    tldr = brief_data.get('tldr', '')
    if tldr:
        st.markdown("### ⚡ TLDR")
        # Split on newlines to render each bullet separately
        bullet_lines = [line.strip() for line in tldr.split('\n') if line.strip()]
        for bullet in bullet_lines:
            st.markdown(bullet)
        st.divider()

    # Summary - Display as bullets (split on ". " to avoid breaking decimals like "0.5")
    summary = brief_data.get('summary', '')
    if summary:
        st.markdown("### 📋 Summary")

        # Split on ". " (period + space) to preserve decimals like "0.5%"
        sentences = [s.strip() for s in summary.split('. ') if s.strip()]

        if sentences:
            for sentence in sentences:
                # Add period back if missing
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                st.markdown(f"• {sentence}")
        else:
            # Fallback: show as single item
            st.markdown(f"• {summary}")

        st.divider()

    # Key Points
    key_points = brief_data.get('key_points', [])
    if key_points:
        st.markdown("### 💡 Key Points")
        for i, point in enumerate(key_points, 1):
            st.markdown(f"{i}. {point}")
        st.divider()

    # Action Items
    action_items = brief_data.get('action_items', [])
    if action_items:
        st.markdown("### ✅ Action Items")
        for i, item in enumerate(action_items, 1):
            st.checkbox(
                item,
                key=f"action_item_persisted_{i}",
                value=False
            )
        st.divider()


def _generate_and_display_brief(max_steps: int, show_thinking: bool):
    """Generate and display the daily brief using ReAct Agent."""

    with st.spinner("🤖 Agent is thinking and gathering information..."):
        try:
            # Initialize ReAct Agent
            agent = ReActAgent(max_steps=max_steps)

            # Execute agent to generate daily brief
            result = agent.run(
                goal="Generate a comprehensive daily brief including emails, calendar events, news, and weather"
            )

            # Check if execution was successful
            if not result.success:
                st.error(f"❌ Agent execution failed: {result.error}")
                return

            # Extract DailyBrief from result
            brief: DailyBrief = result.data

            if brief is None:
                st.error("❌ Agent did not generate a daily brief")
                return

            # Display the brief
            _display_brief(brief, result, show_thinking)

            # Store in session state for later reference
            st.session_state['last_daily_brief'] = brief
            st.session_state['last_brief_timestamp'] = datetime.now()

        except Exception as e:
            st.error(f"❌ Error generating daily brief: {str(e)}")
            st.exception(e)


def _display_brief(brief: DailyBrief, result, show_thinking: bool):
    """Display the generated daily brief."""

    # Success message
    st.success("✅ Daily brief generated successfully!")

    # Statistics
    st.markdown("### 📊 Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📧 Emails", brief.emails_count)

    with col2:
        st.metric("📅 Events", brief.calendar_events_count)

    with col3:
        st.metric("📰 News Items", brief.news_items_count)

    with col4:
        if brief.weather_info:
            temp = brief.weather_info.get('temperature', 'N/A')
            st.metric("🌤️ Weather", f"{temp}°F" if temp != 'N/A' else 'N/A')
        else:
            st.metric("🌤️ Weather", "N/A")

    st.divider()

    # Summary - Display as bullet list for better readability
    st.markdown("### 📋 Summary")

    # Split summary into sentences and display as bullet points
    import re
    sentences = re.split(r'[.!?]+', brief.summary)
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        for sentence in sentences:
            st.markdown(f"- {sentence}")
    else:
        # Fallback: if no sentences detected, show as single bullet
        st.markdown(f"- {brief.summary}")

    st.divider()

    # Key Points
    if brief.key_points:
        st.markdown("### 💡 Key Points")
        for i, point in enumerate(brief.key_points, 1):
            st.markdown(f"{i}. {point}")
        st.divider()

    # Action Items
    if brief.action_items:
        st.markdown("### ✅ Action Items")
        for i, item in enumerate(brief.action_items, 1):
            # Create checkbox for each action item
            checked = st.checkbox(
                item,
                key=f"action_item_{i}",
                value=False
            )
        st.divider()

    # Weather Details
    if brief.weather_info:
        st.markdown("### 🌤️ Weather")
        weather_col1, weather_col2 = st.columns(2)

        with weather_col1:
            st.write(f"**Temperature:** {brief.weather_info.get('temperature', 'N/A')}°F")
            st.write(f"**Feels Like:** {brief.weather_info.get('feels_like', 'N/A')}°F")

        with weather_col2:
            st.write(f"**Conditions:** {brief.weather_info.get('description', 'N/A')}")
            st.write(f"**Humidity:** {brief.weather_info.get('humidity', 'N/A')}%")

        st.divider()

    # Agent Thinking Process (if enabled)
    if show_thinking and hasattr(result, 'metadata') and 'steps' in result.metadata:
        st.markdown("### 🧠 Agent Thinking Process")
        with st.expander("View reasoning steps", expanded=False):
            steps = result.metadata['steps']
            for step in steps:
                st.markdown(f"**Step {step.step_num}**")
                st.write(f"💭 **Thought:** {step.thought}")

                if step.action:
                    st.write(f"⚡ **Action:** `{step.action}`")

                if step.action_input:
                    st.code(str(step.action_input), language="json")

                if step.observation:
                    st.write(f"👁️ **Observation:** {step.observation}")

                st.divider()


def render_previous_brief_section():
    """Render a section showing the previously generated brief."""

    if 'last_daily_brief' not in st.session_state:
        return

    brief: DailyBrief = st.session_state['last_daily_brief']
    timestamp = st.session_state.get('last_brief_timestamp')

    with st.expander("📜 Previous Daily Brief", expanded=False):
        if timestamp:
            time_ago = datetime.now() - timestamp
            hours_ago = time_ago.total_seconds() / 3600
            st.caption(f"Generated {hours_ago:.1f} hours ago")

        # Quick summary
        st.markdown(f"**Summary:** {brief.summary[:200]}..." if len(brief.summary) > 200 else brief.summary)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📧 Emails", brief.emails_count)
        with col2:
            st.metric("📅 Events", brief.calendar_events_count)
        with col3:
            st.metric("📰 News", brief.news_items_count)

        # Action items count
        if brief.action_items:
            st.write(f"**Action Items:** {len(brief.action_items)} items")

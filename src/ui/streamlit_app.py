"""
Daily Minutes - Minimal Streamlit Dashboard

COMPLETELY REWRITTEN FROM SCRATCH for speed and reliability.
This app loads ALL data from database cache - NO network calls, NO heavy services.

Run with: streamlit run streamlit_app.py
"""

import asyncio
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
from src.core.logging import get_logger
from src.services.rag_service import get_rag_service

logger = get_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Daily Minutes",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_async(coro):
    """Run async function synchronously.

    Handles event loop lifecycle properly for Streamlit reruns.
    """
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
    except RuntimeError:
        # Create new loop if none exists or is closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Always run in the current loop (don't retry - coroutines can't be reused)
    return loop.run_until_complete(coro)


@st.cache_resource
def get_db_manager():
    """Get cached database manager (cached across reruns)."""
    from src.database.sqlite_manager import get_db_manager
    db = get_db_manager()
    run_async(db.initialize())
    return db


def format_timestamp(iso_string: Optional[str]) -> str:
    """Format ISO timestamp to human-readable string."""
    if not iso_string:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        # Format as "Dec 28, 2:30 PM"
        return dt.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso_string


def time_ago(timestamp) -> str:
    """Convert timestamp (datetime or ISO string) to 'X hours ago' format."""
    if not timestamp:
        return "Unknown"
    try:
        # Handle datetime objects
        if isinstance(timestamp, datetime):
            dt = timestamp
        else:
            # Handle ISO strings
            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))

        # Make timezone-aware if needed
        if dt.tzinfo is None:
            from datetime import timezone
            now = datetime.now()
        else:
            now = datetime.now(dt.tzinfo)

        diff = now - dt

        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} min ago" if minutes == 1 else f"{minutes} mins ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour ago" if hours == 1 else f"{hours} hours ago"
        else:
            days = int(diff.total_seconds() / 86400)
            return f"{days} day ago" if days == 1 else f"{days} days ago"
    except Exception:
        return str(timestamp) if timestamp else "Unknown"


def safe_get_attr(obj, attr: str, default=None):
    """Safely get attribute from object or dict."""
    if hasattr(obj, attr):
        return getattr(obj, attr, default)
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def load_all_data() -> Dict[str, Any]:
    """Load ALL data from database cache - FAST, no network calls."""
    from src.services.startup_service import get_startup_service

    db = get_db_manager()
    startup_service = get_startup_service(db)
    return run_async(startup_service.load_startup_data(limit=100))


def trigger_refresh():
    """Trigger background refresh of all data sources."""
    from src.services.background_refresh_service import get_background_refresh_service

    refresh_service = get_background_refresh_service()

    # Check if refresh already in progress
    if refresh_service.is_refresh_in_progress():
        st.warning("Refresh already in progress...")
        return

    with st.spinner("Refreshing all data sources..."):
        results = run_async(refresh_service.refresh_all_sources())

    if results:
        success_count = sum(1 for v in results.values() if v)
        st.success(f"Refreshed {success_count}/{len(results)} data sources")

        # Clear cached data to force reload
        if 'data' in st.session_state:
            del st.session_state.data
        st.rerun()
    else:
        st.error("Refresh failed or already in progress")


# ============================================================================
# LOAD DATA ON STARTUP (ONCE PER SESSION)
# ============================================================================

if 'data' not in st.session_state:
    with st.spinner("Loading cached data..."):
        st.session_state.data = load_all_data()

data = st.session_state.data

# CRITICAL: Populate session_state for components that expect specific keys
# Email and Calendar components expect Pydantic models, but DB returns dicts
# Convert cached dicts to EmailMessage objects
from src.core.models import EmailMessage, CalendarEvent

def convert_email_dict_to_model(email_dict):
    """Convert cached email dict to EmailMessage model."""
    try:
        return EmailMessage(**email_dict)
    except Exception as e:
        logger.error(f"Failed to convert email dict: {e}")
        return None

def convert_calendar_dict_to_model(event_dict):
    """Convert cached calendar dict to CalendarEvent model."""
    try:
        return CalendarEvent(**event_dict)
    except Exception as e:
        logger.error(f"Failed to convert calendar dict: {e}")
        return None

# Convert emails from dicts to EmailMessage objects
email_dicts = data.get('emails', [])
if email_dicts:
    email_objects = [convert_email_dict_to_model(e) for e in email_dicts if e]
    email_objects = [e for e in email_objects if e is not None]
    st.session_state.emails = email_objects
else:
    st.session_state.emails = []

# Convert calendar events from dicts to CalendarEvent objects
calendar_dicts = data.get('calendar_events', [])
if calendar_dicts:
    calendar_objects = [convert_calendar_dict_to_model(e) for e in calendar_dicts if e]
    calendar_objects = [e for e in calendar_objects if e is not None]
    st.session_state.calendar_events = calendar_objects
else:
    st.session_state.calendar_events = []

# ============================================================================
# HEADER WITH REFRESH BUTTON
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📰 Daily Minutes")
with col2:
    if st.button("🔄 Refresh All", use_container_width=True):
        trigger_refresh()

# Show data age if available
if data.get('cache_age_hours') is not None:
    age = data['cache_age_hours']
    if age < 1:
        st.caption(f"📊 Data cached {int(age * 60)} minutes ago")
    else:
        st.caption(f"📊 Data cached {age:.1f} hours ago")

# Show error if data loading failed
if data.get('error'):
    st.error(f"Error loading data: {data['error']}")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📰 Overview",
    "📑 News",
    "🌤️ Weather",
    "📧 Email",
    "📅 Calendar",
    "📊 Analytics",
    "🔧 Orchestrator",
    "🏥 System Health",
    "⚙️ Settings",
    "💬 Q&A Assistant"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.header("Overview")

    # AI Brief section - Use existing daily_brief component
    from src.ui.components.daily_brief import render_daily_brief_section, render_previous_brief_section

    render_daily_brief_section()

    st.divider()

    # Today's Schedule and Important Items
    # Get today's calendar events
    from src.ui.components.calendar_components import render_daily_view
    calendar_events = data.get('calendar_events', [])
    if calendar_events:
        st.subheader("📅 Today's Schedule")
        render_daily_view(calendar_events)
    else:
        st.subheader("📅 Today's Schedule")
        st.info("No calendar events for today")

    st.divider()

    # Weather Widget with Safety Tips
    st.subheader("🌤️ Weather & Safety")
    weather = data.get('weather_data')

    if weather:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°F")
            if weather.get('feels_like'):
                st.caption(f"Feels like {weather.get('feels_like')}°F")

        with col2:
            st.metric("Conditions", weather.get('description', 'Unknown'))
            if weather.get('humidity'):
                st.caption(f"Humidity: {weather.get('humidity')}%")

        with col3:
            # Clothing recommendation based on temperature
            temp = weather.get('temperature', 0)
            if temp < 0:
                st.info("🧥 Heavy winter coat")
            elif temp < 10:
                st.info("🧥 Warm jacket")
            elif temp < 20:
                st.info("👕 Light jacket")
            elif temp < 30:
                st.info("👕 Light clothing")
            else:
                st.info("🌡️ Stay hydrated")

            # Umbrella recommendation
            desc_lower = weather.get('description', '').lower()
            if any(word in desc_lower for word in ['rain', 'drizzle', 'shower']):
                st.caption("☂️ Bring umbrella")
            elif any(word in desc_lower for word in ['cloud', 'overcast']):
                st.caption("☁️ Consider umbrella")
    else:
        st.info("No weather data available")

    st.divider()

    # Important/Unread Emails with Details
    st.subheader("📧 Important Emails")
    emails = data.get('emails', [])
    important_emails = [e for e in emails if not e.get('is_read', False) and e.get('importance_score', 0) >= 0.7][:5]

    if important_emails:
        from src.ui.formatters.email_formatter import get_email_formatter
        email_formatter = get_email_formatter()
        for email in important_emails:
            with st.container():
                # Header row
                col1, col2 = st.columns([4, 1])
                with col1:
                    importance = "🔴" if email.get('importance_score', 0) >= 0.9 else "🟠"
                    read_indicator = "✉️" if not email.get('is_read', False) else "📭"
                    st.markdown(f"{read_indicator} {importance} **{email.get('subject', 'No subject')}**")
                    st.caption(f"From: {email.get('sender', 'Unknown')} | {email_formatter.format_relative_time(email.get('received_at'))}")
                with col2:
                    if email.get('has_action_items'):
                        st.caption(f"📋 {len(email.get('action_items', []))} items")

                # Show ONLY TLDR/analysis/todos - NO body previews
                # Priority: ai_summary > snippet > action_items (NO fallback to body)

                # 1. Show AI summary if available (BEST) - show FULL summary, no truncation!
                if email.get('ai_summary'):
                    st.info(f"💡 {email.get('ai_summary')}")

                # 2. Show snippet if available (GOOD)
                elif email.get('snippet'):
                    snippet_text = email.get('snippet')
                    # Show first 150 chars if long
                    if len(snippet_text) > 150:
                        snippet_text = snippet_text[:147] + "..."
                    st.caption(f"💬 {snippet_text}")

                # 3. Show action items summary (FALLBACK)
                if email.get('has_action_items') and email.get('action_items'):
                    action_count = len(email.get('action_items', []))
                    with st.expander(f"📋 {action_count} action item{'s' if action_count > 1 else ''}"):
                        for item in email.get('action_items', []):
                            st.markdown(f"• {item}")

                # NO fallback to body preview - user wants ONLY tldr/analysis/todos

                st.divider()
    else:
        st.info("No important unread emails")

    st.divider()

    # Compact stats summary at bottom
    weather_data = data.get('weather_data')
    temp_str = f"{weather_data.get('temperature', 'N/A')}°F" if weather_data and isinstance(weather_data, dict) else "N/A"

    last_refresh = data.get('last_refresh', {})
    refresh_str = time_ago(last_refresh.get('news')) if last_refresh.get('news') else "Never"

    stats_text = (
        f"📰 {len(data.get('articles', []))} articles · "
        f"📧 {len([e for e in emails if not e.get('is_read', False)])} unread · "
        f"📅 {len(calendar_events)} events · "
        f"🌤️ {temp_str} · "
        f"🔄 {refresh_str}"
    )
    st.markdown(f"<small style='color: gray;'>{stats_text}</small>", unsafe_allow_html=True)

# ============================================================================
# TAB 2: NEWS
# ============================================================================

with tab2:
    st.header("News Articles")

    # CRITICAL: Show last refresh time from DATABASE
    last_refresh = data.get('last_refresh', {}).get('news')
    if last_refresh:
        st.info(f"📰 **Last Refreshed from Database**: {time_ago(last_refresh)} ({last_refresh})")
    else:
        st.warning("⚠️ **Never refreshed** - Click 'Refresh All' to fetch news data")

    articles = data.get('articles', [])

    if not articles:
        st.info("No news articles available. Click 'Refresh All' to fetch news.")
    else:

        # Filter options
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sources = sorted(list(set(safe_get_attr(a, 'source_name', 'Unknown') for a in articles)))
            source_filter = st.selectbox(
                "Source",
                ["All"] + sources,
                key="news_source_filter"
            )
        with col2:
            priorities = sorted(list(set(str(safe_get_attr(a, 'priority', 'medium')) for a in articles)))
            priority_filter = st.selectbox(
                "Priority",
                ["All"] + priorities,
                key="news_priority_filter"
            )
        with col3:
            limit = st.number_input("Show", min_value=5, max_value=100, value=20, step=5)

        # Apply filters
        filtered_articles = articles
        if source_filter != "All":
            filtered_articles = [a for a in filtered_articles if safe_get_attr(a, 'source_name') == source_filter]
        if priority_filter != "All":
            filtered_articles = [a for a in filtered_articles if str(safe_get_attr(a, 'priority')) == priority_filter]

        # Display articles
        st.markdown(f"Showing {min(limit, len(filtered_articles))} of {len(filtered_articles)} articles")

        for article in filtered_articles[:limit]:
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Priority emoji
                    priority = str(safe_get_attr(article, 'priority', 'medium'))
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "urgent": "🔴"}
                    emoji = priority_emoji.get(priority, "⚫")

                    # Title (with link if available)
                    title = safe_get_attr(article, 'title', 'Untitled')
                    url = safe_get_attr(article, 'url')
                    if url:
                        st.markdown(f"### {emoji} [{title}]({url})")
                    else:
                        st.markdown(f"### {emoji} {title}")

                    # Metadata
                    source = safe_get_attr(article, 'source_name', 'Unknown')
                    published = safe_get_attr(article, 'published_at') or safe_get_attr(article, 'fetched_at')
                    time_str = time_ago(published) if published else 'Unknown time'

                    st.caption(f"📌 {source} • {time_str}")

                    # AI-generated TLDR/summary if available, otherwise description
                    ai_summary = safe_get_attr(article, 'ai_summary')
                    tldr = safe_get_attr(article, 'tldr')
                    description = safe_get_attr(article, 'description') or safe_get_attr(article, 'content', '')

                    # Priority: AI summary > TLDR > description
                    if ai_summary:
                        st.info(f"💡 **AI Analysis:** {ai_summary}")
                    elif tldr:
                        st.info(f"⚡ **TLDR:** {tldr}")
                    elif description:
                        # Fallback: show shortened description
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.caption(description)

                    # Show key learnings if available
                    key_learnings = safe_get_attr(article, 'key_learnings', [])
                    if key_learnings and len(key_learnings) > 0:
                        st.markdown("**🔑 Key Learnings:**")
                        for learning in key_learnings:
                            st.markdown(f"- {learning}")

                with col2:
                    # Relevance score
                    relevance = safe_get_attr(article, 'relevance_score', 0.0)
                    st.metric("Relevance", f"{relevance:.2f}")

                st.divider()

# ============================================================================
# TAB 3: WEATHER
# ============================================================================

with tab3:
    st.header("Weather")

    # CRITICAL: Show last refresh time from DATABASE
    last_refresh = data.get('last_refresh', {}).get('weather')
    if last_refresh:
        st.info(f"🌤️ **Last Refreshed from Database**: {time_ago(last_refresh)} ({last_refresh})")
    else:
        st.warning("⚠️ **Never refreshed** - Click 'Refresh All' to fetch weather data")

    weather = data.get('weather_data')

    if not weather:
        st.info("No weather data available. Click 'Refresh All' to fetch weather.")
    else:

        # Current weather
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°F")
            if weather.get('feels_like'):
                st.caption(f"Feels like {weather.get('feels_like')}°F")

        with col2:
            st.metric("Conditions", weather.get('description', 'Unknown'))

        with col3:
            if weather.get('humidity'):
                st.metric("Humidity", f"{weather.get('humidity')}%")

        with col4:
            if weather.get('wind_speed'):
                st.metric("Wind", f"{weather.get('wind_speed')} mph")

        st.divider()

        # Additional details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Details")
            if weather.get('pressure'):
                st.write(f"**Pressure:** {weather.get('pressure')} hPa")
            if weather.get('visibility'):
                st.write(f"**Visibility:** {weather.get('visibility')} m")
            if weather.get('uv_index'):
                st.write(f"**UV Index:** {weather.get('uv_index')}")

        with col2:
            st.subheader("Sun Times")
            if weather.get('sunrise'):
                sunrise = format_timestamp(weather.get('sunrise'))
                st.write(f"**Sunrise:** {sunrise}")
            if weather.get('sunset'):
                sunset = format_timestamp(weather.get('sunset'))
                st.write(f"**Sunset:** {sunset}")

        # Forecast if available
        if weather.get('forecast'):
            st.divider()
            st.subheader("Forecast")

            forecast_data = weather.get('forecast', [])
            if isinstance(forecast_data, list) and forecast_data:
                # Display forecast as cards
                cols = st.columns(min(len(forecast_data), 5))
                for idx, forecast in enumerate(forecast_data[:5]):
                    with cols[idx]:
                        date = forecast.get('date', 'Unknown')
                        temp = forecast.get('temperature', 'N/A')
                        desc = forecast.get('description', 'Unknown')
                        st.markdown(f"**{date}**")
                        st.markdown(f"{temp}°F")
                        st.caption(desc)

        # Weather Recommendations
        st.divider()
        st.subheader("💡 Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            # Clothing recommendations based on temperature
            temp = weather.get('temperature', 0)
            if temp < 0:
                st.info("🧥 **Wear:** Heavy winter coat, gloves, scarf")
            elif temp < 10:
                st.info("🧥 **Wear:** Warm jacket or coat")
            elif temp < 20:
                st.info("👕 **Wear:** Light jacket or sweater")
            elif temp < 30:
                st.info("👕 **Wear:** Light, breathable clothing")
            else:
                st.warning("🌡️ **Wear:** Very light clothing, stay hydrated")

            # Umbrella recommendation
            desc_lower = weather.get('description', '').lower()
            if any(word in desc_lower for word in ['rain', 'drizzle', 'shower']):
                st.info("☂️ **Bring an umbrella** - Rain expected")
            elif any(word in desc_lower for word in ['cloud', 'overcast']):
                st.info("☁️ **Consider an umbrella** - Cloudy conditions")

        with rec_col2:
            # Safety tips for severe weather
            desc_lower = weather.get('description', '').lower()
            if any(word in desc_lower for word in ['storm', 'thunder', 'lightning']):
                st.warning("⚠️ **Severe Weather Alert**")
                st.markdown("""
                - Stay indoors if possible
                - Avoid open areas and tall objects
                - Unplug electrical devices
                - Keep emergency supplies ready
                """)
            elif any(word in desc_lower for word in ['snow', 'blizzard', 'ice']):
                st.warning("❄️ **Winter Weather Advisory**")
                st.markdown("""
                - Drive carefully or avoid travel
                - Dress in layers
                - Watch for ice on walkways
                """)
            elif temp > 35:
                st.warning("🌡️ **Heat Advisory**")
                st.markdown("""
                - Stay hydrated
                - Avoid prolonged sun exposure
                - Check on elderly neighbors
                - Never leave pets/children in cars
                """)

# ============================================================================
# TAB 4: EMAIL
# ============================================================================

with tab4:
    # CRITICAL: Show last refresh time from DATABASE
    last_refresh = data.get('last_refresh', {}).get('email')
    if last_refresh:
        st.info(f"📧 **Last Refreshed from Database**: {time_ago(last_refresh)} ({last_refresh})")
    else:
        st.warning("⚠️ **Never refreshed** - Click 'Refresh All' to fetch email data")

    from src.ui.components.email_components import render_email_tab
    render_email_tab()

# ============================================================================
# TAB 5: CALENDAR
# ============================================================================

with tab5:
    # CRITICAL: Show last refresh time from DATABASE
    last_refresh = data.get('last_refresh', {}).get('calendar')
    if last_refresh:
        st.info(f"📅 **Last Refreshed from Database**: {time_ago(last_refresh)} ({last_refresh})")
    else:
        st.warning("⚠️ **Never refreshed** - Click 'Refresh All' to fetch calendar data")

    from src.ui.components.calendar_components import render_calendar_tab
    render_calendar_tab()

# ============================================================================
# TAB 6: ANALYTICS
# ============================================================================

with tab6:
    st.header("📊 News Analytics Dashboard")

    articles = data.get('articles', [])

    if not articles:
        st.info("No articles available for analytics.")
    else:
        st.write("Visual analysis of the fetched news articles to identify patterns and trends.")

        # Prepare data
        df_data = []
        for a in articles:
            df_data.append({
                'title': safe_get_attr(a, 'title', '')[:50],
                'source': safe_get_attr(a, 'source_name', 'Unknown'),
                'published': safe_get_attr(a, 'published_at'),
                'relevance': safe_get_attr(a, 'relevance_score', 0.0),
                'priority': str(safe_get_attr(a, 'priority', 'medium'))
            })

        df = pd.DataFrame(df_data)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            st.metric("Unique Sources", df['source'].nunique())
        with col3:
            avg_relevance = df['relevance'].mean()
            st.metric("Avg Relevance", f"{avg_relevance:.2f}")

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Articles by source
            st.subheader("Articles by Source")
            source_counts = df['source'].value_counts()
            fig = px.bar(x=source_counts.index, y=source_counts.values,
                        labels={'x': 'Source', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Articles by priority
            st.subheader("Articles by Priority")
            priority_counts = df['priority'].value_counts()
            fig = px.pie(values=priority_counts.values, names=priority_counts.index)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 7: ORCHESTRATOR
# ============================================================================

with tab7:
    st.header("🔧 LangGraph Orchestrator")

    # Initialize session state for orchestrator
    if 'is_orchestrating' not in st.session_state:
        st.session_state.is_orchestrating = False
    if 'orchestration_result' not in st.session_state:
        st.session_state.orchestration_result = None

    # Try to import orchestrator service
    orchestrator_available = False
    try:
        from src.services.langgraph_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        orchestrator_available = True
    except ImportError:
        orchestrator = None

    if not orchestrator_available:
        st.warning("⚠️ LangGraph Orchestrator service not available. Check if langgraph_orchestrator.py exists.")
    else:
        st.write("Run complex AI workflows using LangGraph orchestration.")

        # Workflow input
        workflow_request = st.text_area(
            "Workflow request",
            placeholder="e.g., 'Give me a comprehensive analysis of today's tech news'",
            height=100,
            key="orchestrator_input",
            disabled=st.session_state.is_orchestrating
        )

        # Preferences
        with st.expander("⚙️ Workflow Preferences", expanded=False):
            sources = st.multiselect(
                "Sources",
                ["hackernews", "rss", "weather"],
                default=["hackernews"],
                key="orch_sources",
                disabled=st.session_state.is_orchestrating
            )

            topics = st.text_input(
                "Topics (comma-separated)",
                value="AI, technology",
                key="orch_topics",
                disabled=st.session_state.is_orchestrating
            )

            model_to_use = "llama3.2:3b"
            st.info(f"Using model: {model_to_use}")

        # Button controls
        col1, col2 = st.columns([1, 3])

        with col1:
            if not st.session_state.is_orchestrating:
                if st.button("🚀 Run Orchestration", type="primary", key="orch_button"):
                    if workflow_request:
                        st.session_state.is_orchestrating = True
                        st.rerun()
                    else:
                        st.warning("Please enter a workflow request")
            else:
                if st.button("⏹️ Cancel", type="secondary", key="cancel_orch"):
                    st.session_state.is_orchestrating = False
                    st.rerun()

        # Run orchestration if flagged
        if st.session_state.is_orchestrating and workflow_request:
            with st.status("Running orchestration...", expanded=True) as status:
                try:
                    status.update(label="🚀 Starting orchestration...")

                    status.update(label="📊 Analyzing request...")

                    preferences = {
                        "sources": sources,
                        "topics": [t.strip() for t in topics.split(",")],
                        "model": model_to_use
                    }

                    status.update(label="🔍 Gathering data from sources...")

                    # Run the actual workflow
                    async def run_workflow():
                        try:
                            return await orchestrator.run_orchestrated_workflow(workflow_request, preferences)
                        except Exception as e:
                            return {"success": False, "error": str(e)}

                    result = run_async(run_workflow())

                    status.update(label="✨ Generating insights...")

                    # Store result and complete
                    st.session_state.orchestration_result = result
                    st.session_state.is_orchestrating = False

                    status.update(label="✅ Orchestration complete!", state="complete")
                    st.rerun()

                except Exception as e:
                    st.session_state.is_orchestrating = False
                    status.update(label=f"❌ Orchestration failed: {str(e)}", state="error")

        # Display results if available
        if st.session_state.orchestration_result and not st.session_state.is_orchestrating:
            result = st.session_state.orchestration_result

            if result.get("success"):
                st.success("✅ Workflow completed successfully!")

                # Display output
                if result.get("output"):
                    st.subheader("📋 Results")
                    st.markdown(result["output"])

                # Metrics
                st.subheader("📊 Execution Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    exec_time = result.get('execution_time_ms', 0)
                    st.metric("⏱️ Time", f"{exec_time:.0f}ms" if exec_time < 1000 else f"{exec_time/1000:.1f}s")
                with col2:
                    st.metric("📝 Steps", len(result.get('steps_completed', [])))
                with col3:
                    st.metric("📚 Sources", len(result.get('sources', [])))
                with col4:
                    st.metric("🧠 Model", model_to_use)

                # Clear result button
                if st.button("🧹 Clear Results", key="clear_orch_results"):
                    st.session_state.orchestration_result = None
                    st.rerun()
            else:
                st.error(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")

# ============================================================================
# TAB 8: SYSTEM HEALTH
# ============================================================================

with tab8:
    st.header("🏥 System Health Dashboard")
    st.write("Monitor the health and performance of all system components.")

    st.subheader("🔍 MCP Data Source Diagnostics")
    from src.ui.components.system_diagnostics import render_system_diagnostics
    render_system_diagnostics()

# ============================================================================
# TAB 9: SETTINGS
# ============================================================================

with tab9:
    st.header("⚙️ Settings")
    st.write("Configure your Daily Minutes application.")

    # Database settings
    with st.expander("🗄️ Database", expanded=True):
        db = get_db_manager()
        db_path = run_async(db.get_setting('database_path', 'data/daily_minutes.db'))
        st.code(f"Database: {db_path}")

        # Show cache info
        cache_age = data.get('cache_age_hours')
        if cache_age is not None:
            st.metric("Cache Age", f"{cache_age:.1f} hours")

    # Refresh settings
    with st.expander("🔄 Refresh Intervals", expanded=False):
        st.subheader("Refresh Interval Configuration")
        st.caption("Configure how often each data source is automatically refreshed in the background")

        try:
            from src.services.settings_manager import get_settings_manager
            settings_mgr = get_settings_manager()
            refresh_settings = settings_mgr.get_refresh_intervals()

            # Display current refresh intervals in a table
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📰 News")
                news_interval = st.number_input(
                    "News Refresh (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=refresh_settings["news"],
                    help="How often to refresh news articles",
                    key="refresh_news_interval"
                )

                st.markdown("#### 📧 Email")
                email_interval = st.number_input(
                    "Email Refresh (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=refresh_settings["email"],
                    help="How often to check for new emails",
                    key="refresh_email_interval"
                )

            with col2:
                st.markdown("#### 🌤️ Weather")
                weather_interval = st.number_input(
                    "Weather Refresh (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=refresh_settings["weather"],
                    help="How often to update weather information",
                    key="refresh_weather_interval"
                )

                st.markdown("#### 📅 Calendar")
                calendar_interval = st.number_input(
                    "Calendar Refresh (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=refresh_settings["calendar"],
                    help="How often to sync calendar events",
                    key="refresh_calendar_interval"
                )

            # Save button
            if st.button("💾 Save Refresh Intervals", key="save_refresh_intervals"):
                # Validate all intervals
                all_valid = True
                for interval_name, interval_value in [
                    ("News", news_interval),
                    ("Weather", weather_interval),
                    ("Email", email_interval),
                    ("Calendar", calendar_interval)
                ]:
                    validation = settings_mgr.validate_refresh_interval(interval_value)
                    if not validation["valid"]:
                        for error in validation["errors"]:
                            st.error(f"❌ {interval_name}: {error}")
                        all_valid = False

                if all_valid:
                    settings_mgr.update_refresh_intervals(
                        news_interval=news_interval,
                        weather_interval=weather_interval,
                        email_interval=email_interval,
                        calendar_interval=calendar_interval
                    )
                    st.success("✅ Refresh intervals saved successfully!")

            st.divider()

            # Show last refresh times
            st.caption("**Last Refreshed:**")
            last_refresh = data.get('last_refresh', {})
            if last_refresh:
                lcol1, lcol2, lcol3, lcol4 = st.columns(4)
                with lcol1:
                    st.write(f"📰 News: {time_ago(last_refresh.get('news', 'Never'))}")
                with lcol2:
                    st.write(f"🌤️ Weather: {time_ago(last_refresh.get('weather', 'Never'))}")
                with lcol3:
                    st.write(f"📧 Email: {time_ago(last_refresh.get('email', 'Never'))}")
                with lcol4:
                    st.write(f"📅 Calendar: {time_ago(last_refresh.get('calendar', 'Never'))}")
            else:
                st.write("No data sources have been refreshed yet")

        except ImportError:
            st.error("Settings manager not available")

    # AI Models Section
    with st.expander("🤖 AI Models", expanded=False):
        st.subheader("AI Model Configuration")

        # Check Ollama status
        async def check_ollama_status():
            try:
                from src.services.ollama_service import get_ollama_service
                ollama = get_ollama_service()
                return await ollama.check_availability()
            except:
                return False

        ollama_status = run_async(check_ollama_status())

        if not ollama_status:
            st.warning("⚠️ Ollama is not running. Start Ollama to use AI features.")
            st.code("ollama serve", language="bash")
        else:
            st.success("✅ Ollama is running")

            # Get installed models
            try:
                from src.services.ollama_service import get_ollama_service
                ollama_service = get_ollama_service()
                installed_models = run_async(ollama_service.list_models())

                if installed_models:
                    st.info(f"📦 {len(installed_models)} models installed")

                    # Get current model from session state or use default
                    current_model = st.session_state.get('selected_model', 'llama3.2')

                    # Model selection dropdown
                    model_names = [m.split(':')[0] for m in installed_models]
                    default_index = model_names.index(current_model) if current_model in model_names else 0

                    selected_model = st.selectbox(
                        "Active Model",
                        options=model_names,
                        index=default_index,
                        help="Select the model for text generation and analysis",
                        key="settings_selected_model"
                    )

                    # Temperature slider
                    current_temp = st.session_state.get('temperature_slider', 0.7)
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=current_temp,
                        step=0.1,
                        help="Controls randomness: 0 = focused, 2 = creative",
                        key="settings_temperature"
                    )

                    # Save button
                    if st.button("💾 Save AI Settings", key="save_ai_settings"):
                        st.session_state.selected_model = selected_model
                        st.session_state.temperature_slider = temperature
                        st.success(f"✅ AI settings saved! Using {selected_model} at temp {temperature}")

                else:
                    st.warning("⚠️ No models installed")
                    st.info("Install a model using: `ollama pull llama3.2`")

            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")

    # News Settings Section
    with st.expander("📰 News Settings", expanded=False):
        st.subheader("News Fetching Configuration")

        try:
            from src.services.settings_manager import get_settings_manager
            settings_mgr = get_settings_manager()
            news_settings = settings_mgr.get_news_settings()

            col1, col2 = st.columns(2)
            with col1:
                max_articles = st.number_input(
                    "Maximum Total Articles",
                    min_value=1,
                    max_value=100,
                    value=news_settings["max_articles"],
                    help="Total number of articles to fetch across all sources"
                )

                max_per_source = st.number_input(
                    "Max Articles Per Source (Base)",
                    min_value=1,
                    max_value=50,
                    value=news_settings["max_per_source_base"],
                    help="Base limit per source"
                )

            with col2:
                content_threads = st.number_input(
                    "Content Fetching Threads",
                    min_value=1,
                    max_value=20,
                    value=news_settings["content_threads"],
                    help="Number of parallel threads for fetching article content"
                )

                st.metric(
                    "Calculated Per-Source Limit",
                    f"{min(max_articles // 2, max_per_source)} articles",
                    help="Actual limit based on 2 sources (HackerNews + RSS)"
                )

            if st.button("💾 Save News Settings", key="save_news"):
                validation = settings_mgr.validate_news_settings(
                    max_articles, max_per_source, content_threads
                )

                if validation["valid"]:
                    settings_mgr.update_news_settings(
                        max_articles=max_articles,
                        max_per_source_base=max_per_source,
                        content_threads=content_threads
                    )
                    st.success("✅ News settings saved successfully!")
                else:
                    for error in validation["errors"]:
                        st.error(f"❌ {error}")

        except ImportError:
            st.warning("⚠️ Settings manager not available")

    # News Sources Section
    with st.expander("📡 News Sources", expanded=False):
        st.subheader("Manage News Sources")

        try:
            from src.services.settings_manager import get_settings_manager
            settings_mgr = get_settings_manager()
            sources = settings_mgr.get_news_sources_config()

            for source in sources:
                source_type = source["type"]
                enabled = source.get("enabled", True)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{source_type.title()}**")
                with col2:
                    new_enabled = st.checkbox(
                        "Enabled",
                        value=enabled,
                        key=f"enable_{source_type}"
                    )
                    if new_enabled != enabled:
                        settings_mgr.toggle_source_enabled(source_type, new_enabled)
                        st.rerun()

                # RSS Feeds Management
                if source_type == "rss" and enabled:
                    st.write("**RSS Feeds:**")

                    feeds = source.get("feeds", [])
                    for feed in feeds:
                        col1, col2, col3 = st.columns([3, 4, 1])
                        with col1:
                            st.write(feed["name"])
                        with col2:
                            st.code(feed["url"], language=None)
                        with col3:
                            if st.button("🗑️", key=f"remove_{feed['name']}"):
                                settings_mgr.remove_rss_feed(feed["name"])
                                st.success(f"Removed {feed['name']}")
                                st.rerun()

                    # Add new RSS feed
                    st.write("**Add New RSS Feed:**")
                    col1, col2, col3 = st.columns([2, 3, 1])
                    with col1:
                        new_feed_name = st.text_input("Feed Name", key="new_feed_name")
                    with col2:
                        new_feed_url = st.text_input("Feed URL", key="new_feed_url", placeholder="https://example.com/feed")
                    with col3:
                        st.write("")
                        st.write("")
                        if st.button("➕ Add"):
                            if not new_feed_name or not new_feed_url:
                                st.error("Please provide both name and URL")
                            elif not settings_mgr.validate_rss_url(new_feed_url):
                                st.error("Invalid URL format")
                            else:
                                settings_mgr.add_rss_feed(new_feed_name, new_feed_url)
                                st.success(f"✅ Added {new_feed_name}")
                                st.rerun()

        except ImportError:
            st.warning("⚠️ Settings manager not available")

    # Credential Management Section
    with st.expander("🔐 Credential Management", expanded=False):
        st.subheader("OAuth Credentials")
        st.write("Manage encrypted OAuth credentials for Gmail, Google Calendar, and other services.")

        from src.services.credential_service import CredentialService
        from src.core.config_manager import get_config_manager

        # Initialize credential service (async)
        async def init_and_list_credentials():
            """Initialize credential service and list credentials."""
            service = CredentialService()
            await service.initialize()
            return await service.list_credentials()

        async def delete_credential_async(cred_id: str):
            """Delete a credential."""
            service = CredentialService()
            await service.initialize()
            await service.remove_credential(cred_id)

        # Get stored credentials
        try:
            credentials = run_async(init_and_list_credentials())

            if credentials:
                st.write(f"**Connected Services** ({len(credentials)})")
                st.divider()

                for cred in credentials:
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

                    with col1:
                        service_icons = {
                            "gmail": "📧",
                            "google_calendar": "📅",
                            "outlook": "📬"
                        }
                        icon = service_icons.get(cred.get("service_type", ""), "🔐")
                        st.write(f"{icon} **{cred.get('service_type', 'Unknown').title()}**")
                        st.caption(f"Account: {cred.get('account_email', 'N/A')}")

                    with col2:
                        status = cred.get("status", "unknown")
                        status_colors = {
                            "active": "🟢",
                            "expired": "🟡",
                            "revoked": "🔴"
                        }
                        status_icon = status_colors.get(status, "⚪")
                        st.caption(f"Status: {status_icon} {status.capitalize()}")

                        created_at = cred.get("created_at", "")
                        if created_at:
                            st.caption(f"Added: {created_at[:10]}")

                    with col3:
                        # Edit button (placeholder for future implementation)
                        st.button("✏️", key=f"edit_{cred.get('id')}", disabled=True,
                                 help="Edit functionality coming soon")

                    with col4:
                        # Remove button
                        if st.button("🗑️", key=f"remove_{cred.get('id')}",
                                   help="Remove this credential"):
                            try:
                                run_async(delete_credential_async(cred.get("id")))
                                st.success(f"✅ Removed {cred.get('service_type')} credential")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to remove credential: {str(e)}")

                    st.divider()
            else:
                st.info("📭 No credentials stored yet")

        except Exception as e:
            st.error(f"❌ Error loading credentials: {str(e)}")
            st.caption("Make sure the credential service is properly initialized")

        # Add New Credential Section
        st.subheader("Add New Credential")
        st.info("""
        **To add OAuth credentials:**
        1. **For Testing**: Use the testing tools below (mock OAuth servers)
        2. **For Production**: Implement OAuth flow in your application

        **Supported Services:**
        - 📧 Gmail (OAuth 2.0)
        - 📅 Google Calendar (OAuth 2.0)
        - 📬 Outlook (OAuth 2.0)
        """)

        st.caption("**Note:** Full OAuth flow UI is not yet implemented. Use the mock servers for testing or implement your own OAuth flow.")

# ============================================================================
# TAB 10: Q&A ASSISTANT
# ============================================================================

with tab10:
    st.header("💬 Q&A Assistant")

    st.markdown("""
    Use semantic search to find similar articles in the RAG database,
    or ask questions and get AI-powered answers based on stored content.
    """)

    # Initialize RAG service
    try:
        rag = get_rag_service()
        rag_available = True
    except Exception as e:
        st.error(f"❌ RAG service unavailable: {e}")
        rag_available = False

    if rag_available:
        # Get RAG statistics
        with st.expander("📊 RAG Database Statistics", expanded=False):
            try:
                stats = rag.get_statistics()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("Collection", stats.get('collection_name', 'N/A'))
                with col3:
                    sources_count = len(stats.get('sources', {}))
                    st.metric("Unique Sources", sources_count)

                if stats.get('sources'):
                    st.write("**Sources:**", ", ".join(stats['sources'].keys()))
            except Exception as e:
                st.warning(f"Could not load statistics: {e}")

        st.markdown("---")

        # Section 1: Semantic Search
        st.subheader("🔍 Semantic Search")
        st.caption("Find articles similar to your query using vector similarity")

        search_query = st.text_input(
            "Search query",
            placeholder="e.g., artificial intelligence, machine learning, security",
            key="semantic_search_query"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            max_results = st.number_input("Max results", min_value=1, max_value=20, value=5)
        with col2:
            # Check if search is in progress
            search_disabled = st.session_state.get('search_in_progress', False)

            if st.button("🔍 Search", type="primary", key="search_button", disabled=search_disabled):
                if search_query:
                    st.session_state.search_in_progress = True
                    st.rerun()  # Rerun to disable button
                else:
                    st.warning("Please enter a search query")

        # Execute search if flagged
        if st.session_state.get('search_in_progress', False):
            with st.spinner("Searching RAG database..."):
                try:
                    results = run_async(rag.search_articles(
                        query=search_query,
                        max_results=max_results
                    ))

                    st.session_state.search_results = results
                    st.session_state.search_in_progress = False
                    logger.info("semantic_search_completed",
                              query=search_query,
                              results_count=len(results))
                    st.success(f"Found {len(results)} results!")
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    logger.error("semantic_search_failed", error=str(e), exc_info=True)
                    st.session_state.search_in_progress = False

        # Display search results
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.markdown(f"**Found {len(st.session_state.search_results)} results:**")

            for i, result in enumerate(st.session_state.search_results, 1):
                with st.expander(f"{i}. {result.metadata.get('title', 'Untitled')} (Similarity: {result.similarity:.2%})"):
                    # Metadata
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    with meta_col1:
                        st.caption(f"📰 Source: {result.metadata.get('source_name', 'Unknown')}")
                    with meta_col2:
                        st.caption(f"📊 Similarity: {result.similarity:.2%}")
                    with meta_col3:
                        priority = result.metadata.get('priority', 'N/A')
                        st.caption(f"🎯 Priority: {priority}")

                    # Content preview
                    st.markdown("**Content:**")
                    st.write(result.content[:500] + "..." if len(result.content) > 500 else result.content)

                    # Link if available
                    if result.metadata.get('url'):
                        st.markdown(f"[🔗 Read Full Article]({result.metadata['url']})")

        st.markdown("---")

        # Section 2: Question Answering
        st.subheader("❓ Question Answering")
        st.caption("Ask questions and get AI-powered answers based on RAG context")

        question = st.text_area(
            "Your question",
            placeholder="e.g., What are the latest developments in AI safety?",
            height=100,
            key="qa_question"
        )

        # Check if Q/A is in progress
        qa_disabled = st.session_state.get('qa_in_progress', False)

        if st.button("💬 Ask Question", type="primary", key="ask_button", disabled=qa_disabled):
            if question:
                st.session_state.qa_in_progress = True
                st.rerun()  # Rerun to disable button
            else:
                st.warning("Please enter a question")

        # Execute Q/A if flagged
        if st.session_state.get('qa_in_progress', False):
            with st.spinner("Searching context and generating answer..."):
                try:
                    answer_result = run_async(rag.answer_with_context(
                        question=question,
                        system_prompt="You are a helpful news analyst. Answer questions based on the provided news articles context."
                    ))

                    st.session_state.qa_result = answer_result
                    st.session_state.qa_in_progress = False
                    logger.info("question_answered", question=question[:50])
                    st.success("Answer generated!")
                except Exception as e:
                    st.error(f"Question answering failed: {e}")
                    logger.error("qa_failed", error=str(e), exc_info=True)
                    st.session_state.qa_in_progress = False

        # Display Q&A result
        if 'qa_result' in st.session_state and st.session_state.qa_result:
            result = st.session_state.qa_result

            st.markdown("### 💡 Answer:")
            st.success(result.get('answer', 'No answer generated'))

            # Show sources if available
            if result.get('sources'):
                with st.expander(f"📚 Sources ({len(result['sources'])} documents)", expanded=True):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f"**{i}. {source.get('title', 'Untitled')}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Source: {source.get('source', 'Unknown')}")
                        with col2:
                            st.caption(f"Similarity: {source.get('similarity', 0):.2%}")

                        if source.get('url'):
                            st.markdown(f"[🔗 Read Article]({source['url']})")
                        st.markdown("---")

            # Show metadata
            with st.expander("🔧 Metadata", expanded=False):
                st.json({
                    "context_used": result.get('context_used', False),
                    "documents_retrieved": result.get('documents_retrieved', 0)
                })
    else:
        st.info("""
        **RAG service is not available.**

        To use Q&A features:
        1. Ensure Ollama is running: `ollama serve`
        2. Pull the embedding model: `ollama pull nomic-embed-text`
        3. Add some articles to the RAG database
        4. Reload this page
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Daily Minutes Dashboard • Data loaded from cache")

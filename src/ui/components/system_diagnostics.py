"""System Diagnostics UI Component - Show refresh status for all data sources.

This component provides a visual diagnostics panel showing:
- Status of all data sources (news, weather, email, calendar, brief)
- Last refresh time
- Success/error status with visual indicators
- Item counts
- UI polling metrics
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any
from src.services.system_status_service import get_system_status_service, RefreshStatus


def render_system_diagnostics():
    """Render the system diagnostics panel showing all source statuses."""

    st.subheader("System Diagnostics")

    # Get status service
    status_service = get_system_status_service()

    # Get all source statuses
    all_statuses = status_service.get_all_source_status()
    summary = status_service.get_summary()
    ui_status = status_service.get_ui_status()

    # Overall health summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sources", summary['total_sources'])

    with col2:
        fresh_pct = (summary['fresh'] / summary['total_sources'] * 100) if summary['total_sources'] > 0 else 0
        st.metric("Fresh", f"{summary['fresh']} ({fresh_pct:.0f}%)")

    with col3:
        st.metric("Stale", summary['stale'], delta=None if summary['stale'] == 0 else f"{summary['stale']} warning")

    with col4:
        st.metric("Errors", summary['errors'], delta=None if summary['errors'] == 0 else f"{summary['errors']} critical")

    st.divider()

    # UI Polling Status
    st.subheader("UI Activity")

    col1, col2, col3 = st.columns(3)

    with col1:
        if ui_status.last_poll_time:
            time_ago = _format_time_ago(ui_status.last_poll_time)
            st.metric("Last UI Poll", time_ago)
        else:
            st.metric("Last UI Poll", "Never")

    with col2:
        st.metric("Total Polls", ui_status.poll_count)

    with col3:
        st.metric("Auto Refreshes", ui_status.auto_refresh_count)

    st.divider()

    # Source Status Details
    st.subheader("Data Source Status")

    # Define expected sources
    expected_sources = ['news', 'weather', 'email', 'calendar', 'brief']

    for source in expected_sources:
        status = all_statuses.get(source)

        if status:
            _render_source_status(source, status)
        else:
            # Source not yet initialized
            with st.expander(f"{source.title()} - Not Initialized", expanded=False):
                st.info("This source has not been initialized yet.")

    # Refresh button
    if st.button("Refresh Diagnostics", key="refresh_diagnostics"):
        st.rerun()


def _render_source_status(source_name: str, status):
    """Render status for a single data source.

    Args:
        source_name: Name of the source (news, weather, etc.)
        status: SourceStatus object
    """

    # Determine status indicator and color
    if status.status == RefreshStatus.SUCCESS:
        if status.is_fresh():
            indicator = "🟢"
            health_text = "Fresh"
            color = "green"
        elif status.is_stale():
            indicator = "🟡"
            health_text = "Stale"
            color = "orange"
        elif status.is_critical():
            indicator = "🔴"
            health_text = "Critical"
            color = "red"
        else:
            indicator = "🟢"
            health_text = "OK"
            color = "green"
    elif status.status == RefreshStatus.ERROR:
        indicator = "🔴"
        health_text = "Error"
        color = "red"
    elif status.status == RefreshStatus.IN_PROGRESS:
        indicator = "🔵"
        health_text = "Refreshing"
        color = "blue"
    elif status.status == RefreshStatus.DISABLED:
        indicator = "⚫"
        health_text = "Disabled"
        color = "gray"
    else:
        indicator = "⚪"
        health_text = "Pending"
        color = "gray"

    # Create expander with status indicator
    with st.expander(f"{indicator} {source_name.title()} - {health_text}", expanded=False):

        # Show status details in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Status:**")
            st.markdown(f":{color}[{status.status.value.title()}]")

        with col2:
            st.write("**Refresh Interval:**")
            st.write(f"{status.refresh_interval_minutes} min")

        with col3:
            st.write("**Items:**")
            st.write(f"{status.items_fetched}")

        # Last attempt time
        if status.last_attempt_time:
            st.write(f"**Last Attempt:** {_format_time_ago(status.last_attempt_time)} ({status.last_attempt_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            st.write("**Last Attempt:** Never")

        # Last success time
        if status.last_success_time:
            st.write(f"**Last Success:** {_format_time_ago(status.last_success_time)} ({status.last_success_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            st.write("**Last Success:** Never")

        # Next refresh time
        if status.next_refresh_time:
            st.write(f"**Next Refresh:** {_format_time_ago(status.next_refresh_time)} ({status.next_refresh_time.strftime('%Y-%m-%d %H:%M:%S')})")

        # Error message
        if status.error_message:
            st.error(f"**Error:** {status.error_message}")

        # Data age assessment
        st.write("**Data Freshness:**")
        if status.is_fresh():
            st.success("Data is fresh and up to date")
        elif status.is_stale():
            st.warning(f"Data is stale (older than {status.refresh_interval_minutes} min)")
        elif status.is_critical():
            st.error(f"Data is critically old (older than {status.refresh_interval_minutes * 2} min)")


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as 'X minutes/hours ago'.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable time ago string
    """
    now = datetime.now()
    delta = now - dt

    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"

    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

    return "Just now"


def render_diagnostics_sidebar():
    """Render a compact diagnostics view in the sidebar.

    Shows a quick health check of all sources.
    """
    st.sidebar.subheader("System Health")

    status_service = get_system_status_service()
    summary = status_service.get_summary()

    # Overall health indicator
    if summary['errors'] > 0 or summary['critical'] > 0:
        health_indicator = "🔴"
        health_text = "Issues Detected"
    elif summary['stale'] > 0:
        health_indicator = "🟡"
        health_text = "Some Stale Data"
    else:
        health_indicator = "🟢"
        health_text = "All Systems OK"

    st.sidebar.markdown(f"### {health_indicator} {health_text}")

    # Show metrics
    st.sidebar.metric("Fresh Sources", f"{summary['fresh']}/{summary['total_sources']}")

    if summary['stale'] > 0:
        st.sidebar.warning(f"{summary['stale']} stale source(s)")

    if summary['errors'] > 0:
        st.sidebar.error(f"{summary['errors']} error(s)")

    # Quick source list
    all_statuses = status_service.get_all_source_status()

    with st.sidebar.expander("Source Status", expanded=False):
        for source_name, status in all_statuses.items():
            if status.is_fresh():
                indicator = "🟢"
            elif status.is_stale():
                indicator = "🟡"
            elif status.is_critical() or status.status == RefreshStatus.ERROR:
                indicator = "🔴"
            else:
                indicator = "⚪"

            st.markdown(f"{indicator} {source_name.title()}")

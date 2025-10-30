"""MCP Server Test UI Component

This component provides a UI for testing MCP server tools directly from Streamlit.
It allows testing email and calendar tools before building the ReAct agent.
"""

import streamlit as st
import asyncio
from typing import Optional

from src.services.mcp_server import get_mcp_server


def render_mcp_test_tab():
    """Render MCP server testing tab."""
    st.header("🔧 MCP Server Testing")

    st.info(
        "This tab allows you to test MCP email/calendar tools directly. "
        "Configure OAuth credentials to enable these tools."
    )

    # Get MCP server instance
    try:
        server = get_mcp_server()
    except Exception as e:
        st.error(f"❌ Failed to initialize MCP server: {str(e)}")
        return

    # Show available tools
    st.subheader("Available Tools")
    tools_list = list(server.tools.keys())

    if tools_list:
        st.write(f"**Registered tools ({len(tools_list)}):**")
        cols = st.columns(3)
        for i, tool_name in enumerate(tools_list):
            with cols[i % 3]:
                st.code(tool_name, language="")
    else:
        st.warning("⚠️ No tools registered")

    st.divider()

    # Test email fetching
    _render_email_test_section(server, tools_list)

    st.divider()

    # Test calendar fetching
    _render_calendar_test_section(server, tools_list)


def _render_email_test_section(server, tools_list: list):
    """Render email testing section."""
    st.subheader("📧 Test Email Fetching")

    if "fetch_emails" not in tools_list:
        st.info(
            "ℹ️ Email tool not available. To enable:\n"
            "1. Configure Gmail OAuth credentials\n"
            "2. Save credentials to `credentials/gmail_client_secret.json`\n"
            "3. Restart the application\n\n"
            "See `OAUTH_SETUP_GUIDE.md` for setup instructions."
        )
        return

    # Email tool configuration
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.slider("Max emails", 1, 100, 50, key="email_max")
    with col2:
        query = st.text_input("Gmail query", "is:unread", key="email_query")

    if st.button("Fetch Emails", key="fetch_emails_btn"):
        with st.spinner("Fetching emails..."):
            try:
                # Execute email fetch
                response = asyncio.run(
                    server.execute_tool("fetch_emails", {
                        "max_results": max_results,
                        "query": query
                    })
                )

                if response.success:
                    emails = response.data
                    st.success(f"✅ Fetched {len(emails)} emails")

                    if len(emails) == 0:
                        st.info("No emails found matching query")
                    else:
                        # Display emails
                        for email in emails:
                            _render_email_card(email)
                else:
                    st.error(f"❌ Error: {response.error}")

            except Exception as e:
                st.error(f"❌ Exception: {str(e)}")
                st.exception(e)


def _render_email_card(email: dict):
    """Render a single email card."""
    # Color code by importance
    importance = email.get('importance_score', 0.0)
    if importance >= 0.8:
        border_color = "🔴"
    elif importance >= 0.5:
        border_color = "🟡"
    else:
        border_color = "🟢"

    with st.expander(f"{border_color} {email.get('subject', 'No Subject')}"):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**From:** {email.get('sender', 'Unknown')}")
            st.write(f"**Received:** {email.get('received_at', 'Unknown')}")

        with col2:
            st.metric("Importance", f"{importance:.2f}")
            if email.get('has_action_items', False):
                st.caption("⚡ Has Action Items")

        # Action items
        if email.get('has_action_items', False) and email.get('action_items'):
            st.write("**Action Items:**")
            for item in email['action_items']:
                st.write(f"- {item}")

        # Email body
        st.text_area(
            "Body",
            email.get('body', ''),
            height=150,
            key=f"email_body_{email.get('id', 'unknown')}"
        )


def _render_calendar_test_section(server, tools_list: list):
    """Render calendar testing section."""
    st.subheader("📅 Test Calendar Fetching")

    if "fetch_calendar_events" not in tools_list:
        st.info(
            "ℹ️ Calendar tool not available. To enable:\n"
            "1. Configure Google Calendar OAuth credentials\n"
            "2. Save credentials to `credentials/google_calendar_credentials.json`\n"
            "3. Restart the application\n\n"
            "See `OAUTH_SETUP_GUIDE.md` for setup instructions."
        )
        return

    # Calendar tool configuration
    col1, col2 = st.columns(2)
    with col1:
        days_ahead = st.slider("Days ahead", 1, 30, 7, key="cal_days")
    with col2:
        max_events = st.slider("Max events", 1, 100, 50, key="cal_max")

    if st.button("Fetch Calendar Events", key="fetch_calendar_btn"):
        with st.spinner("Fetching calendar events..."):
            try:
                # Execute calendar fetch
                response = asyncio.run(
                    server.execute_tool("fetch_calendar_events", {
                        "days_ahead": days_ahead,
                        "max_results": max_events
                    })
                )

                if response.success:
                    events = response.data
                    st.success(f"✅ Fetched {len(events)} events")

                    if len(events) == 0:
                        st.info("No upcoming events found")
                    else:
                        # Display events
                        for event in events:
                            _render_calendar_card(event)
                else:
                    st.error(f"❌ Error: {response.error}")

            except Exception as e:
                st.error(f"❌ Exception: {str(e)}")
                st.exception(e)


def _render_calendar_card(event: dict):
    """Render a single calendar event card."""
    # Color code by importance
    importance = event.get('importance_score', 0.0)
    if importance >= 0.8:
        border_color = "🔴"
    elif importance >= 0.5:
        border_color = "🟡"
    else:
        border_color = "🟢"

    with st.expander(f"{border_color} {event.get('summary', 'No Title')}"):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**Start:** {event.get('start_time', 'Unknown')}")
            st.write(f"**End:** {event.get('end_time', 'Unknown')}")
            if event.get('location'):
                st.write(f"**Location:** {event['location']}")

        with col2:
            st.metric("Importance", f"{importance:.2f}")
            if event.get('requires_preparation', False):
                st.caption("📝 Requires Prep")
            if event.get('is_focus_time', False):
                st.caption("🎯 Focus Time")

        # Preparation notes
        if event.get('requires_preparation', False) and event.get('preparation_notes'):
            st.write("**Preparation Notes:**")
            for note in event['preparation_notes']:
                st.write(f"- {note}")

        # Event description
        if event.get('description'):
            st.text_area(
                "Description",
                event['description'],
                height=100,
                key=f"event_desc_{event.get('id', 'unknown')}"
            )

        # Attendees
        if event.get('attendees'):
            st.write(f"**Attendees ({len(event['attendees'])}):**")
            st.write(", ".join(event['attendees']))

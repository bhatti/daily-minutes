"""Email UI Components - Pure rendering layer for email display.

This module contains ONLY UI rendering code using Streamlit.
All business logic is delegated to:
- EmailFormatter (src/ui/formatters/email_formatter.py)
- EmailService (src/services/email_service.py)

Component helper functions prepare data for rendering and handle state management.
"""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime

from src.core.models import EmailMessage
from src.ui.formatters.email_formatter import get_email_formatter
from src.services.email_service import get_email_service
from src.core.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Helper Functions - These have unit tests
# ===================================================================

def prepare_email_display_data(email: EmailMessage) -> Dict[str, any]:
    """Prepare email data for display using EmailFormatter.

    Args:
        email: Email message to prepare

    Returns:
        Dictionary with display-ready data
    """
    formatter = get_email_formatter()

    return {
        "subject": email.subject,
        "sender_display": formatter.format_sender_display(email.sender),
        "time_display": formatter.format_relative_time(email.received_at),
        "preview_text": formatter.get_preview_text(email, max_length=100),
        "importance_badge": get_importance_badge(email),
        "action_items_display": get_action_items_display(email),
        "is_read": email.is_read,
        "importance_score": email.importance_score,
    }


def get_importance_badge(email: EmailMessage) -> str:
    """Get importance badge display for email.

    Args:
        email: Email message

    Returns:
        Badge string (emoji or text)
    """
    if email.importance_score >= 0.9:
        return "🔴 Critical"
    elif email.importance_score >= 0.7:
        return "🟠 Important"
    elif email.importance_score >= 0.5:
        return "🟡 Medium"
    else:
        return ""  # No badge for low importance


def get_action_items_display(email: EmailMessage) -> str:
    """Get action items display for email.

    Args:
        email: Email message

    Returns:
        Formatted action items string
    """
    if not email.has_action_items or not email.action_items:
        return ""

    # Format as bullet list
    items = "\n".join([f"• {item}" for item in email.action_items])
    return f"📋 Action Items:\n{items}"


def should_show_email(
    email: EmailMessage,
    filter_unread: bool = False,
    filter_important: bool = False,
    search_query: Optional[str] = None
) -> bool:
    """Determine if email should be shown based on filters.

    Args:
        email: Email message
        filter_unread: Only show unread emails
        filter_important: Only show important emails
        search_query: Search query to match against

    Returns:
        True if email should be shown, False otherwise
    """
    # Apply unread filter
    if filter_unread and email.is_read:
        return False

    # Apply importance filter
    if filter_important and email.importance_score < 0.7:
        return False

    # Apply search filter
    if search_query:
        query_lower = search_query.lower()

        # Search in subject
        if query_lower in email.subject.lower():
            return True

        # Search in sender
        if query_lower in email.sender.lower():
            return True

        # Search in body
        if query_lower in email.body.lower():
            return True

        # No match found
        return False

    # All filters passed
    return True


def group_emails_for_display(emails: List[EmailMessage]) -> Dict[str, List[EmailMessage]]:
    """Group emails by date for display.

    Args:
        emails: List of emails to group

    Returns:
        Dictionary mapping date labels to email lists
    """
    if not emails:
        return {}

    formatter = get_email_formatter()
    return formatter.group_by_date(emails)


# ===================================================================
# Streamlit Rendering Functions - Pure UI code
# ===================================================================

def render_email_filters() -> Dict[str, any]:
    """Render email filter controls.

    Returns:
        Dictionary with filter settings
    """
    st.subheader("📧 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_unread = st.checkbox("Unread only", value=False, key="email_filter_unread")

    with col2:
        filter_important = st.checkbox("Important only", value=False, key="email_filter_important")

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["importance", "time"],
            index=0,
            key="email_sort_by"
        )

    search_query = st.text_input("🔍 Search emails", placeholder="Search subject, sender, or body...", key="email_search_query")

    return {
        "filter_unread": filter_unread,
        "filter_important": filter_important,
        "sort_by": sort_by,
        "search_query": search_query if search_query else None
    }


def render_email_card(email: EmailMessage, expanded: bool = False) -> None:
    """Render a single email as a card.

    Args:
        email: Email message to render
        expanded: Whether to show full email or preview
    """
    display_data = prepare_email_display_data(email)

    # Container for the email card
    with st.container():
        # Header row with sender and time
        col1, col2 = st.columns([3, 1])

        with col1:
            # Show read/unread indicator
            read_indicator = "✉️" if not display_data["is_read"] else "📭"
            st.markdown(f"**{read_indicator} {display_data['sender_display']}**")

        with col2:
            st.caption(display_data["time_display"])

        # Subject with importance badge
        subject_line = display_data["subject"]
        if display_data["importance_badge"]:
            subject_line = f"{display_data['importance_badge']} {subject_line}"

        st.markdown(f"### {subject_line}")

        # Show AI summary first (priority!) - inline, not hidden
        if hasattr(email, 'ai_summary') and email.ai_summary:
            st.info(f"💡 {email.ai_summary}")
        elif not expanded:
            # Fallback to preview/TLDR if no AI summary
            st.caption(display_data["preview_text"])

        # Action items if present - INLINE, always visible!
        if display_data["action_items_display"]:
            st.markdown(display_data["action_items_display"])

        # Preview or full body
        if expanded:
            st.markdown(email.body)

        # Show key points/insights
        if hasattr(email, 'key_points') and email.key_points:
            with st.expander("🔑 Key Points"):
                for point in email.key_points:
                    st.markdown(f"• {point}")

        # Divider between emails
        st.divider()


def render_email_list(
    emails: List[EmailMessage],
    filters: Optional[Dict[str, any]] = None
) -> None:
    """Render list of emails with grouping and filtering.

    Args:
        emails: List of emails to render
        filters: Optional filter settings
    """
    if not emails:
        st.info("No emails to display")
        return

    # Apply filters if provided
    if filters:
        filtered_emails = [
            email for email in emails
            if should_show_email(
                email,
                filter_unread=filters.get("filter_unread", False),
                filter_important=filters.get("filter_important", False),
                search_query=filters.get("search_query")
            )
        ]

        # Sort emails
        formatter = get_email_formatter()
        if filters.get("sort_by") == "importance":
            filtered_emails = formatter.sort_by_importance(filtered_emails)
        else:
            filtered_emails = formatter.sort_by_timestamp(filtered_emails)
    else:
        filtered_emails = emails

    # Show count
    st.caption(f"Showing {len(filtered_emails)} of {len(emails)} emails")

    # Group by date
    grouped = group_emails_for_display(filtered_emails)

    # Render each group
    for date_label, group_emails in grouped.items():
        st.subheader(date_label)

        for email in group_emails:
            render_email_card(email, expanded=False)


def render_email_statistics(emails: List[EmailMessage]) -> None:
    """Render email statistics.

    Args:
        emails: List of emails to analyze
    """
    if not emails:
        return

    formatter = get_email_formatter()
    stats = formatter.calculate_statistics(emails)

    # Display as metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total", stats["total"])

    with col2:
        st.metric("Unread", stats["unread"])

    with col3:
        st.metric("Read", stats["read"])

    with col4:
        st.metric("Important", stats["high_importance"])

    with col5:
        st.metric("Action Items", stats["with_action_items"])


def render_email_table(
    emails: List[EmailMessage],
    filters: Optional[Dict[str, any]] = None
) -> None:
    """Render emails in table format with all fields visible.

    Args:
        emails: List of emails to render
        filters: Optional filter settings
    """
    import pandas as pd

    if not emails:
        st.info("No emails to display")
        return

    # Apply filters if provided
    if filters:
        filtered_emails = [
            email for email in emails
            if should_show_email(
                email,
                filter_unread=filters.get("filter_unread", False),
                filter_important=filters.get("filter_important", False),
                search_query=filters.get("search_query")
            )
        ]

        # Sort emails
        formatter = get_email_formatter()
        if filters.get("sort_by") == "importance":
            filtered_emails = formatter.sort_by_importance(filtered_emails)
        else:
            filtered_emails = formatter.sort_by_timestamp(filtered_emails)
    else:
        filtered_emails = emails

    # Show count
    st.caption(f"Showing {len(filtered_emails)} of {len(emails)} emails")

    # Build table data
    table_data = []
    formatter = get_email_formatter()

    for email in filtered_emails:
        # Read status indicator
        read_status = "📭 Read" if email.is_read else "✉️ Unread"

        # Importance badge
        importance_badge = get_importance_badge(email)
        if not importance_badge:
            importance_badge = "🔵 Low"

        # Format time
        time_display = formatter.format_relative_time(email.received_at)

        # Action items indicator
        action_items = ""
        if email.has_action_items and email.action_items:
            action_items = f"📋 {len(email.action_items)} items"

        # TLDR/Preview - Priority: ai_summary > snippet > body preview (AI analysis first!)
        preview = ""
        if hasattr(email, 'ai_summary') and email.ai_summary:
            preview = f"💡 {email.ai_summary[:150]}" if len(email.ai_summary) > 150 else f"💡 {email.ai_summary}"
        elif hasattr(email, 'snippet') and email.snippet:
            preview = f"💬 {email.snippet[:100]}" if len(email.snippet) > 100 else f"💬 {email.snippet}"
        elif email.body:
            preview = email.body[:80] + "..." if len(email.body) > 80 else email.body

        table_data.append({
            "Status": read_status,
            "Importance": importance_badge,
            "From": email.sender[:30] + "..." if len(email.sender) > 30 else email.sender,
            "Subject": email.subject[:40] + "..." if len(email.subject) > 40 else email.subject,
            "Preview/Insight": preview,
            "Received": time_display,
            "Action Items": action_items,
        })

    # Create and display DataFrame
    if table_data:
        df = pd.DataFrame(table_data)

        # Display as interactive dataframe
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Importance": st.column_config.TextColumn("Importance", width="small"),
                "From": st.column_config.TextColumn("From", width="small"),
                "Subject": st.column_config.TextColumn("Subject", width="medium"),
                "Preview/Insight": st.column_config.TextColumn("Preview/Insight", width="large"),
                "Received": st.column_config.TextColumn("Received", width="small"),
                "Action Items": st.column_config.TextColumn("Action Items", width="small"),
            }
        )
    else:
        st.info("No emails match the current filters")


def render_email_tab() -> None:
    """Render the complete email tab.

    This is the main entry point for the email UI.
    """
    st.header("📧 Email")

    # Show filters
    filters = render_email_filters()

    # Show statistics if we have emails
    if "emails" in st.session_state:
        emails = st.session_state["emails"]

        # Statistics
        with st.expander("📊 Statistics", expanded=False):
            render_email_statistics(emails)

        # Email display - use LIST view (cards with AI summary + action items visible)
        render_email_list(emails, filters)
    else:
        st.info("No emails loaded. Emails are automatically fetched in the background.")

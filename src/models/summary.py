"""Models for daily minutes summaries and aggregation."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from src.models.base import BaseModel, DataSource, Priority
from src.models.calendar import CalendarEvent
from src.models.chat import BaseMessage, BaseThread
from src.models.email import EmailMessage, EmailThread
from src.models.news import NewsArticle, NewsSummary


class SummarySection(str, Enum):
    """Summary section types."""

    KEY_HIGHLIGHTS = "key_highlights"
    URGENT_ITEMS = "urgent_items"
    NEWS = "news"
    EMAILS = "emails"
    CALENDAR = "calendar"
    MESSAGES = "messages"
    ACTION_ITEMS = "action_items"
    DECISIONS = "decisions"
    METRICS = "metrics"


class SourceMetadata(BaseModel):
    """Metadata for a data source."""

    source: DataSource = Field(..., description="Data source")
    last_sync: Optional[datetime] = Field(None, description="Last sync time")
    items_count: int = Field(0, ge=0, description="Number of items")
    unread_count: int = Field(0, ge=0, description="Number of unread items")
    error_count: int = Field(0, ge=0, description="Number of errors")
    is_active: bool = Field(True, description="Source is active")
    sync_status: str = Field("pending", description="Sync status")

    def mark_synced(self, items: int = 0, unread: int = 0) -> None:
        """Mark source as synced."""
        self.last_sync = datetime.now()
        self.items_count = items
        self.unread_count = unread
        self.sync_status = "success"
        self.update_timestamp()

    def mark_error(self, error: str) -> None:
        """Mark source sync error."""
        self.error_count += 1
        self.sync_status = f"error: {error}"
        self.update_timestamp()


class SectionSummary(BaseModel):
    """Summary for a specific section."""

    section: SummarySection = Field(..., description="Section type")
    title: str = Field(..., description="Section title")

    # Content
    summary_text: str = Field(..., description="Summary text")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Section items")

    # Statistics
    total_items: int = Field(0, ge=0, description="Total items")
    important_items: int = Field(0, ge=0, description="Important items")
    action_required: int = Field(0, ge=0, description="Items requiring action")

    # Prioritization
    priority: Priority = Field(Priority.MEDIUM, description="Section priority")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Section importance")

    # Visualization
    chart_data: Optional[Dict[str, Any]] = Field(None, description="Data for charts")
    display_order: int = Field(0, description="Display order")

    def add_item(self, item: Dict[str, Any], is_important: bool = False) -> None:
        """Add an item to the section."""
        self.items.append(item)
        self.total_items += 1
        if is_important:
            self.important_items += 1
        self.update_timestamp()

    def calculate_importance(self) -> float:
        """Calculate section importance."""
        score = self.importance_score

        # Boost for urgent sections
        if self.section in [SummarySection.URGENT_ITEMS, SummarySection.ACTION_ITEMS]:
            score = min(1.0, score + 0.3)

        # Boost based on action required ratio
        if self.total_items > 0:
            action_ratio = self.action_required / self.total_items
            score = min(1.0, score + (action_ratio * 0.2))

        # Priority modifier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        return min(1.0, max(0.0, score))


class DailyMinutes(BaseModel):
    """Complete daily minutes aggregation."""

    date: datetime = Field(default_factory=datetime.now, description="Minutes date")
    user_id: Optional[str] = Field(None, description="User identifier")

    # Sections
    sections: List[SectionSummary] = Field(default_factory=list, description="Summary sections")

    # Source metadata
    sources: List[SourceMetadata] = Field(default_factory=list, description="Data sources")

    # Overall summary
    executive_summary: Optional[str] = Field(None, description="Executive summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="AI recommendations")

    # Aggregated items
    all_action_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All action items across sources"
    )
    all_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All decisions made"
    )

    # Statistics
    total_items_processed: int = Field(0, ge=0, description="Total items processed")
    total_unread: int = Field(0, ge=0, description="Total unread items")
    total_urgent: int = Field(0, ge=0, description="Total urgent items")

    # Time tracking
    generation_time: Optional[float] = Field(None, description="Time to generate (seconds)")
    last_refresh: Optional[datetime] = Field(None, description="Last refresh time")

    # User preferences
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences for display"
    )

    @field_validator("sections", mode="before")
    @classmethod
    def sort_sections(cls, v: List[SectionSummary]) -> List[SectionSummary]:
        """Sort sections by importance and display order."""
        if not v:
            return []
        # Sort by display_order first, then by importance
        return sorted(v, key=lambda s: (s.display_order, -s.calculate_importance()))

    def add_section(self, section: SectionSummary) -> None:
        """Add a section to the minutes."""
        self.sections.append(section)
        self.total_items_processed += section.total_items
        self.total_urgent += section.action_required
        self.update_timestamp()

    def add_source(self, source: SourceMetadata) -> None:
        """Add source metadata."""
        self.sources.append(source)
        self.total_unread += source.unread_count
        self.update_timestamp()

    def add_news_summary(self, news: NewsSummary) -> None:
        """Add news summary section."""
        section = SectionSummary(
            section=SummarySection.NEWS,
            title="News & Articles",
            summary_text=news.generate_brief(),
            total_items=news.total_articles,
            important_items=news.starred_articles,
            action_required=news.unread_articles,
            priority=Priority.MEDIUM
        )

        # Add top articles as items
        for article in news.get_top_articles(5):
            section.add_item({
                "title": article.title,
                "url": str(article.url),
                "source": article.source_name,
                "importance": article.calculate_importance(),
                "is_unread": not article.is_read,
            }, is_important=article.is_starred)

        self.add_section(section)

    def add_email_summary(self, emails: List[EmailMessage], threads: List[EmailThread]) -> None:
        """Add email summary section."""
        unread_emails = [e for e in emails if not e.is_read]
        important_emails = [e for e in emails if e.is_important or e.is_starred]
        action_emails = [e for e in emails if e.action_required]

        summary_text = f"{len(emails)} emails ({len(unread_emails)} unread)\n"
        summary_text += f"{len(threads)} conversations\n"
        summary_text += f"{len(action_emails)} require action"

        section = SectionSummary(
            section=SummarySection.EMAILS,
            title="Email",
            summary_text=summary_text,
            total_items=len(emails),
            important_items=len(important_emails),
            action_required=len(action_emails),
            priority=Priority.HIGH if action_emails else Priority.MEDIUM
        )

        # Add important emails
        for email in sorted(important_emails, key=lambda e: e.calculate_importance(), reverse=True)[:5]:
            section.add_item({
                "subject": email.subject,
                "from": email.from_name or email.from_address,
                "time": email.sent_at.isoformat(),
                "importance": email.calculate_importance(),
                "has_attachments": email.has_attachments(),
                "action_required": email.action_required,
            }, is_important=True)

        self.add_section(section)

    def add_calendar_summary(self, events: List[CalendarEvent]) -> None:
        """Add calendar summary section with smart prioritization."""
        today_events = [e for e in events if e.start_time.date() == datetime.now().date()]
        upcoming_events = [e for e in events if e.is_upcoming(hours=72)]
        special_events = [e for e in events if e.is_special_event()]

        summary_text = f"{len(today_events)} events today\n"
        summary_text += f"{len(upcoming_events)} upcoming (next 3 days)\n"
        if special_events:
            summary_text += f"⭐ {len(special_events)} special events"

        section = SectionSummary(
            section=SummarySection.CALENDAR,
            title="Calendar",
            summary_text=summary_text,
            total_items=len(events),
            important_items=len(special_events),
            action_required=len([e for e in events if e.response_required]),
            priority=Priority.HIGH if today_events else Priority.MEDIUM
        )

        # Prioritize special events and today's events
        priority_events = sorted(
            special_events + today_events,
            key=lambda e: e.calculate_importance(),
            reverse=True
        )[:7]  # Top 7 events

        for event in priority_events:
            section.add_item({
                "title": event.title,
                "time": event.start_time.isoformat(),
                "duration": event.get_duration_hours(),
                "type": event.event_type.value,
                "location": event.location,
                "meeting_url": str(event.meeting_url) if event.meeting_url else None,
                "importance": event.calculate_importance(),
                "is_special": event.is_special_event(),
                "response_required": event.response_required,
            }, is_important=event.is_special_event())

        self.add_section(section)

    def add_chat_summary(self, messages: List[BaseMessage], threads: List[BaseThread]) -> None:
        """Add chat/messaging summary section."""
        unread_messages = [m for m in messages if not m.is_read]
        mentions = [m for m in messages if m.is_mention]
        actionable = [m for m in messages if m.is_actionable()]

        summary_text = f"{len(messages)} messages ({len(unread_messages)} unread)\n"
        summary_text += f"{len(mentions)} mentions\n"
        summary_text += f"{len(threads)} active threads"

        section = SectionSummary(
            section=SummarySection.MESSAGES,
            title="Messages & Chat",
            summary_text=summary_text,
            total_items=len(messages),
            important_items=len(mentions),
            action_required=len(actionable),
            priority=Priority.HIGH if mentions else Priority.MEDIUM
        )

        # Add important messages
        important_messages = sorted(
            mentions + actionable,
            key=lambda m: m.calculate_importance(),
            reverse=True
        )[:5]

        for msg in important_messages:
            section.add_item({
                "channel": msg.channel_name,
                "sender": msg.sender_name,
                "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                "time": msg.timestamp.isoformat(),
                "type": msg.message_type.value,
                "importance": msg.calculate_importance(),
                "is_mention": msg.is_mention,
                "requires_response": msg.requires_response,
            }, is_important=msg.is_mention)

        self.add_section(section)

    def extract_all_action_items(self) -> None:
        """Extract and aggregate all action items from all sources."""
        action_items = []

        # From sections
        for section in self.sections:
            for item in section.items:
                if item.get("action_required"):
                    action_items.append({
                        "source": section.section.value,
                        "item": item,
                        "priority": section.priority.value,
                        "timestamp": datetime.now().isoformat()
                    })

        # Sort by priority and importance
        self.all_action_items = sorted(
            action_items,
            key=lambda x: (
                Priority[x["priority"].upper()].value == Priority.URGENT.value,
                x["item"].get("importance", 0)
            ),
            reverse=True
        )

        # Create action items section if we have items
        if self.all_action_items:
            action_section = SectionSummary(
                section=SummarySection.ACTION_ITEMS,
                title="Action Required",
                summary_text=f"{len(self.all_action_items)} items need your attention",
                total_items=len(self.all_action_items),
                important_items=len([a for a in self.all_action_items if a["priority"] == Priority.URGENT.value]),
                action_required=len(self.all_action_items),
                priority=Priority.URGENT,
                display_order=-1  # Show first
            )

            for action in self.all_action_items[:10]:  # Top 10
                action_section.items.append(action["item"])

            self.add_section(action_section)

    def generate_executive_summary(self) -> str:
        """Generate executive summary using AI or heuristics."""
        if self.executive_summary:
            return self.executive_summary

        summary = f"Daily Minutes for {self.date.strftime('%A, %B %d, %Y')}\n\n"

        # Key metrics
        summary += "📊 Overview:\n"
        summary += f"• {self.total_items_processed} total items processed\n"
        summary += f"• {self.total_unread} unread items\n"
        summary += f"• {self.total_urgent} urgent items\n\n"

        # Urgent items
        if self.total_urgent > 0:
            summary += "🚨 Urgent Items:\n"
            urgent_section = next((s for s in self.sections if s.section == SummarySection.URGENT_ITEMS), None)
            if urgent_section:
                for item in urgent_section.items[:3]:
                    summary += f"• {item.get('title', 'Urgent item')}\n"
            summary += "\n"

        # Key insights
        if self.key_insights:
            summary += "💡 Key Insights:\n"
            for insight in self.key_insights[:3]:
                summary += f"• {insight}\n"
            summary += "\n"

        # Recommendations
        if self.recommendations:
            summary += "🎯 Recommendations:\n"
            for rec in self.recommendations[:3]:
                summary += f"• {rec}\n"

        self.executive_summary = summary
        return summary

    def get_priority_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top priority items across all sections."""
        all_items = []

        for section in self.sections:
            for item in section.items:
                all_items.append({
                    "section": section.section.value,
                    "item": item,
                    "importance": item.get("importance", section.calculate_importance())
                })

        # Sort by importance
        sorted_items = sorted(all_items, key=lambda x: x["importance"], reverse=True)
        return sorted_items[:limit]

    def calculate_health_score(self) -> float:
        """Calculate overall health score for the day.

        Returns a score 0-1 indicating how well the user is managing their daily tasks.
        """
        score = 1.0

        # Penalize for unread items (max -0.3)
        if self.total_items_processed > 0:
            unread_ratio = self.total_unread / self.total_items_processed
            score -= min(0.3, unread_ratio * 0.3)

        # Penalize for urgent items (max -0.3)
        if self.total_urgent > 5:
            score -= 0.3
        elif self.total_urgent > 0:
            score -= (self.total_urgent / 5) * 0.3

        # Penalize for too many action items (max -0.2)
        if len(self.all_action_items) > 10:
            score -= 0.2
        elif len(self.all_action_items) > 0:
            score -= (len(self.all_action_items) / 10) * 0.2

        # Bonus for being up to date (max +0.2)
        for source in self.sources:
            if source.sync_status == "success" and source.last_sync:
                hours_since = (datetime.now() - source.last_sync).total_seconds() / 3600
                if hours_since < 1:
                    score += 0.05

        return min(1.0, max(0.0, score))

    def to_dashboard_format(self) -> Dict[str, Any]:
        """Format for dashboard display."""
        return {
            "date": self.date.isoformat(),
            "executive_summary": self.generate_executive_summary(),
            "health_score": self.calculate_health_score(),
            "sections": [
                {
                    "type": s.section.value,
                    "title": s.title,
                    "summary": s.summary_text,
                    "importance": s.calculate_importance(),
                    "stats": {
                        "total": s.total_items,
                        "important": s.important_items,
                        "action": s.action_required
                    },
                    "items": s.items[:5]  # Top 5 items per section
                }
                for s in sorted(self.sections, key=lambda x: (x.display_order, -x.calculate_importance()))
            ],
            "priority_items": self.get_priority_items(10),
            "metrics": {
                "total_items": self.total_items_processed,
                "unread": self.total_unread,
                "urgent": self.total_urgent,
                "action_items": len(self.all_action_items)
            },
            "sources": [
                {
                    "name": s.source.value,
                    "status": s.sync_status,
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "items": s.items_count,
                    "unread": s.unread_count
                }
                for s in self.sources
            ]
        }
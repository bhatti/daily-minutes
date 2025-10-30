"""Base models for chat/messaging platforms.

This module provides abstract base models that can be extended
for specific platforms like Slack, Discord, Microsoft Teams, etc.
"""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator

from src.models.base import BaseModel, Priority


class MessageType(str, Enum):
    """Generic message type enum."""

    REGULAR = "regular"
    REPLY = "reply"
    MENTION = "mention"
    DIRECT = "direct"
    SYSTEM = "system"
    BOT = "bot"
    FILE = "file"
    EDITED = "edited"
    DELETED = "deleted"


class ChannelType(str, Enum):
    """Generic channel/conversation type enum."""

    PUBLIC = "public"
    PRIVATE = "private"
    DIRECT = "direct"
    GROUP = "group"
    ANNOUNCEMENT = "announcement"


class BaseMessage(BaseModel):
    """Base model for chat messages across platforms."""

    message_id: str = Field(..., description="Unique message identifier")
    channel_id: str = Field(..., description="Channel/conversation identifier")
    channel_name: str = Field(..., description="Channel/conversation name")

    sender_id: str = Field(..., description="Sender identifier")
    sender_name: str = Field(..., description="Sender display name")
    sender_avatar: Optional[HttpUrl] = Field(None, description="Sender avatar URL")

    content: str = Field(..., description="Message content/text")
    message_type: MessageType = Field(MessageType.REGULAR, description="Message type")

    timestamp: datetime = Field(..., description="Message timestamp")
    edited_at: Optional[datetime] = Field(None, description="Edit timestamp")

    # Threading
    thread_id: Optional[str] = Field(None, description="Thread/conversation identifier")
    reply_to_id: Optional[str] = Field(None, description="ID of message being replied to")
    reply_count: int = Field(0, ge=0, description="Number of replies")

    # Mentions and recipients
    mentions: List[str] = Field(default_factory=list, description="Mentioned user IDs")
    recipients: List[str] = Field(default_factory=list, description="Message recipients")
    is_mention: bool = Field(False, description="Whether current user is mentioned")

    # Engagement
    reactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Message reactions"
    )
    attachments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Message attachments"
    )

    # Status flags
    is_read: bool = Field(False, description="Read status")
    is_starred: bool = Field(False, description="Starred/saved flag")
    is_pinned: bool = Field(False, description="Pinned flag")
    is_deleted: bool = Field(False, description="Deleted flag")
    is_bot: bool = Field(False, description="Bot message flag")

    # Importance and prioritization
    priority: Priority = Field(Priority.MEDIUM, description="Message priority")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    requires_response: bool = Field(False, description="Response required flag")
    has_responded: bool = Field(False, description="User has responded flag")

    # AI-enhanced fields
    summary: Optional[str] = Field(None, description="AI-generated summary")
    sentiment: Optional[str] = Field(None, description="Message sentiment")
    action_items: List[str] = Field(default_factory=list, description="Extracted action items")
    topics: List[str] = Field(default_factory=list, description="Identified topics")

    # Platform-specific data
    platform_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific additional data"
    )

    @field_validator("mentions", "recipients", "topics", mode="before")
    @classmethod
    def clean_string_lists(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate string lists."""
        if not v:
            return []
        return list(set(item.strip() for item in v if item and item.strip()))

    def is_thread(self) -> bool:
        """Check if message is part of a thread."""
        return self.thread_id is not None or self.reply_to_id is not None

    def is_thread_parent(self) -> bool:
        """Check if message is a thread parent."""
        return self.reply_count > 0

    def has_reactions(self) -> bool:
        """Check if message has reactions."""
        return len(self.reactions) > 0

    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0

    def get_reaction_count(self) -> int:
        """Get total reaction count."""
        return sum(r.get("count", len(r.get("users", []))) for r in self.reactions)

    def mark_as_read(self) -> None:
        """Mark message as read."""
        self.is_read = True
        self.update_timestamp()

    def toggle_star(self) -> None:
        """Toggle starred status."""
        self.is_starred = not self.is_starred
        self.update_timestamp()

    def add_reaction(self, reaction: str, user_id: str) -> None:
        """Add a reaction to the message."""
        for r in self.reactions:
            if r.get("reaction") == reaction:
                users = r.get("users", [])
                if user_id not in users:
                    users.append(user_id)
                    r["users"] = users
                    r["count"] = len(users)
                self.update_timestamp()
                return

        # New reaction
        self.reactions.append({
            "reaction": reaction,
            "users": [user_id],
            "count": 1
        })
        self.update_timestamp()

    def calculate_importance(self) -> float:
        """Calculate message importance with intelligent weighting.

        Base implementation that can be overridden by platform-specific classes.
        """
        score = self.importance_score

        # Message type weights
        type_weights = {
            MessageType.MENTION: 0.7,
            MessageType.DIRECT: 0.8,
            MessageType.REGULAR: 0.4,
            MessageType.REPLY: 0.5,
            MessageType.BOT: 0.2,
            MessageType.SYSTEM: 0.1,
        }
        base_score = type_weights.get(self.message_type, 0.4)

        # Start with type-based score
        score = base_score

        # Boost for mentions
        if self.is_mention:
            score = min(1.0, score + 0.3)

        # Boost for urgent keywords
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "blocker",
                          "emergency", "important", "deadline", "eod", "eob"]
        if any(keyword in self.content.lower() for keyword in urgent_keywords):
            score = min(1.0, score + 0.3)

        # Boost for questions requiring response
        if self.requires_response and not self.has_responded:
            score = min(1.0, score + 0.2)

        # Thread engagement
        if self.is_thread_parent() and self.reply_count > 5:
            score = min(1.0, score + 0.1)

        # Reaction engagement
        if self.get_reaction_count() > 10:
            score = min(1.0, score + 0.1)

        # Priority modifier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        # Starred messages get a boost
        if self.is_starred:
            score = min(1.0, score + 0.2)

        # Reduce score for bot messages unless important
        if self.is_bot and self.priority != Priority.HIGH:
            score *= 0.5

        # Reduce score for read messages
        if self.is_read:
            score *= 0.8

        return min(1.0, max(0.0, score))

    def is_actionable(self) -> bool:
        """Check if message requires action."""
        return (
            self.requires_response or
            self.is_mention or
            len(self.action_items) > 0 or
            (self.message_type == MessageType.DIRECT and "?" in self.content) or
            not self.is_read
        )

    @abstractmethod
    def format_for_platform(self) -> str:
        """Format message for specific platform display."""
        pass


class BaseChannel(BaseModel):
    """Base model for chat channels/conversations across platforms."""

    channel_id: str = Field(..., description="Unique channel identifier")
    name: str = Field(..., description="Channel/conversation name")
    channel_type: ChannelType = Field(ChannelType.PUBLIC, description="Channel type")

    description: Optional[str] = Field(None, description="Channel description")
    topic: Optional[str] = Field(None, description="Current topic")

    # Membership
    is_member: bool = Field(True, description="User is member")
    member_count: int = Field(0, ge=0, description="Number of members")
    members: List[str] = Field(default_factory=list, description="Member IDs")

    # Status
    is_archived: bool = Field(False, description="Archived flag")
    is_muted: bool = Field(False, description="Muted flag")
    is_favorite: bool = Field(False, description="Favorite flag")

    # Activity
    created_at: datetime = Field(..., description="Creation timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    last_read: Optional[datetime] = Field(None, description="Last read timestamp")

    # Unread tracking
    unread_count: int = Field(0, ge=0, description="Unread message count")
    unread_mentions: int = Field(0, ge=0, description="Unread mention count")

    # Organization
    categories: List[str] = Field(default_factory=list, description="Channel categories")
    priority: Priority = Field(Priority.MEDIUM, description="Channel priority")

    # Platform-specific data
    platform_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific additional data"
    )

    def has_unread(self) -> bool:
        """Check if channel has unread messages."""
        return self.unread_count > 0

    def has_unread_mentions(self) -> bool:
        """Check if channel has unread mentions."""
        return self.unread_mentions > 0

    def mark_as_read(self) -> None:
        """Mark channel as read."""
        self.unread_count = 0
        self.unread_mentions = 0
        self.last_read = datetime.now()
        self.update_timestamp()

    def calculate_importance(self) -> float:
        """Calculate channel importance."""
        score = 0.5

        # Channel type base scores
        type_scores = {
            ChannelType.DIRECT: 0.7,
            ChannelType.PRIVATE: 0.6,
            ChannelType.GROUP: 0.5,
            ChannelType.PUBLIC: 0.4,
            ChannelType.ANNOUNCEMENT: 0.8,
        }
        score = type_scores.get(self.channel_type, 0.4)

        # Boost for unread mentions
        if self.unread_mentions > 0:
            score = min(1.0, score + 0.3)

        # Boost for favorite channels
        if self.is_favorite:
            score = min(1.0, score + 0.2)

        # Boost based on unread count
        if self.unread_count > 10:
            score = min(1.0, score + 0.1)
        elif self.unread_count > 50:
            score = min(1.0, score + 0.2)

        # Reduce for muted channels
        if self.is_muted:
            score *= 0.3

        # Reduce for archived channels
        if self.is_archived:
            score *= 0.5

        # Priority modifier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        return min(1.0, max(0.0, score))

    def get_activity_level(self) -> str:
        """Get channel activity level."""
        if not self.last_activity:
            return "inactive"

        hours_since = (datetime.now() - self.last_activity).total_seconds() / 3600

        if hours_since < 1:
            return "very_active"
        elif hours_since < 24:
            return "active"
        elif hours_since < 168:  # 1 week
            return "moderate"
        else:
            return "quiet"

    @abstractmethod
    def format_for_platform(self) -> str:
        """Format channel info for specific platform display."""
        pass


class BaseThread(BaseModel):
    """Base model for message threads/conversations."""

    thread_id: str = Field(..., description="Unique thread identifier")
    channel_id: str = Field(..., description="Parent channel ID")
    channel_name: str = Field(..., description="Parent channel name")

    parent_message: BaseMessage = Field(..., description="Thread parent message")
    replies: List[BaseMessage] = Field(default_factory=list, description="Thread replies")

    participants: List[str] = Field(default_factory=list, description="Thread participants")
    reply_count: int = Field(0, ge=0, description="Number of replies")

    started_at: datetime = Field(..., description="Thread start time")
    last_reply_at: Optional[datetime] = Field(None, description="Last reply timestamp")

    is_subscribed: bool = Field(False, description="User subscribed to thread")
    has_unread: bool = Field(False, description="Has unread messages")

    # AI-enhanced fields
    summary: Optional[str] = Field(None, description="Thread summary")
    conclusion: Optional[str] = Field(None, description="Thread conclusion/resolution")
    action_items: List[str] = Field(default_factory=list, description="Extracted action items")
    key_decisions: List[str] = Field(default_factory=list, description="Key decisions made")

    def add_reply(self, message: BaseMessage) -> None:
        """Add a reply to the thread."""
        self.replies.append(message)
        self.reply_count += 1

        if message.sender_id not in self.participants:
            self.participants.append(message.sender_id)

        self.last_reply_at = message.timestamp
        self.update_timestamp()

    def get_latest_reply(self) -> Optional[BaseMessage]:
        """Get the latest reply in the thread."""
        if not self.replies:
            return None
        return max(self.replies, key=lambda m: m.timestamp)

    def mark_as_read(self) -> None:
        """Mark thread as read."""
        self.has_unread = False
        for reply in self.replies:
            reply.is_read = True
        self.update_timestamp()

    def is_active(self, hours: int = 24) -> bool:
        """Check if thread is recently active."""
        if not self.last_reply_at:
            return False
        return (datetime.now() - self.last_reply_at).total_seconds() / 3600 < hours

    def is_resolved(self) -> bool:
        """Check if thread has a conclusion/resolution."""
        return self.conclusion is not None

    def calculate_importance(self) -> float:
        """Calculate thread importance."""
        # Start with parent message importance
        score = self.parent_message.calculate_importance()

        # Boost for active threads
        if self.is_active(hours=24):
            score = min(1.0, score + 0.1)
        elif self.is_active(hours=3):
            score = min(1.0, score + 0.2)

        # Boost for threads with many participants
        if len(self.participants) > 3:
            score = min(1.0, score + 0.1)

        # Boost for subscribed threads
        if self.is_subscribed:
            score = min(1.0, score + 0.15)

        # Boost for unread
        if self.has_unread:
            score = min(1.0, score + 0.2)

        # Boost for threads with action items
        if self.action_items:
            score = min(1.0, score + 0.15)

        # Reduce if resolved
        if self.is_resolved():
            score *= 0.7

        return min(1.0, max(0.0, score))

    def generate_summary(self) -> str:
        """Generate thread summary."""
        if self.summary:
            return self.summary

        summary = f"Thread: {self.parent_message.content[:50]}...\n"
        summary += f"Participants: {len(self.participants)}\n"
        summary += f"Replies: {self.reply_count}\n"

        if self.last_reply_at:
            summary += f"Last activity: {self.last_reply_at.strftime('%Y-%m-%d %H:%M')}\n"

        if self.is_resolved():
            summary += f"\n✓ Resolved: {self.conclusion}\n"

        if self.key_decisions:
            summary += "\nKey Decisions:\n"
            for decision in self.key_decisions:
                summary += f"• {decision}\n"

        if self.action_items:
            summary += "\nAction Items:\n"
            for item in self.action_items:
                summary += f"• {item}\n"

        return summary
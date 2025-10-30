"""Slack-specific implementations of chat models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl

from src.models.base import BaseModel
from src.models.chat import BaseChannel, BaseMessage, BaseThread, ChannelType, MessageType


class SlackMessage(BaseMessage):
    """Slack-specific message implementation."""

    # Slack-specific fields
    ts: str = Field(..., description="Slack timestamp (used as message_id)")
    team_id: Optional[str] = Field(None, description="Slack team/workspace ID")

    # Slack threading
    thread_ts: Optional[str] = Field(None, description="Thread parent timestamp")

    # Slack-specific features
    blocks: List[Dict[str, Any]] = Field(default_factory=list, description="Slack blocks")
    permalink: Optional[HttpUrl] = Field(None, description="Message permalink")

    # Slack reactions use 'name' for emoji
    slack_reactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Slack-specific reaction format"
    )

    # File shares
    files: List[Dict[str, Any]] = Field(default_factory=list, description="Shared files")

    # App/Bot info
    app_id: Optional[str] = Field(None, description="App ID if bot message")
    bot_id: Optional[str] = Field(None, description="Bot ID")

    def __init__(self, **data):
        """Initialize Slack message with proper field mapping."""
        # Map Slack fields to base fields
        if "ts" in data and "message_id" not in data:
            data["message_id"] = data["ts"]

        if "thread_ts" in data and "thread_id" not in data:
            data["thread_id"] = data["thread_ts"]

        # Map Slack reactions to base format
        if "reactions" in data and isinstance(data["reactions"], list):
            for reaction in data["reactions"]:
                if "name" in reaction:
                    reaction["reaction"] = reaction.get("name")

        super().__init__(**data)

    def format_for_platform(self) -> str:
        """Format message for Slack display."""
        formatted = f"*{self.sender_name}* in #{self.channel_name}\n"
        formatted += f"_{self.timestamp.strftime('%b %d at %I:%M %p')}_\n\n"
        formatted += self.content

        if self.thread_ts:
            formatted += f"\n_Thread: {self.reply_count} replies_"

        if self.reactions:
            reactions_str = " ".join([f":{r.get('name', r.get('reaction'))}:{r.get('count', 1)}"
                                     for r in self.reactions])
            formatted += f"\n{reactions_str}"

        return formatted

    def to_slack_format(self) -> Dict[str, Any]:
        """Convert to Slack API format."""
        slack_data = {
            "ts": self.message_id,
            "text": self.content,
            "user": self.sender_id,
            "channel": self.channel_id,
            "type": "message",
        }

        if self.thread_ts:
            slack_data["thread_ts"] = self.thread_ts

        if self.blocks:
            slack_data["blocks"] = self.blocks

        if self.attachments:
            slack_data["attachments"] = self.attachments

        if self.bot_id:
            slack_data["bot_id"] = self.bot_id
            slack_data["subtype"] = "bot_message"

        return slack_data

    def add_slack_reaction(self, emoji_name: str, user_id: str) -> None:
        """Add a Slack-style reaction."""
        # Update both Slack-specific and base reactions
        for reaction in self.slack_reactions:
            if reaction.get("name") == emoji_name:
                users = reaction.get("users", [])
                if user_id not in users:
                    users.append(user_id)
                    reaction["users"] = users
                    reaction["count"] = len(users)

                # Also update base reactions
                self.add_reaction(emoji_name, user_id)
                return

        # New reaction
        self.slack_reactions.append({
            "name": emoji_name,
            "users": [user_id],
            "count": 1
        })

        # Also add to base reactions
        self.add_reaction(emoji_name, user_id)


class SlackChannel(BaseChannel):
    """Slack-specific channel implementation."""

    # Slack-specific identifiers
    team_id: Optional[str] = Field(None, description="Slack team/workspace ID")

    # Slack channel properties
    is_general: bool = Field(False, description="Is #general channel")
    is_shared: bool = Field(False, description="Shared with external workspace")
    is_ext_shared: bool = Field(False, description="Externally shared")
    is_org_shared: bool = Field(False, description="Shared across organization")
    is_pending_ext_shared: bool = Field(False, description="Pending external share")

    # Slack-specific metadata
    purpose: Optional[Dict[str, str]] = Field(None, description="Channel purpose")
    topic: Optional[Dict[str, str]] = Field(None, description="Channel topic")

    # Slack permissions
    is_channel: bool = Field(True, description="Is a channel (vs DM)")
    is_group: bool = Field(False, description="Is a private group")
    is_im: bool = Field(False, description="Is instant message")
    is_mpim: bool = Field(False, description="Is multi-party IM")

    # Pins and bookmarks
    pins: List[Dict[str, Any]] = Field(default_factory=list, description="Pinned items")

    def __init__(self, **data):
        """Initialize Slack channel with proper type mapping."""
        # Map Slack channel types
        if "is_im" in data and data["is_im"]:
            data["channel_type"] = ChannelType.DIRECT
        elif "is_mpim" in data and data["is_mpim"]:
            data["channel_type"] = ChannelType.GROUP
        elif "is_group" in data and data["is_group"]:
            data["channel_type"] = ChannelType.PRIVATE
        elif "is_general" in data and data["is_general"]:
            data["channel_type"] = ChannelType.ANNOUNCEMENT
        else:
            data["channel_type"] = ChannelType.PUBLIC

        super().__init__(**data)

    def format_for_platform(self) -> str:
        """Format channel info for Slack display."""
        prefix = "#" if self.channel_type == ChannelType.PUBLIC else ""
        formatted = f"{prefix}{self.name}"

        if self.is_shared:
            formatted += " 🔗"

        if self.unread_count > 0:
            formatted += f" ({self.unread_count} unread"
            if self.unread_mentions > 0:
                formatted += f", {self.unread_mentions} mentions"
            formatted += ")"

        if self.topic and isinstance(self.topic, dict):
            topic_value = self.topic.get("value", "")
            if topic_value:
                formatted += f"\nTopic: {topic_value}"

        return formatted

    def to_slack_format(self) -> Dict[str, Any]:
        """Convert to Slack API format."""
        return {
            "id": self.channel_id,
            "name": self.name,
            "is_channel": self.is_channel,
            "is_group": self.is_group,
            "is_im": self.is_im,
            "is_mpim": self.is_mpim,
            "is_member": self.is_member,
            "is_archived": self.is_archived,
            "is_general": self.is_general,
            "is_shared": self.is_shared,
            "is_ext_shared": self.is_ext_shared,
            "is_org_shared": self.is_org_shared,
            "num_members": self.member_count,
            "topic": self.topic,
            "purpose": self.purpose,
            "unread_count": self.unread_count,
            "unread_count_display": self.unread_mentions,
        }

    def get_display_name(self) -> str:
        """Get display name with appropriate prefix."""
        if self.is_im:
            return f"@{self.name}"
        elif self.is_mpim:
            return f"💬 {self.name}"
        elif self.channel_type == ChannelType.PRIVATE:
            return f"🔒 {self.name}"
        else:
            return f"#{self.name}"


class SlackThread(BaseThread):
    """Slack-specific thread implementation."""

    # Slack threading
    thread_ts: str = Field(..., description="Thread timestamp (thread_id)")

    # Slack-specific metadata
    is_broadcast: bool = Field(False, description="Thread broadcast to channel")
    reply_users: List[str] = Field(default_factory=list, description="Users who replied")
    reply_users_count: int = Field(0, ge=0, description="Number of reply users")

    def __init__(self, **data):
        """Initialize Slack thread with proper field mapping."""
        if "thread_ts" in data and "thread_id" not in data:
            data["thread_id"] = data["thread_ts"]

        super().__init__(**data)

    def format_for_platform(self) -> str:
        """Format thread for Slack display."""
        formatted = super().generate_summary()

        if self.is_broadcast:
            formatted += "\n📢 _Also sent to channel_"

        return formatted

    def to_slack_format(self) -> Dict[str, Any]:
        """Convert to Slack API format."""
        return {
            "thread_ts": self.thread_id,
            "reply_count": self.reply_count,
            "reply_users": self.reply_users,
            "reply_users_count": self.reply_users_count,
            "latest_reply": self.last_reply_at.isoformat() if self.last_reply_at else None,
            "is_broadcast": self.is_broadcast,
            "subscribed": self.is_subscribed,
        }


class SlackWorkspace(BaseModel):
    """Model for Slack workspace information."""

    workspace_id: str = Field(..., description="Workspace/Team ID")
    name: str = Field(..., description="Workspace name")
    domain: str = Field(..., description="Workspace domain")

    # User info
    user_id: str = Field(..., description="Current user ID")
    user_name: str = Field(..., description="Current user name")
    user_email: Optional[str] = Field(None, description="Current user email")

    # Workspace stats
    total_channels: int = Field(0, ge=0, description="Total channels")
    total_members: int = Field(0, ge=0, description="Total members")

    # Connection info
    is_connected: bool = Field(True, description="Connection status")
    last_sync: Optional[datetime] = Field(None, description="Last sync time")

    # Preferences
    notification_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Notification preferences"
    )

    def get_workspace_url(self) -> str:
        """Get workspace URL."""
        return f"https://{self.domain}.slack.com"
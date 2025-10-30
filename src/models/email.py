"""Models for email data."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import EmailStr, Field, field_validator

from src.models.base import BaseModel, Priority


class EmailStatus(str, Enum):
    """Email status enum."""

    UNREAD = "unread"
    READ = "read"
    REPLIED = "replied"
    FORWARDED = "forwarded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class EmailFolder(str, Enum):
    """Email folder enum."""

    INBOX = "inbox"
    SENT = "sent"
    DRAFTS = "drafts"
    ARCHIVE = "archive"
    TRASH = "trash"
    SPAM = "spam"
    IMPORTANT = "important"


class ImportanceScoringMixin:
    """Learn from user feedback using RLHF (Reinforcement Learning from Human Feedback).

    This mixin enables emails to be scored based on user preferences that are learned
    over time. Users can mark emails as important (👍) or unimportant (👎), and the
    system learns keywords to boost or filter future emails.
    """

    importance_score: float = 0.5  # AI's base score (0.0 to 1.0)
    boost_labels: Set[str] = set()  # Words user marked important
    filter_labels: Set[str] = set()  # Words user wants to skip

    def apply_rlhf_boost(self, content_text: str) -> float:
        """Adjust importance score based on learned preferences.

        Args:
            content_text: Email subject + body text to analyze

        Returns:
            Adjusted importance score between 0.0 and 1.0
        """
        adjusted = self.importance_score
        content_lower = content_text.lower()

        # Boost if content matches important keywords
        for label in self.boost_labels:
            if label.lower() in content_lower:
                adjusted += 0.1  # Bump up priority!

        # Penalize if content matches skip keywords
        for label in self.filter_labels:
            if label.lower() in content_lower:
                adjusted -= 0.2  # Push down priority!

        # Keep in valid range [0, 1]
        return max(0.0, min(1.0, adjusted))


class EmailAttachment(BaseModel):
    """Model for email attachment."""

    filename: str = Field(..., description="Attachment filename")
    content_type: str = Field(..., description="MIME content type")
    size: int = Field(..., gt=0, description="File size in bytes")

    content_id: Optional[str] = Field(None, description="Content ID for inline attachments")
    content: Optional[bytes] = Field(None, description="Attachment content")
    is_inline: bool = Field(False, description="Whether attachment is inline")

    def get_size_mb(self) -> float:
        """Get attachment size in megabytes."""
        return self.size / (1024 * 1024)

    def is_image(self) -> bool:
        """Check if attachment is an image."""
        return self.content_type.startswith("image/")

    def is_document(self) -> bool:
        """Check if attachment is a document."""
        doc_types = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt"]
        return any(self.content_type.find(dt) != -1 for dt in doc_types)


class EmailMessage(BaseModel):
    """Model for an email message."""

    message_id: str = Field(..., description="Unique message ID")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation")

    from_address: EmailStr = Field(..., description="Sender email address")
    from_name: Optional[str] = Field(None, description="Sender display name")

    to_addresses: List[EmailStr] = Field(..., description="Recipient addresses")
    cc_addresses: List[EmailStr] = Field(default_factory=list, description="CC addresses")
    bcc_addresses: List[EmailStr] = Field(default_factory=list, description="BCC addresses")

    subject: str = Field("", description="Email subject")
    body_text: Optional[str] = Field(None, description="Plain text body")
    body_html: Optional[str] = Field(None, description="HTML body")

    sent_at: datetime = Field(..., description="Send timestamp")
    received_at: Optional[datetime] = Field(None, description="Receive timestamp")

    status: EmailStatus = Field(EmailStatus.UNREAD, description="Email status")
    folder: EmailFolder = Field(EmailFolder.INBOX, description="Current folder")
    priority: Priority = Field(Priority.MEDIUM, description="Email priority")

    labels: List[str] = Field(default_factory=list, description="Email labels/tags")
    attachments: List[EmailAttachment] = Field(default_factory=list, description="Attachments")

    is_important: bool = Field(False, description="Important flag")
    is_starred: bool = Field(False, description="Starred flag")
    is_draft: bool = Field(False, description="Draft flag")

    reply_to: Optional[EmailStr] = Field(None, description="Reply-to address")
    in_reply_to: Optional[str] = Field(None, description="In-reply-to message ID")
    references: List[str] = Field(default_factory=list, description="Reference message IDs")

    spam_score: float = Field(0.0, ge=0.0, le=1.0, description="Spam probability (0-1)")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0-1)")

    summary: Optional[str] = Field(None, description="AI-generated summary")
    action_required: bool = Field(False, description="Whether action is required")
    action_deadline: Optional[datetime] = Field(None, description="Action deadline")

    @field_validator("labels", mode="before")
    @classmethod
    def clean_labels(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate labels."""
        if not v:
            return []
        return list(set(label.strip().lower() for label in v if label.strip()))

    def mark_as_read(self) -> None:
        """Mark email as read."""
        if self.status == EmailStatus.UNREAD:
            self.status = EmailStatus.READ
            self.update_timestamp()

    def mark_as_important(self) -> None:
        """Mark email as important."""
        self.is_important = True
        self.priority = Priority.HIGH
        self.update_timestamp()

    def toggle_star(self) -> None:
        """Toggle starred status."""
        self.is_starred = not self.is_starred
        self.update_timestamp()

    def move_to_folder(self, folder: EmailFolder) -> None:
        """Move email to specified folder."""
        self.folder = folder
        self.update_timestamp()

    def add_label(self, label: str) -> None:
        """Add a label to the email."""
        clean_label = label.strip().lower()
        if clean_label and clean_label not in self.labels:
            self.labels.append(clean_label)
            self.update_timestamp()

    def remove_label(self, label: str) -> None:
        """Remove a label from the email."""
        clean_label = label.strip().lower()
        if clean_label in self.labels:
            self.labels.remove(clean_label)
            self.update_timestamp()

    def has_attachments(self) -> bool:
        """Check if email has attachments."""
        return len(self.attachments) > 0

    def get_total_attachment_size(self) -> int:
        """Get total size of all attachments in bytes."""
        return sum(att.size for att in self.attachments)

    def get_recipients_count(self) -> int:
        """Get total number of recipients."""
        return len(self.to_addresses) + len(self.cc_addresses) + len(self.bcc_addresses)

    def is_reply(self) -> bool:
        """Check if email is a reply."""
        return self.in_reply_to is not None or len(self.references) > 0

    def calculate_importance(self) -> float:
        """Calculate email importance based on various factors."""
        score = self.importance_score

        # Boost for important flag
        if self.is_important:
            score = min(1.0, score + 0.3)

        # Boost for starred
        if self.is_starred:
            score = min(1.0, score + 0.2)

        # Boost for action required
        if self.action_required:
            score = min(1.0, score + 0.2)

            # Extra boost if deadline is approaching
            if self.action_deadline:
                hours_until_deadline = (self.action_deadline - datetime.now()).total_seconds() / 3600
                if hours_until_deadline < 24:
                    score = min(1.0, score + 0.2)

        # Priority multiplier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        return min(1.0, score)


class EmailThread(BaseModel):
    """Model for an email conversation thread."""

    thread_id: str = Field(..., description="Unique thread ID")
    subject: str = Field(..., description="Thread subject")

    messages: List[EmailMessage] = Field(default_factory=list, description="Thread messages")
    participants: List[EmailStr] = Field(default_factory=list, description="Thread participants")

    first_message_at: datetime = Field(..., description="First message timestamp")
    last_message_at: datetime = Field(..., description="Last message timestamp")

    message_count: int = Field(0, ge=0, description="Total message count")
    unread_count: int = Field(0, ge=0, description="Unread message count")

    is_important: bool = Field(False, description="Important thread flag")
    is_starred: bool = Field(False, description="Starred thread flag")
    is_muted: bool = Field(False, description="Muted thread flag")

    labels: List[str] = Field(default_factory=list, description="Thread labels")
    folder: EmailFolder = Field(EmailFolder.INBOX, description="Thread folder")

    summary: Optional[str] = Field(None, description="Thread summary")
    action_items: List[str] = Field(default_factory=list, description="Extracted action items")

    def add_message(self, message: EmailMessage) -> None:
        """Add a message to the thread."""
        self.messages.append(message)
        self.message_count += 1

        if message.status == EmailStatus.UNREAD:
            self.unread_count += 1

        # Update participants
        all_addresses = [message.from_address] + message.to_addresses + message.cc_addresses
        for addr in all_addresses:
            if addr not in self.participants:
                self.participants.append(addr)

        # Update timestamps
        if not self.first_message_at or message.sent_at < self.first_message_at:
            self.first_message_at = message.sent_at

        if not self.last_message_at or message.sent_at > self.last_message_at:
            self.last_message_at = message.sent_at

        self.update_timestamp()

    def mark_all_as_read(self) -> None:
        """Mark all messages in thread as read."""
        for message in self.messages:
            message.mark_as_read()
        self.unread_count = 0
        self.update_timestamp()

    def get_latest_message(self) -> Optional[EmailMessage]:
        """Get the latest message in the thread."""
        if not self.messages:
            return None
        return max(self.messages, key=lambda m: m.sent_at)

    def get_sender_statistics(self) -> Dict[str, int]:
        """Get message count by sender."""
        stats = {}
        for message in self.messages:
            sender = message.from_address
            stats[sender] = stats.get(sender, 0) + 1
        return stats

    def extract_action_items(self) -> List[str]:
        """Extract action items from thread messages."""
        # This would use NLP to extract action items
        # For now, return the stored action_items
        return self.action_items

    def generate_summary(self) -> str:
        """Generate a thread summary."""
        if self.summary:
            return self.summary

        # Basic summary generation
        summary = f"Thread: {self.subject}\n"
        summary += f"Messages: {self.message_count} ({self.unread_count} unread)\n"
        summary += f"Participants: {len(self.participants)}\n"
        summary += f"Duration: {self.first_message_at.date()} to {self.last_message_at.date()}\n"

        if self.action_items:
            summary += f"\nAction Items:\n"
            for item in self.action_items:
                summary += f"- {item}\n"

        return summary
"""Models for notifications and alerts."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from src.models.base import BaseModel, DataSource, Priority


class NotificationType(str, Enum):
    """Notification type enum."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"
    REMINDER = "reminder"
    MESSAGE = "message"
    UPDATE = "update"


class NotificationChannel(str, Enum):
    """Notification delivery channel."""

    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"


class NotificationStatus(str, Enum):
    """Notification status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    EXPIRED = "expired"


class Notification(BaseModel):
    """Model for a notification."""

    notification_id: str = Field(..., description="Unique notification ID")

    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    notification_type: NotificationType = Field(NotificationType.INFO, description="Type")

    source: DataSource = Field(..., description="Source of notification")
    source_id: Optional[str] = Field(None, description="Source item ID")
    source_url: Optional[str] = Field(None, description="Link to source")

    priority: Priority = Field(Priority.MEDIUM, description="Priority")
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP],
        description="Delivery channels"
    )

    recipient: str = Field(..., description="Recipient user ID")
    sender: Optional[str] = Field(None, description="Sender ID")

    status: NotificationStatus = Field(NotificationStatus.PENDING, description="Status")

    scheduled_for: Optional[datetime] = Field(None, description="Scheduled send time")
    sent_at: Optional[datetime] = Field(None, description="Actual send time")
    delivered_at: Optional[datetime] = Field(None, description="Delivery time")
    read_at: Optional[datetime] = Field(None, description="Read time")

    expires_at: Optional[datetime] = Field(None, description="Expiration time")

    # Actions
    actions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Available actions (e.g., buttons)"
    )
    action_taken: Optional[str] = Field(None, description="Action taken by user")

    # Tracking
    retry_count: int = Field(0, ge=0, description="Send retry count")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Grouping
    group_id: Optional[str] = Field(None, description="Group ID for bundling")
    is_bundled: bool = Field(False, description="Part of bundled notification")

    # Rich content
    icon: Optional[str] = Field(None, description="Icon identifier")
    color: Optional[str] = Field(None, description="Color theme")
    image_url: Optional[str] = Field(None, description="Image URL")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")

    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def is_actionable(self) -> bool:
        """Check if notification has actions."""
        return len(self.actions) > 0 and not self.action_taken

    def mark_as_sent(self) -> None:
        """Mark notification as sent."""
        self.status = NotificationStatus.SENT
        self.sent_at = datetime.now()
        self.update_timestamp()

    def mark_as_delivered(self) -> None:
        """Mark notification as delivered."""
        self.status = NotificationStatus.DELIVERED
        self.delivered_at = datetime.now()
        self.update_timestamp()

    def mark_as_read(self) -> None:
        """Mark notification as read."""
        self.status = NotificationStatus.READ
        self.read_at = datetime.now()
        self.update_timestamp()

    def mark_as_failed(self, error: str) -> None:
        """Mark notification as failed."""
        self.status = NotificationStatus.FAILED
        self.error_message = error
        self.retry_count += 1
        self.update_timestamp()

    def should_retry(self, max_retries: int = 3) -> bool:
        """Check if notification should be retried."""
        return (
            self.status == NotificationStatus.FAILED and
            self.retry_count < max_retries and
            not self.is_expired()
        )

    def calculate_importance(self) -> float:
        """Calculate notification importance."""
        # Type-based scores
        type_scores = {
            NotificationType.ERROR: 0.9,
            NotificationType.ALERT: 0.8,
            NotificationType.WARNING: 0.7,
            NotificationType.REMINDER: 0.6,
            NotificationType.MESSAGE: 0.5,
            NotificationType.UPDATE: 0.4,
            NotificationType.SUCCESS: 0.3,
            NotificationType.INFO: 0.2,
        }

        score = type_scores.get(self.notification_type, 0.5)

        # Priority modifier
        priority_multipliers = {
            Priority.LOW: 0.7,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 1.5,
        }
        score *= priority_multipliers.get(self.priority, 1.0)

        # Boost for actionable notifications
        if self.is_actionable():
            score = min(1.0, score + 0.2)

        # Reduce for read notifications
        if self.status == NotificationStatus.READ:
            score *= 0.3

        return min(1.0, max(0.0, score))


class NotificationPreferences(BaseModel):
    """Model for user notification preferences."""

    user_id: str = Field(..., description="User ID")

    # Channel preferences
    enabled_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP],
        description="Enabled channels"
    )

    # Type preferences
    enabled_types: List[NotificationType] = Field(
        default_factory=lambda: list(NotificationType),
        description="Enabled notification types"
    )

    # Source preferences
    enabled_sources: List[DataSource] = Field(
        default_factory=lambda: list(DataSource),
        description="Enabled sources"
    )

    # Timing preferences
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23, description="Quiet hours start (hour)")
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23, description="Quiet hours end (hour)")
    timezone: str = Field("UTC", description="User timezone")

    # Delivery preferences
    bundle_notifications: bool = Field(True, description="Bundle similar notifications")
    bundle_interval: int = Field(300, gt=0, description="Bundle interval in seconds")

    min_priority: Priority = Field(Priority.LOW, description="Minimum priority to notify")

    # Channel-specific settings
    email_frequency: str = Field("instant", description="Email frequency (instant/hourly/daily)")
    push_sound: bool = Field(True, description="Enable push notification sound")
    push_vibrate: bool = Field(True, description="Enable push notification vibration")

    # Content preferences
    show_preview: bool = Field(True, description="Show message preview")
    show_sender: bool = Field(True, description="Show sender information")

    # Do not disturb
    dnd_enabled: bool = Field(False, description="Do not disturb enabled")
    dnd_until: Optional[datetime] = Field(None, description="DND until time")

    def is_channel_enabled(self, channel: NotificationChannel) -> bool:
        """Check if channel is enabled."""
        return channel in self.enabled_channels

    def is_type_enabled(self, notification_type: NotificationType) -> bool:
        """Check if notification type is enabled."""
        return notification_type in self.enabled_types

    def is_source_enabled(self, source: DataSource) -> bool:
        """Check if source is enabled."""
        return source in self.enabled_sources

    def is_in_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        if not self.quiet_hours_start or not self.quiet_hours_end:
            return False

        current_hour = datetime.now().hour

        if self.quiet_hours_start < self.quiet_hours_end:
            # Normal case (e.g., 22:00 - 08:00)
            return self.quiet_hours_start <= current_hour < self.quiet_hours_end
        else:
            # Crosses midnight (e.g., 22:00 - 08:00)
            return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end

    def is_dnd_active(self) -> bool:
        """Check if Do Not Disturb is active."""
        if not self.dnd_enabled:
            return False

        if self.dnd_until:
            return datetime.now() < self.dnd_until

        return True

    def should_send_notification(self, notification: Notification) -> bool:
        """Check if notification should be sent based on preferences."""
        # Check DND
        if self.is_dnd_active() and notification.priority != Priority.URGENT:
            return False

        # Check quiet hours (except for urgent)
        if self.is_in_quiet_hours() and notification.priority != Priority.URGENT:
            return False

        # Check minimum priority
        priority_values = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.URGENT: 4,
        }

        if priority_values.get(notification.priority, 0) < priority_values.get(self.min_priority, 0):
            return False

        # Check if type is enabled
        if not self.is_type_enabled(notification.notification_type):
            return False

        # Check if source is enabled
        if not self.is_source_enabled(notification.source):
            return False

        return True


class NotificationQueue(BaseModel):
    """Model for notification queue management."""

    queue_id: str = Field(..., description="Queue ID")
    user_id: str = Field(..., description="User ID")

    pending: List[Notification] = Field(default_factory=list, description="Pending notifications")
    bundled: List[List[Notification]] = Field(default_factory=list, description="Bundled notifications")

    last_processed: Optional[datetime] = Field(None, description="Last processing time")
    next_scheduled: Optional[datetime] = Field(None, description="Next scheduled processing")

    def add_notification(self, notification: Notification, preferences: NotificationPreferences) -> None:
        """Add notification to queue."""
        if preferences.should_send_notification(notification):
            self.pending.append(notification)
            self.update_timestamp()

    def get_ready_notifications(self) -> List[Notification]:
        """Get notifications ready to send."""
        now = datetime.now()
        ready = []

        for notification in self.pending:
            if notification.scheduled_for and notification.scheduled_for > now:
                continue
            if notification.is_expired():
                notification.status = NotificationStatus.EXPIRED
                continue
            ready.append(notification)

        return ready

    def bundle_notifications(self, interval: int = 300) -> List[List[Notification]]:
        """Bundle similar notifications."""
        if not self.pending:
            return []

        # Group by type and source
        groups: Dict[tuple, List[Notification]] = {}

        for notification in self.pending:
            key = (notification.notification_type, notification.source)
            if key not in groups:
                groups[key] = []
            groups[key].append(notification)

        # Create bundles
        bundles = []
        for notifications in groups.values():
            if len(notifications) > 1:
                bundles.append(notifications)
            else:
                bundles.append(notifications)

        self.bundled = bundles
        return bundles

    def clear_sent(self) -> None:
        """Clear sent notifications from queue."""
        self.pending = [
            n for n in self.pending
            if n.status not in [NotificationStatus.SENT, NotificationStatus.DELIVERED, NotificationStatus.READ]
        ]
        self.update_timestamp()
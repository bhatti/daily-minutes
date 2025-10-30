"""System Status Service - Track refresh and polling activities for debugging.

This service provides observability into:
- When each MCP data source was last refreshed
- Whether refreshes succeeded or failed
- How many items were fetched
- When the UI last polled for data
- Any errors encountered

This makes it easy to debug issues like "why isn't weather showing?"
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path


class RefreshStatus(str, Enum):
    """Status of a refresh operation."""
    SUCCESS = "success"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    DISABLED = "disabled"


@dataclass
class SourceStatus:
    """Status information for a data source."""
    source_name: str
    status: RefreshStatus
    last_attempt_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    items_fetched: int = 0
    error_message: Optional[str] = None
    next_refresh_time: Optional[datetime] = None
    refresh_interval_minutes: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO formatted dates."""
        data = asdict(self)
        if self.last_attempt_time:
            data['last_attempt_time'] = self.last_attempt_time.isoformat()
        if self.last_success_time:
            data['last_success_time'] = self.last_success_time.isoformat()
        if self.next_refresh_time:
            data['next_refresh_time'] = self.next_refresh_time.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceStatus':
        """Create from dictionary with ISO formatted dates."""
        if 'last_attempt_time' in data and data['last_attempt_time']:
            data['last_attempt_time'] = datetime.fromisoformat(data['last_attempt_time'])
        if 'last_success_time' in data and data['last_success_time']:
            data['last_success_time'] = datetime.fromisoformat(data['last_success_time'])
        if 'next_refresh_time' in data and data['next_refresh_time']:
            data['next_refresh_time'] = datetime.fromisoformat(data['next_refresh_time'])
        if 'status' in data:
            data['status'] = RefreshStatus(data['status'])
        return cls(**data)

    def is_fresh(self, max_age_minutes: int = None) -> bool:
        """Check if the data is fresh based on last success time."""
        if not self.last_success_time:
            return False

        max_age = max_age_minutes or self.refresh_interval_minutes
        age_minutes = (datetime.now() - self.last_success_time).total_seconds() / 60
        return age_minutes < max_age

    def is_stale(self, max_age_minutes: int = None) -> bool:
        """Check if the data is stale but not critical."""
        if not self.last_success_time:
            return True

        max_age = max_age_minutes or self.refresh_interval_minutes
        age_minutes = (datetime.now() - self.last_success_time).total_seconds() / 60

        # Stale if older than interval but less than 2x interval
        return max_age <= age_minutes < (max_age * 2)

    def is_critical(self, max_age_minutes: int = None) -> bool:
        """Check if the data is critically old or has errors."""
        if self.status == RefreshStatus.ERROR:
            return True

        if not self.last_success_time:
            return True

        max_age = max_age_minutes or self.refresh_interval_minutes
        age_minutes = (datetime.now() - self.last_success_time).total_seconds() / 60

        # Critical if older than 2x the refresh interval
        return age_minutes >= (max_age * 2)


@dataclass
class UIPollingStatus:
    """Status information for UI polling."""
    last_poll_time: Optional[datetime] = None
    poll_count: int = 0
    auto_refresh_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO formatted dates."""
        data = asdict(self)
        if self.last_poll_time:
            data['last_poll_time'] = self.last_poll_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIPollingStatus':
        """Create from dictionary with ISO formatted dates."""
        if 'last_poll_time' in data and data['last_poll_time']:
            data['last_poll_time'] = datetime.fromisoformat(data['last_poll_time'])
        return cls(**data)


class SystemStatusService:
    """Service for tracking and reporting system status."""

    def __init__(self, status_file: str = "data/system_status.json"):
        """Initialize the status service.

        Args:
            status_file: Path to JSON file for persisting status
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory status tracking
        self._source_status: Dict[str, SourceStatus] = {}
        self._ui_status = UIPollingStatus()

        # Load persisted status
        self._load_status()

    def _load_status(self):
        """Load status from disk."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)

                # Load source statuses
                if 'sources' in data:
                    for source_name, source_data in data['sources'].items():
                        self._source_status[source_name] = SourceStatus.from_dict(source_data)

                # Load UI status
                if 'ui' in data:
                    self._ui_status = UIPollingStatus.from_dict(data['ui'])

            except Exception as e:
                # If loading fails, start fresh
                print(f"Warning: Could not load status file: {e}")

    def _save_status(self):
        """Save status to disk."""
        try:
            data = {
                'sources': {
                    name: status.to_dict()
                    for name, status in self._source_status.items()
                },
                'ui': self._ui_status.to_dict(),
                'updated_at': datetime.now().isoformat()
            }

            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save status file: {e}")

    def record_refresh_start(self, source_name: str, refresh_interval_minutes: int = 60):
        """Record that a refresh has started for a source.

        Args:
            source_name: Name of the source (news, weather, email, calendar)
            refresh_interval_minutes: Configured refresh interval
        """
        if source_name not in self._source_status:
            self._source_status[source_name] = SourceStatus(
                source_name=source_name,
                status=RefreshStatus.PENDING,
                refresh_interval_minutes=refresh_interval_minutes
            )

        status = self._source_status[source_name]
        status.status = RefreshStatus.IN_PROGRESS
        status.last_attempt_time = datetime.now()
        status.error_message = None

        self._save_status()

    def record_refresh_success(
        self,
        source_name: str,
        items_fetched: int = 0,
        next_refresh_minutes: int = None
    ):
        """Record a successful refresh.

        Args:
            source_name: Name of the source
            items_fetched: Number of items fetched
            next_refresh_minutes: When the next refresh is scheduled
        """
        if source_name not in self._source_status:
            self._source_status[source_name] = SourceStatus(source_name=source_name, status=RefreshStatus.SUCCESS)

        status = self._source_status[source_name]
        status.status = RefreshStatus.SUCCESS
        status.last_success_time = datetime.now()
        status.items_fetched = items_fetched
        status.error_message = None

        if next_refresh_minutes:
            status.next_refresh_time = datetime.now() + timedelta(minutes=next_refresh_minutes)

        self._save_status()

    def record_refresh_error(self, source_name: str, error_message: str):
        """Record a failed refresh.

        Args:
            source_name: Name of the source
            error_message: Error message
        """
        if source_name not in self._source_status:
            self._source_status[source_name] = SourceStatus(source_name=source_name, status=RefreshStatus.ERROR)

        status = self._source_status[source_name]
        status.status = RefreshStatus.ERROR
        status.error_message = error_message

        self._save_status()

    def record_source_disabled(self, source_name: str):
        """Record that a source is disabled.

        Args:
            source_name: Name of the source
        """
        if source_name not in self._source_status:
            self._source_status[source_name] = SourceStatus(source_name=source_name, status=RefreshStatus.DISABLED)

        status = self._source_status[source_name]
        status.status = RefreshStatus.DISABLED

        self._save_status()

    def record_ui_poll(self, is_auto_refresh: bool = False):
        """Record that the UI polled for data.

        Args:
            is_auto_refresh: Whether this was an automatic refresh
        """
        self._ui_status.last_poll_time = datetime.now()
        self._ui_status.poll_count += 1

        if is_auto_refresh:
            self._ui_status.auto_refresh_count += 1

        self._save_status()

    def get_source_status(self, source_name: str) -> Optional[SourceStatus]:
        """Get status for a specific source.

        Args:
            source_name: Name of the source

        Returns:
            SourceStatus or None if not found
        """
        return self._source_status.get(source_name)

    def get_all_source_status(self) -> Dict[str, SourceStatus]:
        """Get status for all sources.

        Returns:
            Dict mapping source name to status
        """
        return self._source_status.copy()

    def get_ui_status(self) -> UIPollingStatus:
        """Get UI polling status.

        Returns:
            UIPollingStatus
        """
        return self._ui_status

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of system health.

        Returns:
            Dict with summary information
        """
        sources = self.get_all_source_status()

        fresh_count = sum(1 for s in sources.values() if s.is_fresh())
        stale_count = sum(1 for s in sources.values() if s.is_stale())
        critical_count = sum(1 for s in sources.values() if s.is_critical())
        error_count = sum(1 for s in sources.values() if s.status == RefreshStatus.ERROR)
        disabled_count = sum(1 for s in sources.values() if s.status == RefreshStatus.DISABLED)

        return {
            'total_sources': len(sources),
            'fresh': fresh_count,
            'stale': stale_count,
            'critical': critical_count,
            'errors': error_count,
            'disabled': disabled_count,
            'ui_last_poll': self._ui_status.last_poll_time.isoformat() if self._ui_status.last_poll_time else None,
            'ui_poll_count': self._ui_status.poll_count,
            'ui_auto_refresh_count': self._ui_status.auto_refresh_count,
        }


# Global singleton instance
_system_status_service: Optional[SystemStatusService] = None


def get_system_status_service() -> SystemStatusService:
    """Get the global system status service instance.

    Returns:
        SystemStatusService instance
    """
    global _system_status_service

    if _system_status_service is None:
        _system_status_service = SystemStatusService()

    return _system_status_service

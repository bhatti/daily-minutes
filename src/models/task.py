"""Models for tasks and TODO items."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from src.models.base import BaseModel, DataSource, Priority


class TaskStatus(str, Enum):
    """Task status enum."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    DONE = "done"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class TaskCategory(str, Enum):
    """Task category enum."""

    WORK = "work"
    PERSONAL = "personal"
    HOME = "home"
    HEALTH = "health"
    FINANCE = "finance"
    LEARNING = "learning"
    PROJECT = "project"
    ERRANDS = "errands"
    OTHER = "other"


class Task(BaseModel):
    """Model for a task or TODO item."""

    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Detailed description")

    status: TaskStatus = Field(TaskStatus.TODO, description="Task status")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority")
    category: TaskCategory = Field(TaskCategory.OTHER, description="Task category")

    # Timing
    created_date: datetime = Field(default_factory=datetime.now, description="Creation date")
    due_date: Optional[datetime] = Field(None, description="Due date")
    completed_date: Optional[datetime] = Field(None, description="Completion date")

    estimated_duration: Optional[int] = Field(None, gt=0, description="Estimated duration in minutes")
    actual_duration: Optional[int] = Field(None, gt=0, description="Actual duration in minutes")

    # Organization
    project_id: Optional[str] = Field(None, description="Associated project")
    parent_task_id: Optional[str] = Field(None, description="Parent task ID")
    subtasks: List[str] = Field(default_factory=list, description="Subtask IDs")

    tags: List[str] = Field(default_factory=list, description="Task tags")
    labels: List[str] = Field(default_factory=list, description="Task labels")

    # Assignment
    assigned_to: Optional[str] = Field(None, description="Assigned user")
    assigned_by: Optional[str] = Field(None, description="Assigning user")
    collaborators: List[str] = Field(default_factory=list, description="Collaborators")

    # Source integration
    source: DataSource = Field(DataSource.CUSTOM, description="Task source")
    external_id: Optional[str] = Field(None, description="External system ID")
    external_url: Optional[str] = Field(None, description="External system URL")

    # Progress tracking
    progress_percentage: int = Field(0, ge=0, le=100, description="Progress percentage")
    checklist_items: List[Dict[str, bool]] = Field(
        default_factory=list,
        description="Checklist items with completion status"
    )

    # Dependencies
    blocked_by: List[str] = Field(default_factory=list, description="Blocking task IDs")
    blocks: List[str] = Field(default_factory=list, description="Tasks this blocks")

    # Recurrence
    is_recurring: bool = Field(False, description="Recurring task flag")
    recurrence_pattern: Optional[str] = Field(None, description="Recurrence pattern (cron-like)")
    next_occurrence: Optional[datetime] = Field(None, description="Next occurrence for recurring")

    # Notes and attachments
    notes: Optional[str] = Field(None, description="Task notes")
    attachments: List[Dict[str, str]] = Field(default_factory=list, description="Attachments")

    # Tracking
    reminder_time: Optional[datetime] = Field(None, description="Reminder time")
    reminder_sent: bool = Field(False, description="Reminder sent flag")

    is_starred: bool = Field(False, description="Starred/favorite flag")
    is_archived: bool = Field(False, description="Archived flag")

    @field_validator("tags", "labels", mode="before")
    @classmethod
    def clean_string_lists(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate string lists."""
        if not v:
            return []
        return list(set(item.strip().lower() for item in v if item.strip()))

    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.due_date or self.status in [TaskStatus.DONE, TaskStatus.CANCELLED]:
            return False
        return datetime.now() > self.due_date

    def is_due_soon(self, hours: int = 24) -> bool:
        """Check if task is due within specified hours."""
        if not self.due_date or self.status in [TaskStatus.DONE, TaskStatus.CANCELLED]:
            return False
        return datetime.now() + timedelta(hours=hours) > self.due_date

    def is_blocked(self) -> bool:
        """Check if task is blocked."""
        return self.status == TaskStatus.BLOCKED or len(self.blocked_by) > 0

    def mark_as_done(self) -> None:
        """Mark task as done."""
        self.status = TaskStatus.DONE
        self.completed_date = datetime.now()
        self.progress_percentage = 100
        if self.estimated_duration and not self.actual_duration:
            # If no actual duration recorded, use estimate
            self.actual_duration = self.estimated_duration
        self.update_timestamp()

    def update_progress(self, percentage: int) -> None:
        """Update task progress."""
        self.progress_percentage = max(0, min(100, percentage))
        if self.progress_percentage == 100:
            self.mark_as_done()
        elif self.progress_percentage > 0 and self.status == TaskStatus.TODO:
            self.status = TaskStatus.IN_PROGRESS
        self.update_timestamp()

    def add_checklist_item(self, item: str, completed: bool = False) -> None:
        """Add a checklist item."""
        self.checklist_items.append({"item": item, "completed": completed})
        self.update_checklist_progress()

    def update_checklist_progress(self) -> None:
        """Update progress based on checklist completion."""
        if self.checklist_items:
            completed = sum(1 for item in self.checklist_items if item.get("completed", False))
            total = len(self.checklist_items)
            self.progress_percentage = int((completed / total) * 100)
            self.update_timestamp()

    def calculate_importance(self) -> float:
        """Calculate task importance with intelligent prioritization."""
        score = 0.5  # Base score

        # Priority base scores
        priority_scores = {
            Priority.LOW: 0.2,
            Priority.MEDIUM: 0.5,
            Priority.HIGH: 0.7,
            Priority.URGENT: 0.9,
        }
        score = priority_scores.get(self.priority, 0.5)

        # Overdue tasks get major boost
        if self.is_overdue():
            days_overdue = (datetime.now() - self.due_date).days
            overdue_boost = min(0.4, days_overdue * 0.1)  # Max 0.4 boost
            score = min(1.0, score + overdue_boost)

        # Due soon gets boost
        elif self.is_due_soon(hours=24):
            score = min(1.0, score + 0.2)
        elif self.is_due_soon(hours=72):
            score = min(1.0, score + 0.1)

        # Blocked tasks get reduced score (they can't be acted on)
        if self.is_blocked():
            score *= 0.5

        # In-progress tasks get slight boost (momentum)
        if self.status == TaskStatus.IN_PROGRESS:
            score = min(1.0, score + 0.1)

        # Starred tasks get boost
        if self.is_starred:
            score = min(1.0, score + 0.15)

        # Work tasks during work hours get boost
        if self.category == TaskCategory.WORK:
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Work hours
                score = min(1.0, score + 0.1)

        return min(1.0, max(0.0, score))

    def get_time_estimate_display(self) -> str:
        """Get human-readable time estimate."""
        if not self.estimated_duration:
            return "No estimate"

        hours = self.estimated_duration // 60
        minutes = self.estimated_duration % 60

        if hours > 0:
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        return f"{minutes}m"

    def generate_summary(self) -> str:
        """Generate task summary."""
        summary = f"📝 {self.title}\n"
        summary += f"Status: {self.status.value.replace('_', ' ').title()}\n"
        summary += f"Priority: {self.priority.value.upper()}\n"

        if self.due_date:
            if self.is_overdue():
                days_overdue = (datetime.now() - self.due_date).days
                summary += f"⚠️ Overdue by {days_overdue} days\n"
            else:
                summary += f"Due: {self.due_date.strftime('%Y-%m-%d %H:%M')}\n"

        if self.progress_percentage > 0:
            summary += f"Progress: {self.progress_percentage}%\n"

        if self.checklist_items:
            completed = sum(1 for item in self.checklist_items if item.get("completed", False))
            summary += f"Checklist: {completed}/{len(self.checklist_items)} completed\n"

        if self.is_blocked():
            summary += "🚫 Blocked\n"

        return summary


class TaskList(BaseModel):
    """Model for a collection of tasks."""

    list_id: str = Field(..., description="List identifier")
    name: str = Field(..., description="List name")
    description: Optional[str] = Field(None, description="List description")

    tasks: List[Task] = Field(default_factory=list, description="Tasks in list")

    owner: str = Field(..., description="List owner")
    shared_with: List[str] = Field(default_factory=list, description="Shared users")

    default_category: Optional[TaskCategory] = Field(None, description="Default category")
    color: Optional[str] = Field(None, description="List color")
    icon: Optional[str] = Field(None, description="List icon")

    is_archived: bool = Field(False, description="Archived flag")
    is_default: bool = Field(False, description="Default list flag")

    def add_task(self, task: Task) -> None:
        """Add a task to the list."""
        if self.default_category and not task.category:
            task.category = self.default_category
        self.tasks.append(task)
        self.update_timestamp()

    def get_active_tasks(self) -> List[Task]:
        """Get active (non-completed) tasks."""
        return [
            task for task in self.tasks
            if task.status not in [TaskStatus.DONE, TaskStatus.CANCELLED]
            and not task.is_archived
        ]

    def get_overdue_tasks(self) -> List[Task]:
        """Get overdue tasks."""
        return [task for task in self.tasks if task.is_overdue()]

    def get_tasks_by_priority(self, priority: Priority) -> List[Task]:
        """Get tasks by priority."""
        return [task for task in self.tasks if task.priority == priority]

    def get_statistics(self) -> Dict[str, int]:
        """Get task statistics."""
        active_tasks = self.get_active_tasks()
        return {
            "total": len(self.tasks),
            "active": len(active_tasks),
            "completed": len([t for t in self.tasks if t.status == TaskStatus.DONE]),
            "overdue": len(self.get_overdue_tasks()),
            "urgent": len([t for t in active_tasks if t.priority == Priority.URGENT]),
            "blocked": len([t for t in self.tasks if t.is_blocked()]),
        }


class TaskSummary(BaseModel):
    """Model for task summary in daily minutes."""

    date: datetime = Field(default_factory=datetime.now, description="Summary date")

    total_tasks: int = Field(0, ge=0, description="Total tasks")
    completed_today: int = Field(0, ge=0, description="Tasks completed today")
    overdue_tasks: int = Field(0, ge=0, description="Overdue tasks")
    due_today: int = Field(0, ge=0, description="Tasks due today")
    due_soon: int = Field(0, ge=0, description="Tasks due in next 3 days")

    high_priority_tasks: List[Task] = Field(
        default_factory=list,
        description="High priority tasks"
    )

    blocked_tasks: List[Task] = Field(
        default_factory=list,
        description="Blocked tasks"
    )

    productivity_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Daily productivity score"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Task recommendations"
    )

    def calculate_productivity_score(self) -> float:
        """Calculate daily productivity score."""
        score = 0.5  # Base score

        # Reward for completing tasks
        if self.total_tasks > 0:
            completion_rate = self.completed_today / max(self.due_today, 1)
            score = min(1.0, completion_rate)

        # Penalize for overdue tasks
        if self.overdue_tasks > 0:
            penalty = min(0.3, self.overdue_tasks * 0.1)
            score = max(0.0, score - penalty)

        # Bonus for clearing high priority tasks
        if self.completed_today > 0 and not self.high_priority_tasks:
            score = min(1.0, score + 0.2)

        self.productivity_score = score
        return score

    def generate_brief(self) -> str:
        """Generate task brief."""
        brief = f"📋 Tasks Overview\n"
        brief += f"Active: {self.total_tasks} tasks\n"

        if self.completed_today > 0:
            brief += f"✅ Completed today: {self.completed_today}\n"

        if self.overdue_tasks > 0:
            brief += f"⚠️ Overdue: {self.overdue_tasks}\n"

        if self.due_today > 0:
            brief += f"📅 Due today: {self.due_today}\n"

        if self.due_soon > 0:
            brief += f"📆 Due soon: {self.due_soon}\n"

        if self.blocked_tasks:
            brief += f"🚫 Blocked: {len(self.blocked_tasks)}\n"

        brief += f"\n💪 Productivity: {self.productivity_score:.0%}"

        return brief
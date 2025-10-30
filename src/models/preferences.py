"""Models for user preferences and reinforcement learning feedback."""

from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from src.models.base import BaseModel, DataSource, Priority


class FeedbackType(str, Enum):
    """Feedback type for reinforcement learning."""

    LIKE = "like"
    DISLIKE = "dislike"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    TOO_MUCH = "too_much"
    TOO_LITTLE = "too_little"
    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class UserPreferences(BaseModel):
    """Model for user preferences and personalization settings."""

    user_id: str = Field(..., description="User ID")

    # Content preferences
    preferred_sources: List[DataSource] = Field(
        default_factory=lambda: list(DataSource),
        description="Preferred data sources"
    )

    source_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Source importance weights (0-1)"
    )

    interests: List[str] = Field(default_factory=list, description="User interests/topics")
    excluded_topics: List[str] = Field(default_factory=list, description="Topics to exclude")

    # Timing preferences
    daily_summary_time: time = Field(
        time(8, 0),
        description="Preferred daily summary time"
    )

    timezone: str = Field("UTC", description="User timezone")
    workday_start: time = Field(time(9, 0), description="Workday start time")
    workday_end: time = Field(time(17, 0), description="Workday end time")
    work_days: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="Work days (1=Monday, 7=Sunday)"
    )

    # Display preferences
    max_items_per_section: int = Field(10, gt=0, le=50, description="Max items per section")
    show_weather: bool = Field(True, description="Include weather in summary")
    show_tasks: bool = Field(True, description="Include tasks in summary")
    show_calendar: bool = Field(True, description="Include calendar in summary")

    summary_length: str = Field("medium", description="Summary length (brief/medium/detailed)")
    language: str = Field("en", description="Preferred language")

    # Priority preferences
    priority_threshold: Priority = Field(
        Priority.LOW,
        description="Minimum priority to include"
    )

    importance_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance score"
    )

    # AI preferences
    ai_suggestions_enabled: bool = Field(True, description="Enable AI suggestions")
    ai_summarization_enabled: bool = Field(True, description="Enable AI summarization")
    ai_action_extraction: bool = Field(True, description="Extract action items with AI")

    # Learning preferences (for RL)
    enable_learning: bool = Field(True, description="Enable learning from feedback")
    learning_rate: float = Field(0.1, ge=0.0, le=1.0, description="Learning rate for updates")

    # Learned weights (updated by RL)
    topic_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Learned topic relevance scores"
    )

    source_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Source performance metrics"
    )

    time_preferences: Dict[str, float] = Field(
        default_factory=dict,
        description="Learned time-based preferences"
    )

    # Feature toggles
    features_enabled: Dict[str, bool] = Field(
        default_factory=lambda: {
            "smart_bundling": True,
            "auto_categorization": True,
            "duplicate_detection": True,
            "trend_detection": True,
            "anomaly_detection": False,
            "predictive_alerts": False,
        },
        description="Enabled features"
    )

    @field_validator("interests", "excluded_topics", mode="before")
    @classmethod
    def clean_string_lists(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate string lists."""
        if not v:
            return []
        return list(set(item.strip().lower() for item in v if item.strip()))

    def is_work_time(self) -> bool:
        """Check if current time is during work hours."""
        now = datetime.now()
        current_time = now.time()
        current_day = now.isoweekday()

        return (
            current_day in self.work_days and
            self.workday_start <= current_time <= self.workday_end
        )

    def get_source_weight(self, source: DataSource) -> float:
        """Get weight for a data source."""
        return self.source_weights.get(source.value, 0.5)

    def update_source_weight(self, source: DataSource, delta: float) -> None:
        """Update source weight based on feedback."""
        current = self.get_source_weight(source)
        new_weight = max(0.0, min(1.0, current + delta))
        self.source_weights[source.value] = new_weight
        self.update_timestamp()

    def get_topic_score(self, topic: str) -> float:
        """Get learned score for a topic."""
        return self.topic_scores.get(topic.lower(), 0.5)

    def update_topic_score(self, topic: str, delta: float) -> None:
        """Update topic score based on feedback."""
        topic = topic.lower()
        current = self.get_topic_score(topic)
        new_score = max(0.0, min(1.0, current + delta))
        self.topic_scores[topic] = new_score
        self.update_timestamp()

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features_enabled.get(feature, False)

    def calculate_content_relevance(self, content: Dict[str, Any]) -> float:
        """Calculate relevance score for content based on preferences."""
        score = 0.5  # Base score

        # Check source preference
        source = content.get("source")
        if source:
            score *= self.get_source_weight(DataSource(source))

        # Check topic relevance
        topics = content.get("topics", [])
        if topics:
            topic_scores = [self.get_topic_score(t) for t in topics]
            if topic_scores:
                avg_topic_score = sum(topic_scores) / len(topic_scores)
                score = (score + avg_topic_score) / 2

        # Check against interests
        content_text = f"{content.get('title', '')} {content.get('description', '')}".lower()
        interest_matches = sum(1 for interest in self.interests if interest in content_text)
        if interest_matches:
            score = min(1.0, score + (interest_matches * 0.1))

        # Check against excluded topics
        excluded_matches = sum(1 for excluded in self.excluded_topics if excluded in content_text)
        if excluded_matches:
            score = max(0.0, score - (excluded_matches * 0.2))

        # Time-based adjustments
        if self.is_work_time():
            # Boost work-related content during work hours
            if content.get("category") == "work":
                score = min(1.0, score + 0.2)
        else:
            # Boost personal content outside work hours
            if content.get("category") == "personal":
                score = min(1.0, score + 0.1)

        return score


class UserFeedback(BaseModel):
    """Model for user feedback on content/summaries."""

    feedback_id: str = Field(..., description="Feedback ID")
    user_id: str = Field(..., description="User ID")

    # Target of feedback
    target_type: str = Field(..., description="Type of item (article/email/summary/etc)")
    target_id: str = Field(..., description="ID of the item")
    target_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about target"
    )

    # Feedback
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    rating: Optional[float] = Field(
        None,
        ge=0.0,
        le=5.0,
        description="Rating (0-5)"
    )
    comment: Optional[str] = Field(None, description="User comment")

    # Context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context when feedback was given"
    )

    presented_at: datetime = Field(..., description="When item was presented")
    feedback_at: datetime = Field(default_factory=datetime.now, description="When feedback given")

    # Derived metrics
    interaction_time: Optional[float] = Field(
        None,
        description="Time spent on item (seconds)"
    )
    did_click: bool = Field(False, description="User clicked/opened item")
    did_share: bool = Field(False, description="User shared item")
    did_save: bool = Field(False, description="User saved item")

    # Learning
    applied_to_model: bool = Field(False, description="Feedback applied to model")
    learning_weight: float = Field(1.0, ge=0.0, le=1.0, description="Weight for learning")

    def calculate_reward(self) -> float:
        """Calculate reward signal for reinforcement learning."""
        reward = 0.0

        # Explicit feedback
        feedback_rewards = {
            FeedbackType.LIKE: 1.0,
            FeedbackType.DISLIKE: -1.0,
            FeedbackType.HELPFUL: 0.8,
            FeedbackType.NOT_HELPFUL: -0.8,
            FeedbackType.RELEVANT: 0.7,
            FeedbackType.IRRELEVANT: -0.7,
            FeedbackType.ACCURATE: 0.6,
            FeedbackType.INACCURATE: -0.6,
            FeedbackType.TOO_MUCH: -0.3,
            FeedbackType.TOO_LITTLE: -0.3,
        }
        reward = feedback_rewards.get(self.feedback_type, 0.0)

        # Implicit feedback
        if self.did_click:
            reward += 0.2
        if self.did_save:
            reward += 0.3
        if self.did_share:
            reward += 0.4

        # Engagement time signal
        if self.interaction_time:
            if self.interaction_time > 30:  # More than 30 seconds
                reward += 0.1
            elif self.interaction_time < 2:  # Less than 2 seconds (bounce)
                reward -= 0.2

        # Rating adjustment
        if self.rating is not None:
            rating_adjustment = (self.rating - 2.5) / 2.5  # Normalize to -1 to 1
            reward = (reward + rating_adjustment) / 2

        return max(-1.0, min(1.0, reward))


class LearningState(BaseModel):
    """Model for tracking reinforcement learning state."""

    user_id: str = Field(..., description="User ID")

    # Q-values for different actions/content types
    q_values: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Q-values for state-action pairs"
    )

    # State features
    state_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Current state features"
    )

    # Exploration parameters
    epsilon: float = Field(0.1, ge=0.0, le=1.0, description="Exploration rate")
    epsilon_decay: float = Field(0.995, ge=0.0, le=1.0, description="Epsilon decay rate")
    min_epsilon: float = Field(0.01, ge=0.0, le=1.0, description="Minimum epsilon")

    # Learning metrics
    total_episodes: int = Field(0, ge=0, description="Total learning episodes")
    total_reward: float = Field(0.0, description="Cumulative reward")
    average_reward: float = Field(0.0, description="Average reward")

    # History
    recent_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent actions taken"
    )
    recent_rewards: List[float] = Field(
        default_factory=list,
        description="Recent rewards received"
    )

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95
    ) -> None:
        """Update Q-value using Q-learning update rule."""
        if state not in self.q_values:
            self.q_values[state] = {}

        current_q = self.q_values[state].get(action, 0.0)

        # Find max Q-value for next state (simplified)
        max_next_q = 0.0  # Simplified for this model

        # Q-learning update
        new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
        self.q_values[state][action] = new_q

        # Update metrics
        self.total_episodes += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.total_episodes

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Track history
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

        self.update_timestamp()

    def get_best_action(self, state: str, actions: List[str]) -> str:
        """Get best action for a state using epsilon-greedy policy."""
        import random

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation
        if state not in self.q_values:
            return random.choice(actions)

        state_q_values = self.q_values[state]
        best_action = max(actions, key=lambda a: state_q_values.get(a, 0.0))
        return best_action

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get learning performance summary."""
        recent_avg = sum(self.recent_rewards[-10:]) / 10 if len(self.recent_rewards) >= 10 else 0

        return {
            "total_episodes": self.total_episodes,
            "average_reward": self.average_reward,
            "recent_average_reward": recent_avg,
            "epsilon": self.epsilon,
            "exploration_rate": f"{self.epsilon:.1%}",
            "learned_preferences": len(self.q_values),
        }
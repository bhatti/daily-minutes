"""Models for news and RSS feed data."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator

from src.models.base import BaseModel, DataSource, Priority


class NewsArticle(BaseModel):
    """Model for a news article."""

    title: str = Field(..., description="Article title")
    url: HttpUrl = Field(..., description="Article URL")
    source: DataSource = Field(..., description="Source of the article")
    source_name: str = Field(..., description="Human-readable source name")

    author: Optional[str] = Field(None, description="Article author")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    description: Optional[str] = Field(None, description="Article description/summary")
    content: Optional[str] = Field(None, description="Full article content")

    tags: List[str] = Field(default_factory=list, description="Article tags/categories")
    image_url: Optional[HttpUrl] = Field(None, description="Featured image URL")

    priority: Priority = Field(Priority.MEDIUM, description="Article priority")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score (0-1)")
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")

    # AI-generated fields
    ai_summary: Optional[str] = Field(None, description="AI-generated summary/analysis")
    tldr: Optional[str] = Field(None, description="AI-generated TLDR")
    key_learnings: List[str] = Field(default_factory=list, description="AI-extracted key learnings")

    is_read: bool = Field(False, description="Whether article has been read")
    is_starred: bool = Field(False, description="Whether article is starred")

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate tags."""
        if not v:
            return []
        # Remove duplicates and empty strings
        return list(set(tag.strip().lower() for tag in v if tag.strip()))

    @field_validator("relevance_score", "sentiment_score")
    @classmethod
    def validate_scores(cls, v: float, info) -> float:
        """Validate score ranges."""
        field_name = info.field_name
        if field_name == "relevance_score" and not (0 <= v <= 1):
            raise ValueError("Relevance score must be between 0 and 1")
        if field_name == "sentiment_score" and not (-1 <= v <= 1):
            raise ValueError("Sentiment score must be between -1 and 1")
        return v

    def mark_as_read(self) -> None:
        """Mark article as read."""
        self.is_read = True
        self.update_timestamp()

    def toggle_star(self) -> None:
        """Toggle starred status."""
        self.is_starred = not self.is_starred
        self.update_timestamp()

    def calculate_importance(self) -> float:
        """Calculate article importance based on priority and scores."""
        priority_weights = {
            Priority.LOW: 0.25,
            Priority.MEDIUM: 0.5,
            Priority.HIGH: 0.75,
            Priority.URGENT: 1.0,
        }

        # Handle both enum and string values (due to use_enum_values=True)
        if isinstance(self.priority, str):
            # Map string values to weights
            str_priority_weights = {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "urgent": 1.0,
            }
            priority_score = str_priority_weights.get(self.priority, 0.5)
        else:
            priority_score = priority_weights.get(self.priority, 0.5)
        # Combine priority with relevance (weighted average)
        importance = (priority_score * 0.6) + (self.relevance_score * 0.4)

        # Boost for starred items
        if self.is_starred:
            importance = min(1.0, importance * 1.2)

        return importance


class RSSFeed(BaseModel):
    """Model for RSS feed configuration."""

    name: str = Field(..., description="Feed name")
    url: HttpUrl = Field(..., description="Feed URL")
    category: str = Field("general", description="Feed category")

    is_active: bool = Field(True, description="Whether feed is active")
    update_frequency: int = Field(3600, gt=0, description="Update frequency in seconds")
    last_fetched: Optional[datetime] = Field(None, description="Last fetch timestamp")

    max_articles: int = Field(10, gt=0, le=100, description="Maximum articles to fetch")
    filter_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords to filter articles"
    )
    exclude_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords to exclude articles"
    )

    error_count: int = Field(0, ge=0, description="Consecutive error count")
    last_error: Optional[str] = Field(None, description="Last error message")

    @field_validator("filter_keywords", "exclude_keywords", mode="before")
    @classmethod
    def clean_keywords(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate keywords."""
        if not v:
            return []
        return list(set(kw.strip().lower() for kw in v if kw.strip()))

    def record_fetch(self) -> None:
        """Record successful fetch."""
        self.last_fetched = datetime.now()
        self.error_count = 0
        self.last_error = None
        self.update_timestamp()

    def record_error(self, error: str) -> None:
        """Record fetch error."""
        self.error_count += 1
        self.last_error = error
        self.update_timestamp()

    def should_fetch(self) -> bool:
        """Check if feed should be fetched based on frequency."""
        if not self.is_active:
            return False

        if not self.last_fetched:
            return True

        time_since_fetch = (datetime.now() - self.last_fetched).total_seconds()
        return time_since_fetch >= self.update_frequency

    def matches_filters(self, article: NewsArticle) -> bool:
        """Check if article matches feed filters."""
        # Check exclude keywords first
        if self.exclude_keywords:
            text = f"{article.title} {article.description or ''}".lower()
            if any(kw in text for kw in self.exclude_keywords):
                return False

        # Check filter keywords (if specified, at least one must match)
        if self.filter_keywords:
            text = f"{article.title} {article.description or ''}".lower()
            return any(kw in text for kw in self.filter_keywords)

        return True


class NewsSummary(BaseModel):
    """Model for aggregated news summary."""

    date: datetime = Field(default_factory=datetime.now, description="Summary date")
    source: DataSource = Field(..., description="News source")

    total_articles: int = Field(0, ge=0, description="Total articles")
    unread_articles: int = Field(0, ge=0, description="Unread articles")
    starred_articles: int = Field(0, ge=0, description="Starred articles")

    top_articles: List[NewsArticle] = Field(
        default_factory=list,
        description="Top articles by importance"
    )

    categories: Dict[str, int] = Field(
        default_factory=dict,
        description="Article count by category"
    )

    sentiment_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Articles by sentiment (positive/neutral/negative)"
    )

    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics/trends"
    )

    summary_text: Optional[str] = Field(
        None,
        description="AI-generated summary text"
    )

    def add_article(self, article: NewsArticle) -> None:
        """Add article to summary statistics."""
        self.total_articles += 1

        if not article.is_read:
            self.unread_articles += 1

        if article.is_starred:
            self.starred_articles += 1

        # Update categories
        for tag in article.tags:
            self.categories[tag] = self.categories.get(tag, 0) + 1

        # Update sentiment distribution
        if article.sentiment_score > 0.1:
            sentiment = "positive"
        elif article.sentiment_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        self.sentiment_distribution[sentiment] = \
            self.sentiment_distribution.get(sentiment, 0) + 1

        self.update_timestamp()

    def get_top_articles(self, n: int = 5) -> List[NewsArticle]:
        """Get top N articles by importance."""
        sorted_articles = sorted(
            self.top_articles,
            key=lambda a: a.calculate_importance(),
            reverse=True
        )
        return sorted_articles[:n]

    def generate_brief(self) -> str:
        """Generate a brief summary text."""
        brief = f"News Summary for {self.date.strftime('%Y-%m-%d')}\n"
        brief += f"Total Articles: {self.total_articles} "
        brief += f"({self.unread_articles} unread, {self.starred_articles} starred)\n\n"

        if self.sentiment_distribution:
            brief += "Sentiment: "
            for sentiment, count in self.sentiment_distribution.items():
                brief += f"{sentiment.capitalize()}: {count} "
            brief += "\n\n"

        if self.key_topics:
            brief += f"Key Topics: {', '.join(self.key_topics[:5])}\n"

        return brief
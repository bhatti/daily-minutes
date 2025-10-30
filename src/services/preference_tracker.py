"""User Preference Tracking Service.

This service manages user preferences using the RAG memory system.
It learns from user interactions and feedback to personalize the daily brief.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json

from src.memory.retrieval import MemoryRetriever
from src.memory.models import UserPreferenceMemory, MemoryType


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    CONTENT = "content"  # Content-related preferences (topics, sources)
    FORMATTING = "formatting"  # Presentation preferences (length, style)
    PRIORITY = "priority"  # Priority and importance preferences
    TIMING = "timing"  # Timing and frequency preferences
    GENERAL = "general"  # General preferences


class PreferenceTracker:
    """Track and manage user preferences using RAG memory.

    Features:
    - Store and retrieve preferences by category
    - Learn from user feedback
    - Update preferences with confidence scores
    - Export preferences for analysis
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        user_id: str = "default"
    ):
        """Initialize the preference tracker.

        Args:
            persist_directory: ChromaDB persistence directory
            user_id: User identifier for multi-user support
        """
        self.memory_retriever = MemoryRetriever(persist_directory=persist_directory)
        self.user_id = user_id

    async def store_preference(
        self,
        category: PreferenceCategory,
        key: str,
        value: Any,
        confidence: float = 1.0
    ) -> None:
        """Store a user preference in memory.

        Args:
            category: Preference category
            key: Preference key/name
            value: Preference value (can be any JSON-serializable type)
            confidence: Confidence score (0.0-1.0)
        """
        # Create preference memory
        preference = UserPreferenceMemory(
            id=f"pref-{self.user_id}-{key}-{datetime.now().timestamp()}",
            user_id=self.user_id,
            preference_key=key,
            preference_value=value,
            category=category.value,
            confidence_score=confidence
        )

        # Store in memory with embedding
        await self.memory_retriever.store_memory(preference)

    async def get_preferences(
        self,
        category: Optional[PreferenceCategory] = None
    ) -> List[UserPreferenceMemory]:
        """Get preferences, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of preference memories
        """
        # Get all USER_PREFERENCE memories
        all_prefs = await self.memory_retriever.memory_store.filter_by_type(
            MemoryType.USER_PREFERENCE
        )

        # Filter by user_id
        user_prefs = [
            p for p in all_prefs
            if p.metadata.get("user_id") == self.user_id
        ]

        # Filter by category if specified
        if category:
            user_prefs = [
                p for p in user_prefs
                if p.metadata.get("category") == category.value
            ]

        return user_prefs

    async def get_all_preferences(self) -> List[UserPreferenceMemory]:
        """Get all preferences for the user.

        Returns:
            List of all preference memories
        """
        return await self.get_preferences()

    async def get_preference(self, key: str) -> Optional[UserPreferenceMemory]:
        """Get a specific preference by key.

        Args:
            key: Preference key

        Returns:
            Preference memory or None if not found
        """
        all_prefs = await self.get_all_preferences()

        # Find preference with matching key (get most recent if multiple)
        matching_prefs = [
            p for p in all_prefs
            if p.metadata.get("key") == key
        ]

        if not matching_prefs:
            return None

        # Return most recent
        matching_prefs.sort(key=lambda p: p.timestamp, reverse=True)
        return matching_prefs[0]

    async def update_preference(
        self,
        key: str,
        value: Any,
        confidence: float
    ) -> None:
        """Update an existing preference or create new one.

        Args:
            key: Preference key
            value: New preference value
            confidence: Updated confidence score
        """
        # Get existing preference to determine category
        existing_pref = await self.get_preference(key)

        if existing_pref:
            category_str = existing_pref.metadata.get("category", "general")
            category = PreferenceCategory(category_str)
        else:
            category = PreferenceCategory.GENERAL

        # Store updated preference
        await self.store_preference(category, key, value, confidence)

    async def learn_from_feedback(
        self,
        feedback_type: str,
        context: Dict[str, Any]
    ) -> None:
        """Learn preferences from user feedback.

        Args:
            feedback_type: Type of feedback (liked_article, disliked_article, etc.)
            context: Context dictionary with relevant information
        """
        # Extract preference information from feedback
        if feedback_type == "liked_article":
            # User liked this content, store as preference
            topic = context.get("topic")
            source = context.get("source")

            if topic:
                await self.store_preference(
                    category=PreferenceCategory.CONTENT,
                    key="liked_topics",
                    value=[topic],
                    confidence=0.7  # Initial confidence from single feedback
                )

            if source:
                await self.store_preference(
                    category=PreferenceCategory.CONTENT,
                    key="preferred_sources",
                    value=[source],
                    confidence=0.7
                )

        elif feedback_type == "disliked_article":
            # User disliked this content
            topic = context.get("topic")

            if topic:
                await self.store_preference(
                    category=PreferenceCategory.CONTENT,
                    key="disliked_topics",
                    value=[topic],
                    confidence=0.7
                )

        # Additional feedback types can be added here

    async def get_recent_preferences(
        self,
        days: int = 30
    ) -> List[UserPreferenceMemory]:
        """Get preferences updated within the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent preferences
        """
        all_prefs = await self.get_all_preferences()

        # Filter by timestamp
        cutoff = datetime.now() - timedelta(days=days)
        recent_prefs = [
            p for p in all_prefs
            if p.timestamp >= cutoff
        ]

        return recent_prefs

    async def export_preferences(self) -> Dict[str, Any]:
        """Export all preferences as a dictionary.

        Returns:
            Dictionary of preferences grouped by category
        """
        all_prefs = await self.get_all_preferences()

        export = {}
        for pref in all_prefs:
            category = pref.metadata.get("category", "general")
            key = pref.metadata.get("key", "unknown")
            value = pref.metadata.get("value")
            confidence = pref.metadata.get("confidence", 0.0)

            if category not in export:
                export[category] = {}

            export[category][key] = {
                "value": value,
                "confidence": confidence,
                "timestamp": pref.timestamp.isoformat()
            }

        return export


# Singleton instance
_preference_tracker: Optional[PreferenceTracker] = None


def get_preference_tracker(
    persist_directory: str = "./chroma_data",
    user_id: str = "default"
) -> PreferenceTracker:
    """Get singleton PreferenceTracker instance.

    Args:
        persist_directory: ChromaDB persistence directory
        user_id: User identifier

    Returns:
        PreferenceTracker instance
    """
    global _preference_tracker

    if _preference_tracker is None:
        _preference_tracker = PreferenceTracker(
            persist_directory=persist_directory,
            user_id=user_id
        )

    return _preference_tracker

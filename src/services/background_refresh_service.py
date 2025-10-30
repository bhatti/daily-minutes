"""Background Refresh Service - Automatic periodic refresh of all data sources."""

import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime
from src.core.logging import get_logger
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class BackgroundRefreshService:
    """
    Service for background refresh of all data sources.

    Handles:
    - Automatic periodic refresh (news, weather, email, calendar)
    - Refresh tracking and logging
    - Non-blocking UI operations
    - Concurrent refresh prevention
    """

    def __init__(self):
        """Initialize background refresh service."""
        self.db_manager = get_db_manager()
        self._refresh_in_progress = False
        self._refresh_lock = asyncio.Lock()

    async def refresh_all_sources(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, bool]:
        """Refresh all data sources (news, weather, email, calendar).

        Args:
            progress_callback: Optional callback for progress updates (source, progress)

        Returns:
            Dict mapping source name to success status
        """
        # Fast rejection if refresh already in progress
        if self._refresh_in_progress:
            logger.warning("refresh_already_in_progress")
            return {}

        async with self._refresh_lock:
            # Double-check inside lock to prevent race condition
            if self._refresh_in_progress:
                logger.warning("refresh_already_in_progress")
                return {}

            self._refresh_in_progress = True

            try:
                results = {}
                sources = ['news', 'weather', 'email', 'calendar']
                total_sources = len(sources)

                for idx, source in enumerate(sources):
                    logger.info(f"background_refresh_starting", source=source)

                    if progress_callback:
                        progress = (idx + 1) / total_sources
                        progress_callback(source, progress)

                    success = await self.refresh_single_source(source)
                    results[source] = success

                    logger.info(f"background_refresh_completed",
                               source=source,
                               success=success)

                # Log overall refresh operation
                await self._log_refresh_operation(
                    source="all",
                    status="success" if all(results.values()) else "partial",
                    details={"results": results}
                )

                return results

            finally:
                self._refresh_in_progress = False

    async def refresh_single_source(self, source: str) -> bool:
        """Refresh a single data source.

        Args:
            source: Data source name (news, weather, email, calendar)

        Returns:
            True if refresh succeeded, False otherwise
        """
        try:
            if source == 'news':
                return await self._refresh_news()
            elif source == 'weather':
                return await self._refresh_weather()
            elif source == 'email':
                return await self._refresh_email()
            elif source == 'calendar':
                return await self._refresh_calendar()
            else:
                logger.error(f"unknown_source", source=source)
                return False

        except Exception as e:
            logger.error(f"refresh_failed",
                        source=source,
                        error=str(e))

            await self._log_refresh_operation(
                source=source,
                status="failed",
                details={"error": str(e)}
            )
            return False

    async def _refresh_news(self) -> bool:
        """Refresh news articles and generate AI summaries.

        Returns:
            True if refresh succeeded
        """
        from src.services.news_service import get_news_service
        from src.services.article_analyzer import get_article_analyzer

        try:
            news_service = get_news_service()
            articles = await news_service.fetch_all_news(max_articles=30)

            if articles:
                # Generate AI summaries for articles
                article_analyzer = get_article_analyzer()
                logger.info("generating_ai_summaries", count=len(articles))

                for article in articles:
                    # Generate AI analysis (TLDR + analysis)
                    try:
                        analysis = await article_analyzer.analyze_article(
                            title=article.title,
                            content=article.content or article.description,
                            url=str(article.url),
                            use_cache=True
                        )

                        # Populate AI fields on the article
                        article.ai_summary = analysis.get('analysis', '')
                        # Generate TLDR from analysis (first sentence)
                        if analysis.get('analysis'):
                            sentences = analysis['analysis'].split('.')
                            article.tldr = sentences[0].strip() + '.' if sentences else analysis['analysis']
                        article.key_learnings = analysis.get('key_learnings', [])

                        logger.debug("article_analyzed", title=article.title[:50])
                    except Exception as e:
                        logger.warning("article_analysis_failed",
                                     title=article.title[:50],
                                     error=str(e))
                        # Continue with other articles even if one fails

                # Save articles to database
                for article in articles:
                    await self.db_manager.save_article(article)

                # Save refresh timestamp
                await self.db_manager.set_setting(
                    "last_news_refresh",
                    datetime.now().isoformat()
                )

                logger.info("news_refresh_success", count=len(articles))
                return True
            else:
                logger.warning("news_refresh_no_articles")
                return False

        except Exception as e:
            logger.error("news_refresh_failed", error=str(e))
            return False

    async def _refresh_weather(self) -> bool:
        """Refresh weather data.

        Returns:
            True if refresh succeeded
        """
        from src.services.weather_service import get_weather_service
        from src.core.config_manager import get_config_manager

        try:
            weather_service = get_weather_service()
            config = get_config_manager()

            # Get default location from config or use fallback
            location = config.get("weather.default_location", "Seattle")

            weather_data = await weather_service.get_current_weather(location)

            if weather_data:
                # Convert WeatherData dataclass to dict for storage
                weather_dict = {
                    "location": weather_data.location,
                    "temperature": weather_data.temperature,
                    "feels_like": weather_data.feels_like,
                    "humidity": weather_data.humidity,
                    "pressure": weather_data.pressure,
                    "wind_speed": weather_data.wind_speed,
                    "wind_direction": weather_data.wind_direction,
                    "description": weather_data.description,
                    "icon": weather_data.icon,
                    "timestamp": weather_data.timestamp.isoformat(),
                    "sunrise": weather_data.sunrise.isoformat() if weather_data.sunrise else None,
                    "sunset": weather_data.sunset.isoformat() if weather_data.sunset else None,
                    "visibility": weather_data.visibility,
                    "uv_index": weather_data.uv_index,
                    "forecast": weather_data.forecast
                }

                # Store weather data in database cache
                await self.db_manager.set_cache('weather_data', weather_dict)

                # Save refresh timestamp
                await self.db_manager.set_setting(
                    "last_weather_refresh",
                    datetime.now().isoformat()
                )

                logger.info("weather_refresh_success", location=location,
                           temperature=weather_data.temperature)
                return True
            else:
                logger.warning("weather_refresh_no_data")
                return False

        except Exception as e:
            logger.error("weather_refresh_failed", error=str(e))
            return False

    async def _refresh_email(self) -> bool:
        """Refresh email messages and generate AI summaries.

        Returns:
            True if refresh succeeded
        """
        from src.services.email_service import get_email_service

        try:
            email_service = get_email_service()

            # Fetch emails
            emails = await email_service.fetch_emails(max_results=50)

            if emails:
                # Convert EmailMessage objects to dicts for JSON serialization
                emails_data = []
                for email in emails:
                    # Generate snippet if missing (first 150 chars of body)
                    snippet = email.snippet
                    if not snippet and hasattr(email, 'body') and email.body:
                        snippet = email.body[:150].strip()
                        if len(email.body) > 150:
                            snippet += "..."

                    # Generate AI summary for important emails (score > 0.7)
                    ai_summary = None
                    if email.importance_score > 0.7:
                        try:
                            # Use Ollama to generate summary
                            from src.services.ollama_service import get_ollama_service
                            ollama = get_ollama_service()

                            prompt = f"""Analyze this important email and extract key information:

Subject: {email.subject}
From: {email.sender}
Body: {email.body[:800] if hasattr(email, 'body') and email.body else 'N/A'}

Provide analysis in this format:

Summary: [2-3 sentences capturing the main points, requests, or issues. Be specific about what is being asked, reported, or announced.]

Key Points:
- [First key point or finding]
- [Second key point]
- [Additional points as relevant]

Action Items:
- [Specific action 1 if any]
- [Specific action 2 if any]

CRITICAL: Be direct and specific. Extract actual facts, requests, deadlines, or decisions from the email. NO meta descriptions like "The email discusses" - write what it actually says."""

                            response = await ollama.chat(
                                messages=[{"role": "user", "content": prompt}]
                                # Uses default model from OllamaService (reads from OLLAMA_MODEL env var)
                            )

                            if hasattr(response, 'content'):
                                response_text = response.content.strip()
                            elif isinstance(response, dict) and 'content' in response:
                                response_text = response['content'].strip()
                            else:
                                response_text = str(response)

                            # Extract just the Summary section for ai_summary field
                            # Full response has Summary, Key Points, Action Items
                            # We want the Summary part for preview
                            if 'Summary:' in response_text:
                                # Extract text after "Summary:" up to next section
                                summary_start = response_text.find('Summary:') + len('Summary:')
                                summary_text = response_text[summary_start:].split('\n\n')[0].strip()
                                ai_summary = summary_text
                            else:
                                # Fallback: use first 2-3 sentences
                                sentences = response_text.split('.')[:2]
                                ai_summary = '. '.join(sentences).strip() + '.'

                            logger.debug("email_ai_summary_generated",
                                       subject=email.subject[:50],
                                       summary_length=len(ai_summary))
                        except Exception as e:
                            logger.warning("email_ai_summary_failed",
                                         subject=email.subject[:50],
                                         error=str(e))

                    emails_data.append({
                        'id': email.id,
                        'subject': email.subject,
                        'sender': email.sender,
                        'received_at': email.received_at.isoformat() if email.received_at else None,
                        'body': email.body if hasattr(email, 'body') else snippet,
                        'snippet': snippet,
                        'ai_summary': ai_summary,
                        'importance_score': email.importance_score,
                        'has_action_items': email.has_action_items,
                        'action_items': email.action_items if hasattr(email, 'action_items') else [],
                        'is_read': email.is_read if hasattr(email, 'is_read') else False,
                        'labels': list(email.labels) if hasattr(email, 'labels') else []
                    })

                # Cache email data to database
                await self.db_manager.set_cache('emails_data', emails_data)

                # Save refresh timestamp
                await self.db_manager.set_setting(
                    "last_email_refresh",
                    datetime.now().isoformat()
                )

                logger.info("email_refresh_success", count=len(emails))
                return True
            else:
                logger.warning("email_refresh_no_emails")
                return False

        except Exception as e:
            logger.error("email_refresh_failed", error=str(e))
            return False

    async def _refresh_calendar(self) -> bool:
        """Refresh calendar events.

        Returns:
            True if refresh succeeded
        """
        from src.services.calendar_service import get_calendar_service

        try:
            calendar_service = get_calendar_service()

            # Fetch events for next 7 days
            from datetime import timedelta
            time_min = datetime.now()
            time_max = time_min + timedelta(days=7)
            events = await calendar_service.fetch_events(
                time_min=time_min,
                time_max=time_max,
                max_results=50
            )

            if events:
                # Convert CalendarEvent objects to dicts for JSON serialization
                calendar_data = [
                    {
                        'id': event.id,
                        'summary': event.summary,
                        'start_time': event.start_time.isoformat() if event.start_time else None,
                        'end_time': event.end_time.isoformat() if event.end_time else None,
                        'location': event.location if hasattr(event, 'location') else None,
                        'description': event.description if hasattr(event, 'description') else None,
                        'attendees': event.attendees if hasattr(event, 'attendees') else [],
                        'importance_score': event.importance_score if hasattr(event, 'importance_score') else 0.5,
                        'requires_preparation': event.requires_preparation if hasattr(event, 'requires_preparation') else False,
                        'is_focus_time': event.is_focus_time if hasattr(event, 'is_focus_time') else False
                    }
                    for event in events
                ]

                # Cache calendar data to database
                await self.db_manager.set_cache('calendar_data', calendar_data)

                # Save refresh timestamp
                await self.db_manager.set_setting(
                    "last_calendar_refresh",
                    datetime.now().isoformat()
                )

                logger.info("calendar_refresh_success", count=len(events))
                return True
            else:
                logger.warning("calendar_refresh_no_events")
                return False

        except Exception as e:
            logger.error("calendar_refresh_failed", error=str(e))
            return False

    async def _log_refresh_operation(
        self,
        source: str,
        status: str,
        details: Optional[Dict] = None
    ):
        """Log refresh operation to database.

        Args:
            source: Data source name
            status: Operation status (success, failed, partial)
            details: Optional details dict
        """
        try:
            # Use kv_store for refresh history
            history_key = f"refresh_history_{source}"
            history = await self.db_manager.get_setting(history_key, default=[])

            if not isinstance(history, list):
                history = []

            # Add new entry
            history.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "details": details or {}
            })

            # Keep only last 100 entries
            history = history[:100]

            await self.db_manager.set_setting(history_key, history)

        except Exception as e:
            logger.error("failed_to_log_refresh", error=str(e))

    async def get_refresh_history(
        self,
        source: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get refresh history for a source or all sources.

        Args:
            source: Optional source name (None = all sources)
            limit: Maximum number of entries to return

        Returns:
            List of refresh history entries
        """
        try:
            if source:
                history_key = f"refresh_history_{source}"
                history = await self.db_manager.get_setting(history_key, default=[])
                return history[:limit] if isinstance(history, list) else []
            else:
                # Get history from all sources
                all_history = []
                for src in ['news', 'weather', 'email', 'calendar', 'all']:
                    history_key = f"refresh_history_{src}"
                    history = await self.db_manager.get_setting(history_key, default=[])
                    if isinstance(history, list):
                        for entry in history:
                            entry['source'] = src
                            all_history.append(entry)

                # Sort by timestamp
                all_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                return all_history[:limit]

        except Exception as e:
            logger.error("failed_to_get_refresh_history", error=str(e))
            return []

    def is_refresh_in_progress(self) -> bool:
        """Check if refresh is currently in progress.

        Returns:
            True if refresh is running
        """
        return self._refresh_in_progress


# Singleton instance
_background_refresh_service = None


def get_background_refresh_service() -> BackgroundRefreshService:
    """Get singleton instance of BackgroundRefreshService.

    Returns:
        BackgroundRefreshService instance
    """
    global _background_refresh_service
    if _background_refresh_service is None:
        _background_refresh_service = BackgroundRefreshService()
    return _background_refresh_service

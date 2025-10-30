"""NewsService - Centralized news fetching and enrichment logic."""

import asyncio
from typing import List, Optional, Callable
from src.models.news import NewsArticle
from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.services.content_fetcher import ContentFetcher, get_content_fetcher
from src.services.article_analyzer import ArticleAnalyzer, get_article_analyzer
from src.core.config_manager import get_config_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


class NewsService:
    """
    Service for fetching and managing news articles.

    Handles:
    - Fetching from multiple sources (HackerNews, RSS, etc.)
    - Enriching articles with AI-powered analysis and key learnings
    - Respecting configuration limits
    - Progress reporting
    """

    def __init__(self):
        """Initialize NewsService."""
        self.config_mgr = get_config_manager()
        self.content_fetcher = get_content_fetcher()
        self.article_analyzer = get_article_analyzer()

    async def fetch_news_from_sources(
        self,
        progress_callback: Optional[Callable] = None
    ) -> List[NewsArticle]:
        """
        Fetch news from all enabled sources in parallel.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of news articles from all sources
        """
        num_sources = self.config_mgr.get_num_sources()
        per_source_limit = self.config_mgr.get_per_source_limit(num_sources)

        logger.info("fetch_news_start",
                   num_sources=num_sources,
                   per_source_limit=per_source_limit)

        all_articles = []

        # Define fetch functions for each source
        async def fetch_hackernews():
            if not self.config_mgr.is_source_enabled("hackernews"):
                logger.info("hackernews_disabled")
                return []
            try:
                hn_connector = HackerNewsConnector(max_stories=per_source_limit)
                articles = await hn_connector.execute_async()
                logger.info("fetched_hackernews", count=len(articles), limit=per_source_limit)
                return articles
            except Exception as e:
                logger.error("hackernews_fetch_failed", error=str(e))
                return []

        async def fetch_rss():
            if not self.config_mgr.is_source_enabled("rss"):
                logger.info("rss_disabled")
                return []
            try:
                rss_feed_configs = self.config_mgr.get_rss_feeds()
                if not rss_feed_configs:
                    logger.warning("no_rss_feeds_configured")
                    return []

                # Convert feed configs to RSSFeed objects
                from src.models.news import RSSFeed
                rss_feeds = []
                for config in rss_feed_configs:
                    # Skip commented-out feeds
                    if isinstance(config, str):
                        continue

                    feed = RSSFeed(
                        name=config.get("name", "Unknown"),
                        url=config.get("url"),
                        category=config.get("category", "general"),
                        max_articles=per_source_limit
                    )
                    rss_feeds.append(feed)

                # Create connector with configured feeds
                rss_connector = RSSConnector(
                    feeds=rss_feeds,
                    max_articles_per_feed=per_source_limit
                )
                articles = await rss_connector.fetch_all_feeds()
                logger.info("fetched_rss", count=len(articles), limit=per_source_limit, feeds=len(rss_feeds))
                return articles
            except Exception as e:
                logger.error("rss_fetch_failed", error=str(e))
                return []

        # Fetch all sources in parallel
        try:
            results = await asyncio.gather(
                fetch_hackernews(),
                fetch_rss(),
                return_exceptions=True
            )

            # Combine results from all sources
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error("source_fetch_exception", error=str(result))

            logger.info("parallel_fetch_complete", total_articles=len(all_articles))

        except Exception as e:
            logger.error("parallel_fetch_failed", error=str(e))

        return all_articles

    async def enrich_with_analysis(
        self,
        articles: List[NewsArticle],
        progress_callback: Optional[Callable] = None
    ) -> List[NewsArticle]:
        """
        Enrich articles with AI-powered analysis and key learnings.

        Args:
            articles: List of articles to enrich
            progress_callback: Optional callback for progress updates

        Returns:
            List of enriched articles with analysis and key learnings
        """
        if not articles:
            logger.info("no_articles_to_analyze")
            return articles

        # Get thread pool size from config
        content_threads = self.config_mgr.get_content_threads()
        semaphore = asyncio.Semaphore(content_threads)

        enriched_count = 0
        failed_count = 0

        async def analyze_with_progress(article: NewsArticle, index: int):
            """Analyze a single article with AI."""
            nonlocal enriched_count, failed_count

            async with semaphore:
                try:
                    # Update progress
                    if progress_callback:
                        await progress_callback(
                            0.4 + (0.3 * index / len(articles)),
                            f"Analyzing articles... ({index + 1}/{len(articles)})"
                        )

                    # First, try to fetch article content for better analysis
                    content_result = await self.content_fetcher.fetch_article(
                        str(article.url),
                        use_cache=True,
                        timeout=10
                    )

                    # Use content if available, otherwise just use title and existing description
                    content = None
                    content_status = content_result.get('status')

                    if content_status in ['success', 'cached']:
                        content = content_result.get('content', content_result.get('excerpt'))
                    elif content_status == 'blocked':
                        # Site is blocking access - skip analysis, just note it
                        error_msg = content_result.get('error', 'Site blocked')
                        article.description = f"🚫 {error_msg}\n\nAnalysis skipped for blocked site."
                        logger.info("article_blocked_skipped",
                                   url=str(article.url),
                                   error=error_msg)
                        failed_count += 1
                        return  # Skip to next article
                    elif content_status in ['timeout', 'ssl_error']:
                        # Temporary or SSL errors - log but continue with title-only analysis
                        logger.info("content_fetch_issue",
                                   url=str(article.url),
                                   status=content_status,
                                   error=content_result.get('error'))
                        # Will use title-only for analysis below

                    # Generate AI analysis
                    analysis_result = await self.article_analyzer.analyze_article(
                        title=article.title,
                        content=content,
                        url=str(article.url),
                        use_cache=True
                    )

                    # Check if analysis was successful (no error or has valid analysis despite error)
                    has_error = 'error' in analysis_result
                    has_valid_analysis = analysis_result.get('analysis') and analysis_result['analysis'] != 'Analysis failed'

                    if has_valid_analysis and not has_error:
                        # Format the analysis and key learnings into description
                        formatted_desc = analysis_result.get('analysis', 'No analysis available')

                        key_learnings = analysis_result.get('key_learnings', [])
                        if key_learnings:
                            formatted_desc += "\n\n**Key Learnings:**"
                            for learning in key_learnings[:3]:  # Limit to top 3
                                formatted_desc += f"\n• {learning}"

                        article.description = formatted_desc
                        enriched_count += 1
                        logger.info("article_analyzed",
                                   url=str(article.url),
                                   learnings_count=len(key_learnings))
                    else:
                        # Store error or fallback to original description
                        if has_error:
                            article.description = f"⚠️ Analysis unavailable: {analysis_result.get('error', 'Unknown error')[:100]}"
                            logger.warning("analysis_error",
                                         url=str(article.url),
                                         error=analysis_result.get('error'))
                        failed_count += 1

                except Exception as e:
                    # Log and store error in description
                    error_msg = str(e)
                    article.description = f"⚠️ Error analyzing article: {error_msg[:50]}"
                    logger.warning("analysis_failed", url=str(article.url), error=error_msg)
                    failed_count += 1

        # Analyze all articles in parallel with semaphore limiting concurrency
        tasks = [analyze_with_progress(article, i) for i, article in enumerate(articles)]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("articles_analyzed",
                   total=len(articles),
                   enriched=enriched_count,
                   failed=failed_count)

        return articles

    async def fetch_all_news(
        self,
        max_articles: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[NewsArticle]:
        """
        Fetch news from all sources and enrich with AI-powered analysis.

        Args:
            max_articles: Maximum number of articles to return (uses config default if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List of enriched news articles with AI analysis and key learnings
        """
        if max_articles is None:
            max_articles = self.config_mgr.get_max_articles()

        # Fetch from sources
        if progress_callback:
            await progress_callback(0.1, "Fetching news from sources...")

        articles = await self.fetch_news_from_sources(progress_callback)

        # Limit to requested count
        articles = articles[:max_articles]

        # Enrich with AI analysis and key learnings
        if progress_callback:
            await progress_callback(0.4, f"Analyzing {len(articles)} articles with AI...")

        articles = await self.enrich_with_analysis(articles, progress_callback)

        if progress_callback:
            await progress_callback(1.0, "Complete!")

        logger.info("fetch_all_news_complete",
                   total_articles=len(articles),
                   max_requested=max_articles)

        return articles


# Singleton instance
_news_service: Optional[NewsService] = None


def get_news_service() -> NewsService:
    """
    Get or create NewsService instance.

    Returns:
        NewsService instance
    """
    global _news_service
    if _news_service is None:
        _news_service = NewsService()
    return _news_service

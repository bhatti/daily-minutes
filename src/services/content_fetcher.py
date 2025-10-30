"""Article Content Fetcher service for downloading and caching article content."""

import httpx
import os
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.logging import get_logger
from src.database.sqlite_manager import SQLiteManager, get_db_manager

logger = get_logger(__name__)


class ContentFetcher:
    """Service for fetching and caching article content."""

    def __init__(self, db_manager: Optional[SQLiteManager] = None):
        """Initialize ContentFetcher.

        Args:
            db_manager: Optional SQLiteManager instance (uses singleton if not provided)
        """
        self.db_manager = db_manager or get_db_manager()
        self.default_timeout = 30  # seconds

        # Cache article content for 100 days (permanent data - doesn't change)
        # Can be cleared manually via Settings if needed
        self.default_cache_days = 100

        # Get SSL verification setting from environment
        env_verify = os.getenv('VERIFY_SSL', 'true').lower()
        self.verify_ssl = env_verify in ('true', '1', 'yes')

        if not self.verify_ssl:
            logger.warning("content_fetcher_ssl_disabled",
                         message="SSL verification disabled - not recommended for production")

    async def fetch_article(
        self,
        url: str,
        use_cache: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch article content from URL with optional caching.

        Args:
            url: Article URL to fetch
            use_cache: Whether to use cached content if available
            timeout: Request timeout in seconds (uses default if not provided)

        Returns:
            Dict with keys: status, url, content, excerpt, title, error (if error)
        """
        # Check if domain is blocked first
        is_blocked = await self.db_manager.is_domain_blocked(url)
        if is_blocked:
            logger.info("domain_blocked_skip", url=url)
            return {
                'status': 'blocked',
                'url': url,
                'error': 'Domain previously blocked - skipping to avoid repeated failures',
                'content': '',
                'excerpt': '',
                'cached': False
            }

        # Check cache first if enabled
        if use_cache:
            cached = await self.get_cached_article(url)
            if cached:
                logger.info("article_cache_hit", url=url)
                return {
                    'status': 'cached',
                    'url': url,
                    'content': cached.get('processed_content', ''),
                    'excerpt': cached.get('summary', ''),
                    'title': cached.get('title', ''),
                    'cached': True
                }

        # Fetch from web
        try:
            timeout_val = timeout or self.default_timeout

            # Configure SSL verification based on settings
            async with httpx.AsyncClient(timeout=timeout_val, verify=self.verify_ssl) as client:
                response = await client.get(url)
                response.raise_for_status()

                html = response.text

                # Extract text content
                content = self.extract_text(html)
                excerpt = self.generate_excerpt(content)

                # Try to extract title
                title = self.extract_title(html) or "Article"

                # Cache the content
                if use_cache:
                    await self.cache_article(
                        url=url,
                        title=title,
                        content=content,
                        excerpt=excerpt
                    )

                logger.info("article_fetched", url=url, content_length=len(content))

                return {
                    'status': 'success',
                    'url': url,
                    'content': content,
                    'excerpt': excerpt,
                    'title': title,
                    'cached': False
                }

        except httpx.HTTPStatusError as e:
            # Handle specific HTTP error codes
            status_code = e.response.status_code
            error_msg = ""

            if status_code == 403:
                error_msg = "Access forbidden (403) - Site may be blocking automated requests"
            elif status_code == 451:
                error_msg = "Content unavailable for legal reasons (451) - Blocked in your region"
            elif status_code == 429:
                error_msg = "Rate limited (429) - Too many requests to this site"
            elif status_code == 404:
                error_msg = "Article not found (404)"
            elif status_code >= 500:
                error_msg = f"Server error ({status_code}) - Site may be down"
            else:
                error_msg = f"HTTP error {status_code}"

            # Add to blocked domains if it's a blocking error (403, 451)
            if status_code in (403, 451):
                await self.db_manager.add_blocked_domain(
                    url=url,
                    reason=error_msg,
                    status_code=status_code,
                    auto_unblock_days=30  # Auto-retry after 30 days
                )

            logger.warning("article_blocked_or_error",
                          url=url,
                          status_code=status_code,
                          error=error_msg)

            return {
                'status': 'blocked' if status_code in (403, 451) else 'error',
                'url': url,
                'error': error_msg,
                'status_code': status_code,
                'content': '',
                'excerpt': '',
                'cached': False
            }

        except httpx.TimeoutException:
            error_msg = f"Request timed out after {timeout_val}s"
            logger.warning("article_fetch_timeout", url=url, timeout=timeout_val)
            return {
                'status': 'timeout',
                'url': url,
                'error': error_msg,
                'content': '',
                'excerpt': '',
                'cached': False
            }

        except (httpx.ConnectError, Exception) as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                error_msg = f"SSL certificate error - {str(e)[:100]}"
                logger.warning("article_ssl_error", url=url, error=str(e))
            else:
                error_msg = f"Connection error - {str(e)[:100]}"
                logger.warning("article_connection_error", url=url, error=str(e))
            return {
                'status': 'error',
                'url': url,
                'error': error_msg,
                'content': '',
                'excerpt': '',
                'cached': False
            }

        except Exception as e:
            logger.error("article_fetch_failed", url=url, error=str(e))
            return {
                'status': 'error',
                'url': url,
                'error': str(e),
                'content': '',
                'excerpt': '',
                'cached': False
            }

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Clean text content
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Try to find main content area
            main_content = soup.find('article') or soup.find('main') or soup.find('body')

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = ' '.join(text.split())

            return text

        except Exception as e:
            logger.error("text_extraction_failed", error=str(e))
            return ""

    def extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Page title or None
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Try various title sources
            title = None

            # Try meta og:title
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content')

            # Try h1
            if not title:
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)

            # Try <title> tag
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True)

            return title

        except Exception as e:
            logger.error("title_extraction_failed", error=str(e))
            return None

    def generate_excerpt(self, text: str, max_length: int = 200) -> str:
        """Generate excerpt from text.

        Args:
            text: Full text content
            max_length: Maximum excerpt length

        Returns:
            Excerpt with ellipsis if truncated
        """
        if not text:
            return ""

        # Clean and truncate
        clean_text = ' '.join(text.split())

        if len(clean_text) <= max_length:
            return clean_text

        # Find last complete word before max_length
        truncated = clean_text[:max_length]
        last_space = truncated.rfind(' ')

        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + "..."

    async def cache_article(
        self,
        url: str,
        title: str,
        content: str,
        excerpt: str,
        expires_in_days: Optional[int] = None
    ) -> int:
        """Cache article content in database.

        Args:
            url: Article URL
            title: Article title
            content: Full article content
            excerpt: Short excerpt/summary
            expires_in_days: Days until cache expires (uses default if not provided)

        Returns:
            Content ID
        """
        expires_days = expires_in_days or self.default_cache_days

        content_id = await self.db_manager.save_content(
            title=title,
            source="content_fetcher",
            url=url,
            content_type="article",
            processed_content=content,
            summary=excerpt,
            expires_in_days=expires_days
        )

        logger.info("article_cached", url=url, content_id=content_id)
        return content_id

    async def get_cached_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached article content.

        Args:
            url: Article URL

        Returns:
            Cached content dict or None if not found/expired
        """
        cached = await self.db_manager.get_content_by_url(url)

        if cached:
            # Check if expired
            expires_at = cached.get('expires_at')
            if expires_at:
                # Parse timestamp string
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    else:
                        expires_dt = expires_at

                    if expires_dt < datetime.now():
                        logger.info("article_cache_expired", url=url)
                        return None
                except Exception as e:
                    logger.warning("cache_expiry_check_failed", error=str(e))

            return cached

        return None


# Singleton instance
_content_fetcher: Optional[ContentFetcher] = None


def get_content_fetcher() -> ContentFetcher:
    """Get or create ContentFetcher instance.

    Returns:
        ContentFetcher instance
    """
    global _content_fetcher
    if _content_fetcher is None:
        _content_fetcher = ContentFetcher()
    return _content_fetcher

"""SQLite database manager for Daily Minutes."""

import aiosqlite
import sqlite3
import json
import hashlib
from ulid import ULID
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

from src.core.logging import get_logger
from src.models.news import NewsArticle

logger = get_logger(__name__)


class SQLiteManager:
    """Manage SQLite database operations with flexible content_identifier."""

    def __init__(self, db_path: str = "./data/daily_minutes.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # For in-memory databases, maintain a persistent connection
        self._is_memory = (db_path == ":memory:")
        self._conn = None
        self._initialized = False
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensure database directory exists."""
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize database with schema."""
        if self._initialized:
            return

        schema_path = Path(__file__).parent / "schema.sql"

        # For in-memory databases, create persistent connection
        if self._is_memory:
            self._conn = await aiosqlite.connect(self.db_path)
            db = self._conn
        else:
            db = await aiosqlite.connect(self.db_path)

        try:
            # Read and execute schema
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            await db.executescript(schema_sql)
            await db.commit()
        finally:
            # Only close if not in-memory (we keep in-memory connection open)
            if not self._is_memory:
                await db.close()

        self._initialized = True
        logger.info("database_initialized", path=self.db_path)

    async def close(self):
        """Close database connection (mainly for in-memory databases)."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection as async context manager.

        For in-memory databases, yields the persistent connection.
        For file-based databases, creates a new connection and closes it when done.
        """
        if self._is_memory:
            if not self._conn:
                raise RuntimeError("Database not initialized. Call initialize() first.")
            # Yield persistent connection without closing
            yield self._conn
        else:
            # Create new connection and close when done
            async with aiosqlite.connect(self.db_path) as db:
                yield db

    # =========================================================================
    # Identifier Generation
    # =========================================================================

    @staticmethod
    def generate_identifier(content_type: str, url: Optional[str] = None) -> str:
        """Generate a content_identifier using ULID for time-ordering.

        Args:
            content_type: Type of content (article, note, summary, etc.)
            url: Optional URL for web content

        Returns:
            Unique identifier string
        """
        if url:
            # For articles and web content, use the URL
            return url
        else:
            # For non-web content, generate ULID with type prefix
            # ULID is time-ordered, making it better for database indexing
            return f"{content_type}:{ULID()}"

    # =========================================================================
    # CONTENT Operations
    # =========================================================================

    async def get_content(self, content_identifier: str) -> Optional[Dict[str, Any]]:
        """Get content by identifier.

        Args:
            content_identifier: Content identifier

        Returns:
            Content dict or None if not found
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM content WHERE content_identifier = ?",
                (content_identifier,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                # Update last_accessed_at and access_count
                await db.execute(
                    """UPDATE content
                       SET last_accessed_at = CURRENT_TIMESTAMP,
                           access_count = access_count + 1
                       WHERE content_identifier = ?""",
                    (content_identifier,)
                )
                await db.commit()

                # Re-fetch to get updated access_count
                async with db.execute(
                    "SELECT * FROM content WHERE content_identifier = ?",
                    (content_identifier,)
                ) as cursor2:
                    updated_row = await cursor2.fetchone()
                    return dict(updated_row) if updated_row else None

    async def get_content_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get content by URL (convenience method).

        Args:
            url: Content URL

        Returns:
            Content dict or None if not found
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM content WHERE url = ?", (url,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                # Update access stats
                await db.execute(
                    """UPDATE content
                       SET last_accessed_at = CURRENT_TIMESTAMP,
                           access_count = access_count + 1
                       WHERE url = ?""",
                    (url,)
                )
                await db.commit()

                # Re-fetch to get updated access_count
                async with db.execute(
                    "SELECT * FROM content WHERE url = ?", (url,)
                ) as cursor2:
                    updated_row = await cursor2.fetchone()
                    return dict(updated_row) if updated_row else None

    async def save_content(
        self,
        title: str,
        source: str,
        content_identifier: Optional[str] = None,
        url: Optional[str] = None,
        content_type: str = "article",
        raw_content: Optional[str] = None,
        processed_content: Optional[str] = None,
        summary: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        published_at: Optional[datetime] = None,
        expires_in_days: int = 30
    ) -> int:
        """Save or update content.

        Args:
            title: Content title
            source: Source name
            content_identifier: Optional custom identifier (generated if not provided)
            url: Optional URL for web content
            content_type: Type of content
            raw_content: Original HTML/raw data
            processed_content: Clean text content
            summary: AI-generated summary
            key_points: List of key learnings
            metadata: Additional metadata as dict
            published_at: Original publish date
            expires_in_days: Days until content expires

        Returns:
            Content ID
        """
        # Generate identifier if not provided
        if not content_identifier:
            content_identifier = self.generate_identifier(content_type, url)

        # Calculate content hash
        content_hash = self._hash_content(processed_content or title)

        # Calculate expiry date
        expires_at = datetime.now() + timedelta(days=expires_in_days)

        metadata_json = json.dumps(metadata or {})
        key_points_json = json.dumps(key_points or [])

        async with self._get_connection() as db:
            await db.execute(
                """INSERT INTO content (
                    content_identifier, url, content_hash, content_type, title, source,
                    metadata, raw_content, processed_content, summary,
                    key_points, published_at, fetched_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(content_identifier) DO UPDATE SET
                    title = excluded.title,
                    processed_content = excluded.processed_content,
                    summary = excluded.summary,
                    key_points = excluded.key_points,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    content_identifier, url, content_hash, content_type, title, source,
                    metadata_json, raw_content, processed_content,
                    summary, key_points_json, published_at, expires_at
                )
            )
            await db.commit()

            # Get the content ID
            async with db.execute(
                "SELECT id FROM content WHERE content_identifier = ?",
                (content_identifier,)
            ) as cursor:
                row = await cursor.fetchone()
                content_id = row[0] if row else None

        logger.info("content_saved",
                   content_identifier=content_identifier,
                   content_id=content_id)
        return content_id

    async def save_article(self, article: NewsArticle, **kwargs) -> int:
        """Save a NewsArticle to database.

        Args:
            article: NewsArticle object
            **kwargs: Additional arguments for save_content

        Returns:
            Content ID
        """
        metadata = {
            "author": article.author,
            "tags": article.tags,
            "priority": article.priority if isinstance(article.priority, str) else article.priority.value,
            "relevance_score": article.relevance_score,
            "sentiment_score": article.sentiment_score,
            # Save AI-generated fields in metadata
            "ai_summary": article.ai_summary,
            "tldr": article.tldr,
        }

        # Convert URL to string (handles Pydantic HttpUrl types)
        url_str = str(article.url) if article.url else None

        return await self.save_content(
            content_identifier=url_str,  # Use URL as identifier for articles
            url=url_str,
            title=article.title,
            source=str(article.source),
            content_type="article",
            processed_content=article.description,
            summary=article.ai_summary,  # Save AI summary
            key_points=article.key_learnings,  # Save key learnings
            metadata=metadata,
            published_at=article.published_at,
            **kwargs
        )

    async def get_all_articles(self, limit: int = 100) -> List['NewsArticle']:
        """Get all cached articles from database.

        Args:
            limit: Maximum number of articles to retrieve (default: 100)

        Returns:
            List of NewsArticle objects ordered by published_at descending
        """
        from src.models.news import NewsArticle
        from src.models.base import Priority, DataSource
        from datetime import datetime

        articles = []

        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                """SELECT content_identifier, url, title, source, metadata, processed_content,
                          summary, key_points, published_at, fetched_at
                   FROM content
                   WHERE content_type = 'article'
                     AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                   ORDER BY published_at DESC
                   LIMIT ?""",
                (limit,)
            ) as cursor:
                async for row in cursor:
                    try:
                        # Parse metadata
                        metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'] or '{}')

                        # Parse key_points JSON
                        key_points = row['key_points'] if isinstance(row['key_points'], list) else json.loads(row['key_points'] or '[]')

                        # Convert source string to DataSource enum
                        source_str = row['source'] or 'hackernews'
                        try:
                            source = DataSource(source_str.lower())
                        except ValueError:
                            source = DataSource.HACKERNEWS

                        # Convert priority string to Priority enum
                        priority_str = metadata.get('priority', 'medium')
                        try:
                            priority = Priority(priority_str.lower()) if isinstance(priority_str, str) else priority_str
                        except ValueError:
                            priority = Priority.MEDIUM

                        # Parse timestamps
                        published_at = None
                        if row['published_at']:
                            if isinstance(row['published_at'], str):
                                published_at = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
                            else:
                                published_at = row['published_at']

                        # Create NewsArticle object
                        article = NewsArticle(
                            url=row['url'] or row['content_identifier'],
                            title=row['title'],
                            description=row['processed_content'] or '',
                            author=metadata.get('author'),
                            published_at=published_at,
                            source=source,
                            source_name=row['source'] or 'Unknown',
                            tags=metadata.get('tags', []),
                            priority=priority,
                            relevance_score=metadata.get('relevance_score', 0.5),
                            sentiment_score=metadata.get('sentiment_score', 0.0),
                            # Load AI-generated fields
                            ai_summary=row['summary'],
                            tldr=metadata.get('tldr'),
                            key_learnings=key_points
                        )

                        articles.append(article)

                    except Exception as e:
                        logger.error("failed_to_parse_article", error=str(e), content_id=row['content_identifier'])
                        continue

        logger.info("loaded_articles_from_cache", count=len(articles), limit=limit)
        return articles

    async def cleanup_expired_content(self) -> int:
        """Delete expired content.

        Returns:
            Number of deleted records
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                "DELETE FROM content WHERE expires_at < CURRENT_TIMESTAMP"
            )
            await db.commit()
            deleted = cursor.rowcount

        logger.info("expired_content_cleaned", deleted=deleted)
        return deleted

    async def cleanup_stale_content(
        self,
        days_old: int = 90,
        max_access: int = 5
    ) -> int:
        """Delete stale unused content.

        Args:
            days_old: Delete content not accessed in this many days
            max_access: Delete content with access count below this

        Returns:
            Number of deleted records
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                """DELETE FROM content
                   WHERE last_accessed_at < datetime('now', ?)
                     AND access_count < ?""",
                (f'-{days_old} days', max_access)
            )
            await db.commit()
            deleted = cursor.rowcount

        logger.info("stale_content_cleaned", deleted=deleted, days=days_old)
        return deleted

    # =========================================================================
    # KV Store Operations
    # =========================================================================

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        async with self._get_connection() as db:
            async with db.execute(
                "SELECT value FROM kv_store WHERE key = ? AND category = 'settings'",
                (key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    value = row[0]
                    # SQLite JSON columns auto-deserialize, check if already decoded
                    if isinstance(value, str):
                        return json.loads(value)
                    return value
                return default

    async def set_setting(self, key: str, value: Any):
        """Set a setting value.

        Args:
            key: Setting key
            value: Setting value (will be JSON serialized)
        """
        value_json = json.dumps(value)

        async with self._get_connection() as db:
            await db.execute(
                """INSERT INTO kv_store (key, value, category)
                   VALUES (?, ?, 'settings')
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       updated_at = CURRENT_TIMESTAMP""",
                (key, value_json)
            )
            await db.commit()

    async def set_encrypted_setting(
        self,
        key: str,
        value: str,
        pepper_key: Optional[str] = None
    ):
        """Set an encrypted setting value.

        Args:
            key: Setting key
            value: Plain text value to encrypt
            pepper_key: Optional pepper key for encryption (uses default if not provided)
        """
        import base64
        from src.security.encryption_service import get_encryption_service

        # Encrypt the value
        encryption_service = get_encryption_service(pepper_key=pepper_key)
        encrypted_data = encryption_service.encrypt_credential(value)

        # Convert bytes to base64 strings for JSON serialization
        encrypted_data_serializable = {
            'ciphertext': base64.b64encode(encrypted_data['ciphertext']).decode('utf-8'),
            'salt': base64.b64encode(encrypted_data['salt']).decode('utf-8'),
            'is_encrypted': encrypted_data['is_encrypted'],
            'algorithm': encrypted_data['algorithm'],
            'version': encrypted_data['version']
        }

        # Store encrypted data as JSON
        await self.set_setting(key, encrypted_data_serializable)

    async def get_encrypted_setting(
        self,
        key: str,
        pepper_key: Optional[str] = None
    ) -> Optional[str]:
        """Get and decrypt an encrypted setting value.

        Args:
            key: Setting key
            pepper_key: Optional pepper key for decryption (uses default if not provided)

        Returns:
            Decrypted plain text value or None if not found
        """
        import base64
        from src.security.encryption_service import get_encryption_service

        # Get encrypted data
        encrypted_data_serialized = await self.get_setting(key)
        if encrypted_data_serialized is None:
            return None

        # Convert base64 strings back to bytes
        encrypted_data = {
            'ciphertext': base64.b64decode(encrypted_data_serialized['ciphertext']),
            'salt': base64.b64decode(encrypted_data_serialized['salt']),
            'is_encrypted': encrypted_data_serialized['is_encrypted'],
            'algorithm': encrypted_data_serialized['algorithm'],
            'version': encrypted_data_serialized['version']
        }

        # Decrypt the value
        encryption_service = get_encryption_service(pepper_key=pepper_key)
        try:
            decrypted_value = encryption_service.decrypt_credential(encrypted_data)
            return decrypted_value
        except Exception as e:
            logger.error("decryption_failed", key=key, error=str(e))
            raise

    async def delete_setting(self, key: str):
        """Delete a setting from kv_store.

        Args:
            key: Setting key to delete
        """
        async with self._get_connection() as db:
            await db.execute(
                "DELETE FROM kv_store WHERE key = ?",
                (key,)
            )
            await db.commit()

    async def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings from kv_store.

        Returns:
            Dictionary mapping keys to values
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT key, value FROM kv_store"
            )
            rows = await cursor.fetchall()

        result = {}
        for key, value_json in rows:
            try:
                result[key] = json.loads(value_json)
            except json.JSONDecodeError:
                # If not JSON, store as-is
                result[key] = value_json

        return result

    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        async with self._get_connection() as db:
            async with db.execute(
                """SELECT value FROM kv_store
                   WHERE key = ? AND category = 'cache'
                     AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)""",
                (key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    value = row[0]
                    # SQLite JSON columns auto-deserialize, check if already decoded
                    if isinstance(value, str):
                        return json.loads(value)
                    return value
                return None

    async def set_cache(
        self,
        key: str,
        value: Any,
        expires_in_seconds: Optional[int] = None
    ):
        """Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            expires_in_seconds: Expiration time in seconds
        """
        value_json = json.dumps(value)

        async with self._get_connection() as db:
            if expires_in_seconds:
                # Use SQLite datetime arithmetic for consistent comparison
                await db.execute(
                    """INSERT INTO kv_store (key, value, category, expires_at)
                       VALUES (?, ?, 'cache', datetime('now', ? || ' seconds'))
                       ON CONFLICT(key) DO UPDATE SET
                           value = excluded.value,
                           expires_at = excluded.expires_at,
                           updated_at = CURRENT_TIMESTAMP""",
                    (key, value_json, f'+{expires_in_seconds}')
                )
            else:
                # No expiration
                await db.execute(
                    """INSERT INTO kv_store (key, value, category, expires_at)
                       VALUES (?, ?, 'cache', NULL)
                       ON CONFLICT(key) DO UPDATE SET
                           value = excluded.value,
                           expires_at = NULL,
                           updated_at = CURRENT_TIMESTAMP""",
                    (key, value_json)
                )
            await db.commit()

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_content_stats(self) -> Dict[str, Any]:
        """Get content statistics.

        Returns:
            Statistics dict
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row

            # Total counts
            async with db.execute(
                "SELECT COUNT(*) as total FROM content"
            ) as cursor:
                total_row = await cursor.fetchone()
                total = total_row[0] if total_row else 0

            # By source
            async with db.execute(
                """SELECT source, content_type, COUNT(*) as count
                   FROM content
                   GROUP BY source, content_type"""
            ) as cursor:
                by_source = [dict(row) async for row in cursor]

            # Expired count
            async with db.execute(
                "SELECT COUNT(*) as count FROM v_expired_content"
            ) as cursor:
                expired_row = await cursor.fetchone()
                expired = expired_row[0] if expired_row else 0

            # Stale count
            async with db.execute(
                "SELECT COUNT(*) as count FROM v_stale_content"
            ) as cursor:
                stale_row = await cursor.fetchone()
                stale = stale_row[0] if stale_row else 0

        return {
            "total_content": total,
            "by_source": by_source,
            "expired_content": expired,
            "stale_content": stale
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content.

        Args:
            content: Content to hash

        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(content.encode()).hexdigest()

    # =========================================================================
    # Blocked Domains Operations
    # =========================================================================

    async def is_domain_blocked(self, url: str) -> bool:
        """Check if a domain is blocked.

        Args:
            url: Full URL to check

        Returns:
            True if domain is blocked, False otherwise
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        if not domain:
            return False

        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row

            # Check if domain is blocked and not expired
            async with db.execute(
                """SELECT domain FROM blocked_domains
                   WHERE domain = ?
                   AND (unblock_after IS NULL OR unblock_after > CURRENT_TIMESTAMP)""",
                (domain,)
            ) as cursor:
                row = await cursor.fetchone()
                return row is not None

    async def add_blocked_domain(
        self,
        url: str,
        reason: str,
        status_code: int,
        auto_unblock_days: Optional[int] = 30
    ):
        """Add or update a blocked domain.

        Args:
            url: Full URL that was blocked
            reason: Reason for blocking (e.g., "403 Forbidden")
            status_code: HTTP status code
            auto_unblock_days: Days until auto-unblock (None = permanent)
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        if not domain:
            logger.warning("blocked_domain_invalid_url", url=url)
            return

        async with self._get_connection() as db:
            if auto_unblock_days:
                unblock_after = f"datetime('now', '+{auto_unblock_days} days')"
            else:
                unblock_after = "NULL"

            # Insert or update
            await db.execute(
                f"""INSERT INTO blocked_domains (domain, reason, status_code, unblock_after, block_count)
                   VALUES (?, ?, ?, {unblock_after}, 1)
                   ON CONFLICT(domain) DO UPDATE SET
                       reason = excluded.reason,
                       status_code = excluded.status_code,
                       block_count = block_count + 1,
                       last_blocked_at = CURRENT_TIMESTAMP""",
                (domain, reason, status_code)
            )
            await db.commit()

        logger.info("domain_blocked",
                   domain=domain,
                   reason=reason,
                   status_code=status_code,
                   auto_unblock_days=auto_unblock_days)

    async def get_blocked_domains(self) -> list:
        """Get all blocked domains.

        Returns:
            List of blocked domain dicts
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                """SELECT domain, reason, status_code, block_count,
                          first_blocked_at, last_blocked_at, unblock_after
                   FROM blocked_domains
                   WHERE unblock_after IS NULL OR unblock_after > CURRENT_TIMESTAMP
                   ORDER BY last_blocked_at DESC"""
            ) as cursor:
                return [dict(row) async for row in cursor]

    async def unblock_domain(self, domain: str):
        """Manually unblock a domain.

        Args:
            domain: Domain to unblock
        """
        async with self._get_connection() as db:
            await db.execute(
                "DELETE FROM blocked_domains WHERE domain = ?",
                (domain,)
            )
            await db.commit()

        logger.info("domain_unblocked", domain=domain)

    async def cleanup_expired_blocks(self):
        """Remove domains whose unblock_after date has passed."""
        async with self._get_connection() as db:
            await db.execute(
                """DELETE FROM blocked_domains
                   WHERE unblock_after IS NOT NULL
                   AND unblock_after <= CURRENT_TIMESTAMP"""
            )
            await db.commit()

        logger.info("expired_blocks_cleaned")

    # =========================================================================
    # Cache Clearing Operations (Manual control for testing/model changes)
    # =========================================================================

    async def clear_news_list_cache(self):
        """Clear cached news lists (HackerNews, RSS feeds).

        This clears the list of top stories/feeds but keeps article content and analysis.
        Useful when you want fresh news without losing analyzed articles.
        """
        async with self._get_connection() as db:
            # Clear HackerNews and RSS list caches from kv_store
            await db.execute(
                """DELETE FROM kv_store
                   WHERE category = 'cache'
                   AND (key LIKE 'hackernews_%' OR key LIKE 'rss_%')"""
            )
            await db.commit()

        logger.info("news_list_cache_cleared")

    async def clear_article_content_cache(self):
        """Clear cached article content.

        This removes downloaded article text but keeps metadata and analysis.
        Useful when you want to re-fetch article content.
        """
        async with self._get_connection() as db:
            # Clear content table where content_type is 'article'
            await db.execute(
                """UPDATE content
                   SET raw_content = NULL,
                       processed_content = NULL
                   WHERE content_type = 'article'"""
            )
            await db.commit()

        logger.info("article_content_cache_cleared")

    async def clear_article_analysis_cache(self):
        """Clear cached article AI analysis.

        This removes AI-generated summaries and key learnings.
        Useful when you change models or want to regenerate analysis.
        """
        async with self._get_connection() as db:
            # Clear analysis data from content table
            await db.execute(
                """UPDATE content
                   SET summary = NULL,
                       key_points = NULL,
                       ai_metadata = '{}'
                   WHERE content_type = 'article'"""
            )

            # Also clear analysis caches from kv_store
            await db.execute(
                """DELETE FROM kv_store
                   WHERE category = 'cache'
                   AND key LIKE 'analysis_%'"""
            )
            await db.commit()

        logger.info("article_analysis_cache_cleared")

    async def clear_weather_cache(self):
        """Clear cached weather data.

        This forces fresh weather data on next fetch.
        """
        async with self._get_connection() as db:
            await db.execute(
                """DELETE FROM kv_store
                   WHERE category = 'cache'
                   AND key LIKE 'weather_%'"""
            )
            await db.commit()

        logger.info("weather_cache_cleared")

    async def clear_all_caches(self):
        """Clear ALL caches (news lists, articles, analysis, weather).

        Nuclear option: Start fresh with everything.
        """
        async with self._get_connection() as db:
            # Clear all cache entries from kv_store
            await db.execute(
                """DELETE FROM kv_store WHERE category = 'cache'"""
            )

            # Clear content and analysis from content table
            await db.execute(
                """UPDATE content
                   SET raw_content = NULL,
                       processed_content = NULL,
                       summary = NULL,
                       key_points = NULL,
                       ai_metadata = '{}'
                   WHERE content_type = 'article'"""
            )
            await db.commit()

        logger.info("all_caches_cleared")

    async def get_cache_age_hours(self, category: str = "cache") -> Optional[float]:
        """Get the age of the oldest non-expired cache entry in hours.

        This helps determine if a cache refresh is needed.

        Args:
            category: Cache category to check (default: "cache")

        Returns:
            Age in hours of the oldest cache entry, or None if no cache exists
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT MIN(created_at) as oldest
                   FROM kv_store
                   WHERE category = ?
                   AND (expires_at IS NULL OR expires_at > ?)""",
                (category, datetime.now().isoformat())
            )
            row = await cursor.fetchone()

            if row and row['oldest']:
                oldest_time = datetime.fromisoformat(row['oldest'])
                age_seconds = (datetime.now() - oldest_time).total_seconds()
                return age_seconds / 3600  # Convert to hours

            return None

    async def get_news_cache_age_hours(self) -> Optional[float]:
        """Get age of news list cache in hours.

        Returns:
            Age in hours, or None if no news cache exists
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT MAX(created_at) as newest
                   FROM kv_store
                   WHERE category = 'cache'
                   AND (key LIKE 'hackernews_%' OR key LIKE 'rss_%')
                   AND (expires_at IS NULL OR expires_at > ?)""",
                (datetime.now().isoformat(),)
            )
            row = await cursor.fetchone()

            if row and row['newest']:
                newest_time = datetime.fromisoformat(row['newest'])
                age_seconds = (datetime.now() - newest_time).total_seconds()
                return age_seconds / 3600

            return None

    async def get_weather_cache_age_hours(self) -> Optional[float]:
        """Get age of weather cache in hours.

        Returns:
            Age in hours, or None if no weather cache exists
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT MAX(created_at) as newest
                   FROM kv_store
                   WHERE category = 'cache'
                   AND key LIKE 'weather_%'
                   AND (expires_at IS NULL OR expires_at > ?)""",
                (datetime.now().isoformat(),)
            )
            row = await cursor.fetchone()

            if row and row['newest']:
                newest_time = datetime.fromisoformat(row['newest'])
                age_seconds = (datetime.now() - newest_time).total_seconds()
                return age_seconds / 3600

            return None

    async def vacuum(self):
        """Vacuum database to reclaim space."""
        async with self._get_connection() as db:
            await db.execute("VACUUM")
            await db.commit()

        logger.info("database_vacuumed")

    async def get_database_size(self) -> Tuple[int, str]:
        """Get database file size.

        Returns:
            Tuple of (size_bytes, size_human_readable)
        """
        size_bytes = Path(self.db_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb < 1:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_mb < 1024:
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = f"{size_mb / 1024:.1f} GB"

        return size_bytes, size_str


# Singleton instance
_db_manager: Optional[SQLiteManager] = None


def get_db_manager() -> SQLiteManager:
    """Get or create database manager instance.

    Reads database path from SQLITE_DB_PATH environment variable,
    defaults to ./data/daily_minutes.db

    Returns:
        SQLiteManager instance
    """
    import os
    global _db_manager
    if _db_manager is None:
        db_path = os.getenv('SQLITE_DB_PATH', './data/daily_minutes.db')
        _db_manager = SQLiteManager(db_path=db_path)
        logger.info("db_manager_initialized", db_path=db_path)
    return _db_manager

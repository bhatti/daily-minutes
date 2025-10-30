"""Cache models for efficient data storage and retrieval."""

import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from src.models.base import BaseModel


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class CacheEntry(BaseModel):
    """Model for a cache entry."""

    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")

    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    last_accessed: datetime = Field(default_factory=datetime.now, description="Last access time")

    access_count: int = Field(0, ge=0, description="Number of accesses")
    size_bytes: int = Field(0, ge=0, description="Size in bytes")

    tags: List[str] = Field(default_factory=list, description="Cache tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Entry metadata")

    is_compressed: bool = Field(False, description="Data is compressed")
    compression_ratio: float = Field(1.0, gt=0, description="Compression ratio")

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def is_stale(self, max_age_seconds: int) -> bool:
        """Check if cache entry is stale based on age."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > max_age_seconds

    def access(self) -> None:
        """Record cache access."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.update_timestamp()

    def calculate_priority(self, strategy: CacheStrategy) -> float:
        """Calculate eviction priority based on strategy."""
        if strategy == CacheStrategy.LRU:
            # Lower priority for recently used
            return -(datetime.now() - self.last_accessed).total_seconds()

        elif strategy == CacheStrategy.LFU:
            # Higher priority for frequently used
            return self.access_count

        elif strategy == CacheStrategy.FIFO:
            # Lower priority for older entries
            return -(datetime.now() - self.created_at).total_seconds()

        elif strategy == CacheStrategy.TTL:
            # Lower priority for entries closer to expiration
            if self.expires_at:
                return (self.expires_at - datetime.now()).total_seconds()
            return float("inf")

        return 0.0

    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def get_size_kb(self) -> float:
        """Get size in kilobytes."""
        return self.size_bytes / 1024

    def get_effective_size(self) -> int:
        """Get effective size considering compression."""
        if self.is_compressed:
            return int(self.size_bytes / self.compression_ratio)
        return self.size_bytes


class CacheStore(BaseModel):
    """Model for cache store management."""

    store_id: str = Field(..., description="Cache store ID")
    name: str = Field(..., description="Store name")

    strategy: CacheStrategy = Field(CacheStrategy.LRU, description="Eviction strategy")
    max_size_mb: int = Field(100, gt=0, description="Maximum cache size in MB")
    max_entries: int = Field(10000, gt=0, description="Maximum number of entries")
    default_ttl_seconds: int = Field(3600, gt=0, description="Default TTL in seconds")

    entries: Dict[str, CacheEntry] = Field(default_factory=dict, description="Cache entries")

    # Statistics
    hits: int = Field(0, ge=0, description="Cache hits")
    misses: int = Field(0, ge=0, description="Cache misses")
    evictions: int = Field(0, ge=0, description="Number of evictions")
    total_size_bytes: int = Field(0, ge=0, description="Total size in bytes")

    # Performance metrics
    average_access_time_ms: float = Field(0.0, ge=0.0, description="Average access time")
    last_eviction: Optional[datetime] = Field(None, description="Last eviction time")
    last_cleanup: Optional[datetime] = Field(None, description="Last cleanup time")

    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self.entries.get(key)

        if not entry:
            self.misses += 1
            return None

        if entry.is_expired():
            del self.entries[key]
            self.misses += 1
            return None

        entry.access()
        self.hits += 1
        return entry.value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Put value in cache."""
        # Calculate size (simplified)
        size_bytes = len(json.dumps(value, default=str))

        # Check if eviction needed
        if len(self.entries) >= self.max_entries:
            self.evict()

        while self.total_size_bytes + size_bytes > self.max_size_mb * 1024 * 1024:
            self.evict()

        # Create entry
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size_bytes,
            tags=tags or []
        )

        self.entries[key] = entry
        self.total_size_bytes += size_bytes
        self.update_timestamp()

    def evict(self) -> None:
        """Evict entry based on strategy."""
        if not self.entries:
            return

        # Find entry to evict
        entries_with_priority = [
            (key, entry.calculate_priority(self.strategy))
            for key, entry in self.entries.items()
        ]

        # Sort by priority (lowest priority gets evicted)
        entries_with_priority.sort(key=lambda x: x[1])
        key_to_evict = entries_with_priority[0][0]

        # Evict
        entry = self.entries[key_to_evict]
        self.total_size_bytes -= entry.size_bytes
        del self.entries[key_to_evict]
        self.evictions += 1
        self.last_eviction = datetime.now()

    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        if key in self.entries:
            entry = self.entries[key]
            self.total_size_bytes -= entry.size_bytes
            del self.entries[key]
            return True
        return False

    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries with matching tags."""
        count = 0
        keys_to_remove = []

        for key, entry in self.entries.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if self.invalidate(key):
                count += 1

        return count

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        count = 0
        keys_to_remove = []

        for key, entry in self.entries.items():
            if entry.is_expired():
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if self.invalidate(key):
                count += 1

        self.last_cleanup = datetime.now()
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries.clear()
        self.total_size_bytes = 0
        self.update_timestamp()

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def get_size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self.entries),
            "size_mb": f"{self.get_size_mb():.2f}",
            "max_size_mb": self.max_size_mb,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.get_hit_rate():.2%}",
            "evictions": self.evictions,
            "strategy": self.strategy if isinstance(self.strategy, str) else self.strategy.value,
            "last_eviction": self.last_eviction.isoformat() if self.last_eviction else None,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
        }


class CacheManager(BaseModel):
    """Model for managing multiple cache stores."""

    stores: Dict[str, CacheStore] = Field(default_factory=dict, description="Cache stores")
    default_store: str = Field("default", description="Default store name")

    # Global settings
    enable_compression: bool = Field(False, description="Enable compression")
    enable_encryption: bool = Field(False, description="Enable encryption")
    enable_stats: bool = Field(True, description="Enable statistics")

    # Global statistics
    total_requests: int = Field(0, ge=0, description="Total cache requests")
    total_hits: int = Field(0, ge=0, description="Total cache hits")
    total_misses: int = Field(0, ge=0, description="Total cache misses")

    def create_store(
        self,
        name: str,
        strategy: CacheStrategy = CacheStrategy.LRU,
        max_size_mb: int = 100,
        max_entries: int = 10000,
        default_ttl_seconds: int = 3600
    ) -> CacheStore:
        """Create a new cache store."""
        store = CacheStore(
            store_id=name,
            name=name,
            strategy=strategy,
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            default_ttl_seconds=default_ttl_seconds
        )

        self.stores[name] = store
        return store

    def get_store(self, name: Optional[str] = None) -> Optional[CacheStore]:
        """Get cache store by name."""
        store_name = name or self.default_store
        return self.stores.get(store_name)

    def get(self, key: str, store: Optional[str] = None) -> Optional[Any]:
        """Get value from cache."""
        self.total_requests += 1

        cache_store = self.get_store(store)
        if not cache_store:
            self.total_misses += 1
            return None

        value = cache_store.get(key)
        if value is not None:
            self.total_hits += 1
        else:
            self.total_misses += 1

        return value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        store: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Put value in cache."""
        cache_store = self.get_store(store)
        if not cache_store:
            cache_store = self.create_store(store or self.default_store)

        cache_store.put(key, value, ttl_seconds, tags)

    def invalidate_all(self, key: str) -> int:
        """Invalidate key across all stores."""
        count = 0
        for store in self.stores.values():
            if store.invalidate(key):
                count += 1
        return count

    def cleanup_all(self) -> int:
        """Clean up expired entries in all stores."""
        count = 0
        for store in self.stores.values():
            count += store.cleanup_expired()
        return count

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        total_entries = sum(len(store.entries) for store in self.stores.values())
        total_size_mb = sum(store.get_size_mb() for store in self.stores.values())

        return {
            "stores": len(self.stores),
            "total_entries": total_entries,
            "total_size_mb": f"{total_size_mb:.2f}",
            "total_requests": self.total_requests,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "global_hit_rate": f"{self.total_hits / max(self.total_requests, 1):.2%}",
            "stores_stats": {
                name: store.get_statistics()
                for name, store in self.stores.items()
            }
        }


class CachedResult(BaseModel):
    """Model for cached operation results."""

    operation: str = Field(..., description="Operation name")
    params: Dict[str, Any] = Field(..., description="Operation parameters")
    result: Any = Field(..., description="Cached result")

    cache_key: str = Field(..., description="Cache key")
    cache_hit: bool = Field(False, description="Was cache hit")
    cache_store: str = Field("default", description="Cache store used")

    execution_time_ms: float = Field(0.0, ge=0.0, description="Execution time if computed")
    cache_time_saved_ms: float = Field(0.0, ge=0.0, description="Time saved by cache")

    def calculate_efficiency(self) -> float:
        """Calculate cache efficiency."""
        if not self.cache_hit or self.execution_time_ms == 0:
            return 0.0

        return self.cache_time_saved_ms / self.execution_time_ms
-- Daily Minutes SQLite Schema
-- Flexible design with JSON columns, timestamps, and indexes
-- Easy cleanup and extensibility

PRAGMA foreign_keys = ON;

-- =============================================================================
-- CONTENT: Universal content storage (articles, notes, etc.)
--
-- USAGE PATTERNS:
-- - content_identifier: Universal business key for all content
--   * For articles: Use URL (https://example.com/article)
--   * For notes: Use ULID (note:01ARZ3NDEKTSV4RRFFQ69G5FAV)
--   * For summaries: Use hash or custom ID (summary:abc123)
--   * For todos: Use ULID (todo:01ARZ3NDEKTSV4RRFFQ69G5FAV)
-- - url: Optional field, only populate when content has a web URL
-- - Always query by content_identifier for lookups
-- - ULID provides time-ordering and better index performance than UUID
-- =============================================================================

CREATE TABLE IF NOT EXISTS content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity (universal business key)
    content_identifier TEXT UNIQUE NOT NULL,  -- URL for articles, UUID for notes, custom ID, etc.
    url TEXT,                       -- Optional URL (nullable for non-web content)
    content_hash TEXT,              -- SHA256 for deduplication
    content_type TEXT DEFAULT 'article',  -- article, note, summary, etc.

    -- Metadata (flexible JSON for extensibility)
    title TEXT NOT NULL,
    source TEXT,                    -- hackernews, rss, manual, etc.
    metadata JSON DEFAULT '{}',     -- author, tags, priority, scores, custom fields

    -- Content
    raw_content TEXT,               -- Original HTML/raw data
    processed_content TEXT,         -- Clean text
    summary TEXT,                   -- AI-generated summary/excerpt
    key_points JSON DEFAULT '[]',   -- AI-extracted learnings

    -- AI Processing status
    ai_metadata JSON DEFAULT '{}',  -- processing_status, models_used, etc.

    -- Timestamps (ALL tables need these!)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP,         -- Original publish date
    fetched_at TIMESTAMP,           -- When we fetched it
    expires_at TIMESTAMP,           -- For automatic cleanup
    last_accessed_at TIMESTAMP,     -- Usage tracking

    -- Stats
    access_count INTEGER DEFAULT 0
);

-- Indexes for fast queries on timestamps
CREATE INDEX IF NOT EXISTS idx_content_created ON content(created_at);
CREATE INDEX IF NOT EXISTS idx_content_updated ON content(updated_at);
CREATE INDEX IF NOT EXISTS idx_content_expires ON content(expires_at);
CREATE INDEX IF NOT EXISTS idx_content_published ON content(published_at);
CREATE INDEX IF NOT EXISTS idx_content_accessed ON content(last_accessed_at);

-- Other indexes
CREATE INDEX IF NOT EXISTS idx_content_identifier ON content(content_identifier);  -- Primary business key
CREATE INDEX IF NOT EXISTS idx_content_url ON content(url);                         -- For URL lookups
CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type);
CREATE INDEX IF NOT EXISTS idx_content_source ON content(source);
CREATE INDEX IF NOT EXISTS idx_content_hash ON content(content_hash);

-- Trigger to auto-update updated_at
CREATE TRIGGER IF NOT EXISTS update_content_timestamp
AFTER UPDATE ON content
FOR EACH ROW
BEGIN
    UPDATE content SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- =============================================================================
-- KV_STORE: Flexible key-value storage for settings, cache, etc.
-- =============================================================================

CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value JSON,                     -- Any JSON data
    category TEXT DEFAULT 'general', -- settings, cache, temp
    expires_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kv_category ON kv_store(category);
CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_store(expires_at);
CREATE INDEX IF NOT EXISTS idx_kv_created ON kv_store(created_at);

CREATE TRIGGER IF NOT EXISTS update_kv_timestamp
AFTER UPDATE ON kv_store
FOR EACH ROW
BEGIN
    UPDATE kv_store SET updated_at = CURRENT_TIMESTAMP WHERE key = OLD.key;
END;

-- =============================================================================
-- TASKS: Todos/Tasks
-- =============================================================================

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, cancelled
    priority TEXT DEFAULT 'medium', -- low, medium, high, urgent

    -- Flexible metadata
    metadata JSON DEFAULT '{}',     -- tags, assignee, custom fields

    -- Relationships
    linked_content_id INTEGER,
    parent_task_id INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (linked_content_id) REFERENCES content(id) ON DELETE SET NULL,
    FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_task_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_task_due ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_task_created ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_task_updated ON tasks(updated_at);
CREATE INDEX IF NOT EXISTS idx_task_completed ON tasks(completed_at);
CREATE INDEX IF NOT EXISTS idx_task_linked ON tasks(linked_content_id);

CREATE TRIGGER IF NOT EXISTS update_task_timestamp
AFTER UPDATE ON tasks
FOR EACH ROW
BEGIN
    UPDATE tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- =============================================================================
-- EVENTS: Calendar/Schedule
-- =============================================================================

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT NOT NULL,
    description TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    all_day BOOLEAN DEFAULT 0,
    location TEXT,

    -- Source tracking
    source TEXT,                    -- google_calendar, outlook, manual
    external_id TEXT,

    -- Flexible metadata
    metadata JSON DEFAULT '{}',     -- attendees, recurrence, reminders

    -- Relationships
    linked_content_id INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (linked_content_id) REFERENCES content(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_start ON events(start_time);
CREATE INDEX IF NOT EXISTS idx_event_end ON events(end_time);
CREATE INDEX IF NOT EXISTS idx_event_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_event_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_event_updated ON events(updated_at);
CREATE INDEX IF NOT EXISTS idx_event_external ON events(external_id);

CREATE TRIGGER IF NOT EXISTS update_event_timestamp
AFTER UPDATE ON events
FOR EACH ROW
BEGIN
    UPDATE events SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- =============================================================================
-- ACTIVITY_LOG: Track all operations
-- =============================================================================

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    action TEXT NOT NULL,           -- create, update, delete, fetch, process
    entity_type TEXT,               -- content, task, event
    entity_id INTEGER,
    details JSON DEFAULT '{}',
    source TEXT,                    -- ui, api, background_job

    -- Timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_action ON activity_log(action);
CREATE INDEX IF NOT EXISTS idx_log_entity ON activity_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_log_created ON activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_log_source ON activity_log(source);

-- =============================================================================
-- BLOCKED_DOMAINS: Track domains that block access (403, 451, etc.)
-- =============================================================================

CREATE TABLE IF NOT EXISTS blocked_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    domain TEXT UNIQUE NOT NULL,        -- e.g., "example.com"
    reason TEXT,                        -- "403 Forbidden", "451 Unavailable"
    status_code INTEGER,                -- HTTP status code
    block_count INTEGER DEFAULT 1,      -- Number of times blocked

    -- First and last blocked timestamps
    first_blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Metadata
    metadata JSON DEFAULT '{}',         -- User notes, retry_after, etc.

    -- Auto-unblock after X days
    unblock_after TIMESTAMP             -- Null = permanent, or set date to auto-retry
);

CREATE INDEX IF NOT EXISTS idx_blocked_domain ON blocked_domains(domain);
CREATE INDEX IF NOT EXISTS idx_blocked_first ON blocked_domains(first_blocked_at);
CREATE INDEX IF NOT EXISTS idx_blocked_last ON blocked_domains(last_blocked_at);
CREATE INDEX IF NOT EXISTS idx_blocked_unblock ON blocked_domains(unblock_after);

-- Trigger to auto-update last_blocked_at
CREATE TRIGGER IF NOT EXISTS update_blocked_domain_timestamp
AFTER UPDATE ON blocked_domains
FOR EACH ROW
BEGIN
    UPDATE blocked_domains SET last_blocked_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- =============================================================================
-- CLEANUP VIEWS
-- =============================================================================

-- Expired content
CREATE VIEW IF NOT EXISTS v_expired_content AS
SELECT * FROM content
WHERE expires_at IS NOT NULL
  AND expires_at < CURRENT_TIMESTAMP;

-- Stale content (not accessed in 90 days, low usage)
CREATE VIEW IF NOT EXISTS v_stale_content AS
SELECT * FROM content
WHERE last_accessed_at < datetime('now', '-90 days')
  AND access_count < 5;

-- Old content (created more than 90 days ago)
CREATE VIEW IF NOT EXISTS v_old_content AS
SELECT * FROM content
WHERE created_at < datetime('now', '-90 days');

-- Content statistics
CREATE VIEW IF NOT EXISTS v_content_stats AS
SELECT
    source,
    content_type,
    COUNT(*) as count,
    AVG(access_count) as avg_access,
    MAX(created_at) as latest_created,
    MAX(published_at) as latest_published,
    MIN(expires_at) as earliest_expiry
FROM content
GROUP BY source, content_type;

-- =============================================================================
-- SCHEMA VERSION
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema with timestamps, indexes, and JSON support');

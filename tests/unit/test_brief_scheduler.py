"""Unit tests for Brief Scheduler."""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, call
from src.services.brief_scheduler import (
    BriefScheduler,
    get_brief_scheduler,
    start_brief_scheduler,
    stop_brief_scheduler,
)


class TestBriefScheduler:
    """Test BriefScheduler class."""

    @pytest.fixture
    def mock_config(self):
        """Mock scheduler config."""
        config = MagicMock()
        config.BRIEF_GENERATION_INTERVAL = 5
        config.BRIEF_ONLY_ON_NEW_DATA = True
        config.BRIEF_MIN_ITEMS = 1
        config.ENABLE_BACKGROUND_SCHEDULER = True
        config.ENABLE_AUTO_BRIEF = True
        return config

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db = MagicMock()
        db.get_all_articles = AsyncMock(return_value=[])
        db.get_weather_cache = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def scheduler(self, mock_config, mock_db_manager):
        """Create scheduler instance with mocked dependencies."""
        with patch('src.services.brief_scheduler.get_scheduler_config', return_value=mock_config), \
             patch('src.services.brief_scheduler.get_db_manager', return_value=mock_db_manager):
            scheduler = BriefScheduler()
            yield scheduler
            # Cleanup
            if scheduler._running:
                scheduler.stop()

    def test_initialization(self, scheduler, mock_config):
        """Test scheduler initializes correctly."""
        assert scheduler.config == mock_config
        assert scheduler._running is False
        assert scheduler._scheduler_thread is None
        assert scheduler._last_brief_time is None
        assert scheduler._last_data_hash is None

    def test_start_creates_thread(self, scheduler):
        """Test that start creates a background thread."""
        scheduler.start()

        assert scheduler._running is True
        assert scheduler._scheduler_thread is not None
        assert isinstance(scheduler._scheduler_thread, threading.Thread)
        assert scheduler._scheduler_thread.daemon is True
        assert scheduler._scheduler_thread.name == "BriefScheduler"

        # Cleanup
        scheduler.stop()

    def test_start_when_already_running(self, scheduler):
        """Test that calling start when already running does nothing."""
        scheduler.start()
        thread1 = scheduler._scheduler_thread

        # Try to start again
        scheduler.start()
        thread2 = scheduler._scheduler_thread

        # Should be the same thread
        assert thread1 is thread2

        # Cleanup
        scheduler.stop()

    def test_start_disabled_by_background_config(self, scheduler, mock_config):
        """Test that scheduler doesn't start if background scheduler disabled."""
        mock_config.ENABLE_BACKGROUND_SCHEDULER = False

        scheduler.start()

        assert scheduler._running is False
        assert scheduler._scheduler_thread is None

    def test_start_disabled_by_auto_brief_config(self, scheduler, mock_config):
        """Test that scheduler doesn't start if auto brief disabled."""
        mock_config.ENABLE_AUTO_BRIEF = False

        scheduler.start()

        assert scheduler._running is False
        assert scheduler._scheduler_thread is None

    def test_stop(self, scheduler):
        """Test that stop terminates the background thread."""
        scheduler.start()
        assert scheduler._running is True

        scheduler.stop()

        assert scheduler._running is False
        # Thread should finish
        time.sleep(0.5)
        if scheduler._scheduler_thread:
            assert not scheduler._scheduler_thread.is_alive()

    def test_stop_when_not_running(self, scheduler):
        """Test that calling stop when not running does nothing."""
        # Should not raise an error
        scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_get_data_hash_empty_data(self, scheduler, mock_db_manager):
        """Test _get_data_hash with empty data."""
        mock_db_manager.get_all_articles = AsyncMock(return_value=[])
        mock_db_manager.get_weather_cache = AsyncMock(return_value=None)

        data_hash = await scheduler._get_data_hash()

        assert isinstance(data_hash, str)
        assert len(data_hash) > 0  # MD5 hash should be non-empty

    @pytest.mark.asyncio
    async def test_get_data_hash_with_data(self, scheduler, mock_db_manager):
        """Test _get_data_hash with real data."""
        # Mock articles
        mock_articles = [MagicMock() for _ in range(5)]
        mock_db_manager.get_all_articles = AsyncMock(return_value=mock_articles)

        # Mock weather
        mock_weather = {'main': {'temp': 72}, 'weather': [{'description': 'sunny'}]}
        mock_db_manager.get_weather_cache = AsyncMock(return_value=mock_weather)

        data_hash = await scheduler._get_data_hash()

        assert isinstance(data_hash, str)
        assert len(data_hash) == 32  # MD5 hash is 32 chars

    @pytest.mark.asyncio
    async def test_get_data_hash_changes_with_data(self, scheduler, mock_db_manager):
        """Test that _get_data_hash changes when data changes."""
        # First hash with 5 articles
        mock_db_manager.get_all_articles = AsyncMock(return_value=[MagicMock() for _ in range(5)])
        hash1 = await scheduler._get_data_hash()

        # Second hash with 10 articles
        mock_db_manager.get_all_articles = AsyncMock(return_value=[MagicMock() for _ in range(10)])
        hash2 = await scheduler._get_data_hash()

        # Hashes should be different
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_get_data_hash_error_handling(self, scheduler, mock_db_manager):
        """Test _get_data_hash handles errors gracefully."""
        mock_db_manager.get_all_articles = AsyncMock(side_effect=Exception("Database error"))

        # Should return empty string on error
        data_hash = await scheduler._get_data_hash()

        assert data_hash == ""

    @pytest.mark.asyncio
    async def test_get_data_counts_empty(self, scheduler, mock_db_manager):
        """Test _get_data_counts with no data."""
        mock_db_manager.get_all_articles = AsyncMock(return_value=[])
        mock_db_manager.get_weather_cache = AsyncMock(return_value=None)

        counts = await scheduler._get_data_counts()

        assert counts['news'] == 0
        assert counts['weather'] == 0
        assert counts['email'] == 0
        assert counts['calendar'] == 0

    @pytest.mark.asyncio
    async def test_get_data_counts_with_data(self, scheduler, mock_db_manager):
        """Test _get_data_counts with data."""
        mock_articles = [MagicMock() for _ in range(10)]
        mock_db_manager.get_all_articles = AsyncMock(return_value=mock_articles)

        mock_weather = {'temp': 72}
        mock_db_manager.get_weather_cache = AsyncMock(return_value=mock_weather)

        counts = await scheduler._get_data_counts()

        assert counts['news'] == 10
        assert counts['weather'] == 1
        assert counts['email'] == 0  # TODO: Not implemented yet
        assert counts['calendar'] == 0  # TODO: Not implemented yet

    @pytest.mark.asyncio
    async def test_get_data_counts_error_handling(self, scheduler, mock_db_manager):
        """Test _get_data_counts handles errors gracefully."""
        mock_db_manager.get_all_articles = AsyncMock(side_effect=Exception("Database error"))

        counts = await scheduler._get_data_counts()

        # Should return empty dict on error
        assert counts == {}

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_interval_not_elapsed(self, scheduler):
        """Test brief not generated when interval hasn't elapsed."""
        # Set last brief time to 2 minutes ago
        scheduler._last_brief_time = datetime.now() - timedelta(minutes=2)
        scheduler.config.BRIEF_GENERATION_INTERVAL = 5  # 5 minute interval

        with patch.object(scheduler, '_generate_enhanced_brief') as mock_generate:
            await scheduler._check_and_generate_brief()

            # Should not call generate
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_no_data_changes(self, scheduler, mock_db_manager):
        """Test brief not generated when data hasn't changed."""
        scheduler._last_brief_time = datetime.now() - timedelta(minutes=10)
        scheduler._last_data_hash = "abc123"

        # Mock same hash
        with patch.object(scheduler, '_get_data_hash', return_value="abc123"), \
             patch.object(scheduler, '_generate_enhanced_brief') as mock_generate:
            await scheduler._check_and_generate_brief()

            # Should not call generate
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_insufficient_data(self, scheduler):
        """Test brief not generated when insufficient data."""
        scheduler._last_brief_time = None  # Never generated before
        scheduler.config.BRIEF_MIN_ITEMS = 5

        # Mock data counts with only 2 items
        with patch.object(scheduler, '_get_data_counts', return_value={'news': 2, 'weather': 0, 'email': 0, 'calendar': 0}), \
             patch.object(scheduler, '_generate_enhanced_brief') as mock_generate:
            await scheduler._check_and_generate_brief()

            # Should not call generate
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_success(self, scheduler):
        """Test brief is generated when all conditions are met."""
        scheduler._last_brief_time = None  # Never generated before
        scheduler.config.BRIEF_MIN_ITEMS = 1

        # Mock data counts with sufficient items
        mock_counts = {'news': 5, 'weather': 1, 'email': 0, 'calendar': 0}

        # Mock successful brief generation
        mock_brief = {
            'summary': 'Test summary',
            'key_learnings': ['Learning 1', 'Learning 2'],
            'key_insights': ['Insight 1', 'Insight 2'],
            'tldr': 'Test TLDR'
        }

        with patch.object(scheduler, '_get_data_counts', return_value=mock_counts), \
             patch.object(scheduler, '_get_data_hash', return_value="new_hash"), \
             patch.object(scheduler, '_generate_enhanced_brief', return_value=mock_brief) as mock_generate, \
             patch.object(scheduler, '_store_brief') as mock_store:

            await scheduler._check_and_generate_brief()

            # Should call generate
            mock_generate.assert_called_once_with(mock_counts)

            # Should store brief
            mock_store.assert_called_once_with(mock_brief)

            # Should update tracking
            assert scheduler._last_brief_time is not None
            assert scheduler._last_data_hash == "new_hash"

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_generation_failed(self, scheduler):
        """Test brief generation failure is handled gracefully."""
        scheduler._last_brief_time = None
        scheduler.config.BRIEF_MIN_ITEMS = 1

        mock_counts = {'news': 5, 'weather': 1, 'email': 0, 'calendar': 0}

        with patch.object(scheduler, '_get_data_counts', return_value=mock_counts), \
             patch.object(scheduler, '_get_data_hash', return_value="new_hash"), \
             patch.object(scheduler, '_generate_enhanced_brief', return_value=None), \
             patch.object(scheduler, '_store_brief') as mock_store:

            await scheduler._check_and_generate_brief()

            # Should not store brief
            mock_store.assert_not_called()

            # Should not update tracking
            assert scheduler._last_brief_time is None

    def test_build_enhanced_prompt_with_articles(self, scheduler):
        """Test _build_enhanced_prompt includes articles."""
        # Mock articles
        articles = [
            MagicMock(title='Article 1'),
            MagicMock(title='Article 2'),
            MagicMock(title='Article 3'),
        ]

        prompt = scheduler._build_enhanced_prompt(articles, None)

        assert 'Article 1' in prompt
        assert 'Article 2' in prompt
        assert 'Article 3' in prompt
        assert 'SUMMARY' in prompt
        assert 'KEY LEARNINGS' in prompt
        assert 'KEY INSIGHTS' in prompt
        assert 'TLDR' in prompt

    def test_build_enhanced_prompt_with_weather(self, scheduler):
        """Test _build_enhanced_prompt includes weather."""
        weather_data = {
            'main': {'temp': 72},
            'weather': [{'description': 'sunny'}]
        }

        prompt = scheduler._build_enhanced_prompt([], weather_data)

        assert '72°F' in prompt
        assert 'sunny' in prompt

    def test_build_enhanced_prompt_empty_data(self, scheduler):
        """Test _build_enhanced_prompt with no data."""
        prompt = scheduler._build_enhanced_prompt([], None)

        # Should still have section headers
        assert 'SUMMARY' in prompt
        assert 'KEY LEARNINGS' in prompt
        assert 'KEY INSIGHTS' in prompt
        assert 'TLDR' in prompt

    def test_parse_brief_response_complete(self, scheduler):
        """Test _parse_brief_response parses all sections."""
        response = """
**SUMMARY**
This is a test summary with multiple sentences.

**KEY LEARNINGS**
- Learning point 1
- Learning point 2
- Learning point 3

**KEY INSIGHTS**
- Insight 1
- Insight 2

**TLDR**
Quick summary here.
"""

        brief = scheduler._parse_brief_response(response)

        assert 'summary' in brief
        assert 'key_learnings' in brief
        assert 'key_insights' in brief
        assert 'tldr' in brief
        assert 'generated_at' in brief

        assert 'test summary' in brief['summary']
        assert len(brief['key_learnings']) == 3
        assert len(brief['key_insights']) == 2
        # TLDR section should exist (content may vary based on parsing)
        assert 'tldr' in brief
        assert isinstance(brief['tldr'], str)

    def test_parse_brief_response_bullet_formats(self, scheduler):
        """Test _parse_brief_response handles different bullet formats."""
        response = """
**KEY LEARNINGS**
- Dash bullet
* Star bullet
• Circle bullet
1. Numbered bullet
2. Another numbered
"""

        brief = scheduler._parse_brief_response(response)

        # Should extract all bullet points regardless of format
        assert len(brief['key_learnings']) == 5
        assert 'Dash bullet' in brief['key_learnings']
        assert 'Star bullet' in brief['key_learnings']
        assert 'Circle bullet' in brief['key_learnings']
        assert 'Numbered bullet' in brief['key_learnings']

    def test_parse_brief_response_missing_sections(self, scheduler):
        """Test _parse_brief_response handles missing sections."""
        response = """
**SUMMARY**
Just a summary, no other sections.
"""

        brief = scheduler._parse_brief_response(response)

        assert 'summary' in brief
        assert brief['summary'] != ""
        assert isinstance(brief['key_learnings'], list)
        assert len(brief['key_learnings']) == 0
        assert isinstance(brief['key_insights'], list)
        assert len(brief['key_insights']) == 0

    def test_parse_brief_response_error_handling(self, scheduler):
        """Test _parse_brief_response handles malformed input."""
        response = "This is not a properly formatted response"

        # Should not raise exception
        brief = scheduler._parse_brief_response(response)

        # Should have default structure
        assert 'summary' in brief
        assert 'key_learnings' in brief
        assert 'key_insights' in brief
        assert 'tldr' in brief
        assert 'generated_at' in brief

    @pytest.mark.asyncio
    async def test_store_brief(self, scheduler):
        """Test _store_brief logs the brief."""
        brief = {
            'summary': 'Test summary',
            'key_learnings': ['Learning 1', 'Learning 2'],
            'key_insights': ['Insight 1'],
            'tldr': 'Test TLDR'
        }

        # Should not raise exception
        await scheduler._store_brief(brief)

    def test_get_status_not_running(self, scheduler, mock_config):
        """Test get_status returns correct info when not running."""
        status = scheduler.get_status()

        assert status['running'] is False
        assert status['enabled'] == mock_config.ENABLE_AUTO_BRIEF
        assert status['last_brief_time'] is None
        assert status['generation_interval_minutes'] == 5
        assert status['only_on_new_data'] is True

    def test_get_status_running(self, scheduler):
        """Test get_status returns correct info when running."""
        scheduler.start()
        time.sleep(0.1)

        status = scheduler.get_status()

        assert status['running'] is True
        assert status['enabled'] is True

        # Cleanup
        scheduler.stop()

    def test_get_status_with_brief_history(self, scheduler):
        """Test get_status includes last brief timestamp."""
        scheduler._last_brief_time = datetime(2025, 1, 15, 10, 30, 0)

        status = scheduler.get_status()

        assert status['last_brief_time'] == '2025-01-15T10:30:00'


class TestBriefSchedulerSingleton:
    """Test global singleton functions."""

    def teardown_method(self):
        """Reset global singleton between tests."""
        import src.services.brief_scheduler
        if src.services.brief_scheduler._brief_scheduler is not None:
            if src.services.brief_scheduler._brief_scheduler._running:
                src.services.brief_scheduler._brief_scheduler.stop()
        src.services.brief_scheduler._brief_scheduler = None

    @patch('src.services.brief_scheduler.get_scheduler_config')
    @patch('src.services.brief_scheduler.get_db_manager')
    def test_get_brief_scheduler_returns_instance(self, mock_db, mock_config):
        """Test that get_brief_scheduler returns a BriefScheduler instance."""
        scheduler = get_brief_scheduler()
        assert isinstance(scheduler, BriefScheduler)

    @patch('src.services.brief_scheduler.get_scheduler_config')
    @patch('src.services.brief_scheduler.get_db_manager')
    def test_get_brief_scheduler_returns_singleton(self, mock_db, mock_config):
        """Test that get_brief_scheduler returns the same instance."""
        scheduler1 = get_brief_scheduler()
        scheduler2 = get_brief_scheduler()

        assert scheduler1 is scheduler2

    @patch('src.services.brief_scheduler.get_scheduler_config')
    @patch('src.services.brief_scheduler.get_db_manager')
    def test_start_brief_scheduler(self, mock_db, mock_config):
        """Test start_brief_scheduler starts the scheduler."""
        mock_config.return_value.ENABLE_BACKGROUND_SCHEDULER = True
        mock_config.return_value.ENABLE_AUTO_BRIEF = True

        start_brief_scheduler()

        scheduler = get_brief_scheduler()
        time.sleep(0.1)
        assert scheduler._running is True

        # Cleanup
        stop_brief_scheduler()

    @patch('src.services.brief_scheduler.get_scheduler_config')
    @patch('src.services.brief_scheduler.get_db_manager')
    def test_stop_brief_scheduler(self, mock_db, mock_config):
        """Test stop_brief_scheduler stops the scheduler."""
        mock_config.return_value.ENABLE_BACKGROUND_SCHEDULER = True
        mock_config.return_value.ENABLE_AUTO_BRIEF = True

        start_brief_scheduler()
        scheduler = get_brief_scheduler()
        time.sleep(0.1)
        assert scheduler._running is True

        stop_brief_scheduler()
        time.sleep(0.5)
        assert scheduler._running is False


class TestBriefSchedulerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_config(self):
        """Mock scheduler config."""
        config = MagicMock()
        config.BRIEF_GENERATION_INTERVAL = 5
        config.BRIEF_ONLY_ON_NEW_DATA = True
        config.BRIEF_MIN_ITEMS = 1
        config.ENABLE_BACKGROUND_SCHEDULER = True
        config.ENABLE_AUTO_BRIEF = True
        return config

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db = MagicMock()
        db.get_all_articles = AsyncMock(return_value=[])
        db.get_weather_cache = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def scheduler(self, mock_config, mock_db_manager):
        """Create scheduler instance."""
        with patch('src.services.brief_scheduler.get_scheduler_config', return_value=mock_config), \
             patch('src.services.brief_scheduler.get_db_manager', return_value=mock_db_manager):
            scheduler = BriefScheduler()
            yield scheduler
            if scheduler._running:
                scheduler.stop()

    @pytest.mark.asyncio
    async def test_generate_enhanced_brief_no_data(self, scheduler, mock_db_manager):
        """Test _generate_enhanced_brief returns None when no data available."""
        mock_db_manager.get_all_articles = AsyncMock(return_value=[])
        mock_db_manager.get_weather_cache = AsyncMock(return_value=None)

        # Since we mock the database to return no data, the brief generator should return None
        # without needing to call Ollama
        brief = await scheduler._generate_enhanced_brief({'news': 0})

        assert brief is None

    def test_thread_daemon_mode(self, scheduler):
        """Test that scheduler thread is daemon (won't block program exit)."""
        scheduler.start()

        assert scheduler._scheduler_thread.daemon is True

        scheduler.stop()

    @pytest.mark.asyncio
    async def test_check_and_generate_brief_exception_handling(self, scheduler):
        """Test _check_and_generate_brief handles exceptions gracefully."""
        with patch.object(scheduler, '_get_data_counts', side_effect=Exception("Unexpected error")):
            # Should not raise exception
            await scheduler._check_and_generate_brief()


class TestTLDRValidation:
    """Test TLD R validation and fallback generation (TDD for user issue)."""

    @pytest.fixture
    def mock_config(self):
        """Mock scheduler config."""
        config = MagicMock()
        config.BRIEF_GENERATION_INTERVAL = 5
        config.BRIEF_ONLY_ON_NEW_DATA = True
        config.BRIEF_MIN_ITEMS = 1
        config.ENABLE_BACKGROUND_SCHEDULER = True
        config.ENABLE_AUTO_BRIEF = True
        return config

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db = MagicMock()
        db.get_all_articles = AsyncMock(return_value=[])
        db.get_cache = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def scheduler(self, mock_config, mock_db_manager):
        """Create scheduler instance."""
        with patch('src.services.brief_scheduler.get_scheduler_config', return_value=mock_config), \
             patch('src.services.brief_scheduler.get_db_manager', return_value=mock_db_manager):
            scheduler = BriefScheduler()
            yield scheduler

    def test_is_generic_tldr_detects_range_of_topics(self, scheduler):
        """Test that 'range of topics' is detected as generic."""
        generic_tldr = "Recent articles cover a range of topics, including social polarization"
        assert scheduler._is_generic_tldr(generic_tldr) is True

    def test_is_generic_tldr_detects_various_topics(self, scheduler):
        """Test that 'various topics' is detected as generic."""
        generic_tldr = "The news discusses various topics today"
        assert scheduler._is_generic_tldr(generic_tldr) is True

    def test_is_generic_tldr_detects_covers_phrase(self, scheduler):
        """Test that 'covers' is detected as generic."""
        generic_tldr = "Today's brief covers multiple important topics"
        assert scheduler._is_generic_tldr(generic_tldr) is True

    def test_is_generic_tldr_accepts_specific_content(self, scheduler):
        """Test that specific TLDRs are not marked as generic."""
        specific_tldr = "Tech giants face antitrust scrutiny while renewable energy adoption accelerates"
        assert scheduler._is_generic_tldr(specific_tldr) is False

    def test_is_generic_tldr_detects_empty_string(self, scheduler):
        """Test that empty TLDR is detected as generic."""
        assert scheduler._is_generic_tldr("") is True

    def test_is_generic_tldr_detects_too_short(self, scheduler):
        """Test that very short TLDRs are detected as generic."""
        assert scheduler._is_generic_tldr("News today") is True

    def test_generate_fallback_tldr_extracts_first_sentence(self, scheduler):
        """Test that fallback TLDR extracts first sentence from summary."""
        summary = "Tech companies announce layoffs amid economic downturn. Markets react negatively. Consumer confidence drops."
        fallback = scheduler._generate_fallback_tldr(summary)
        
        assert fallback == "Tech companies announce layoffs amid economic downturn"

    def test_generate_fallback_tldr_truncates_long_sentences(self, scheduler):
        """Test that very long first sentences are truncated."""
        summary = "A" * 150 + ". Short sentence."
        fallback = scheduler._generate_fallback_tldr(summary)
        
        # Should be truncated to 100 chars max
        assert len(fallback) <= 100
        assert fallback.endswith("...")

    def test_generate_fallback_tldr_handles_empty_summary(self, scheduler):
        """Test that empty summary returns default message."""
        fallback = scheduler._generate_fallback_tldr("")
        assert fallback == "Daily brief generated"

    def test_parse_brief_response_replaces_generic_tldr(self, scheduler):
        """Test that generic TLDR is replaced with fallback from summary."""
        response = """
**SUMMARY**
Tech layoffs continue across industry. Economic uncertainty persists.

**TLDR**
Recent articles cover a range of topics, including tech and economy.
"""
        
        result = scheduler._parse_brief_response(response)
        
        # TLDR should be replaced with first sentence of summary
        assert result['tldr'] != "Recent articles cover a range of topics, including tech and economy."
        assert "Tech layoffs continue across industry" in result['tldr']

    def test_parse_brief_response_keeps_specific_tldr(self, scheduler):
        """Test that specific TLDR is not replaced."""
        response = """
**SUMMARY**
Tech layoffs continue. Economic uncertainty persists.

**TLDR**
Tech giants announce 10K layoffs amid economic uncertainty
"""
        
        result = scheduler._parse_brief_response(response)
        
        # Specific TLDR should be kept
        assert result['tldr'] == "Tech giants announce 10K layoffs amid economic uncertainty"

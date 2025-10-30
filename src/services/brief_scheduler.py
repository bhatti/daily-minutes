"""Brief Scheduler - Periodic AI brief/summary generation.

This service runs in the background and automatically generates AI briefs when:
- MCP data has changed since last brief
- Configured interval has elapsed (default 5 minutes)
- Minimum data requirements are met

The brief includes enhanced sections:
- Summary
- Key Learnings
- Key Insights
- TLDR

All briefs are stored in the database for later retrieval.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from src.core.logging import get_logger
from src.core.scheduler_config import get_scheduler_config
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class BriefScheduler:
    """Periodic scheduler for AI brief generation with change detection."""

    def __init__(self):
        """Initialize brief scheduler."""
        self.config = get_scheduler_config()
        self.db_manager = get_db_manager()

        # Status tracking service for diagnostics
        from src.services.system_status_service import get_system_status_service
        self.status_service = get_system_status_service()

        # Track last brief generation
        self._last_brief_time: Optional[datetime] = None
        self._last_data_hash: Optional[str] = None

        # Scheduler control
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        logger.info("brief_scheduler_initialized", config={
            "generation_interval": self.config.BRIEF_GENERATION_INTERVAL,
            "only_on_new_data": self.config.BRIEF_ONLY_ON_NEW_DATA,
            "min_items": self.config.BRIEF_MIN_ITEMS,
        })

    def start(self):
        """Start the background scheduler in a separate thread."""
        if self._running:
            logger.warning("brief_scheduler_already_running")
            return

        if not self.config.ENABLE_BACKGROUND_SCHEDULER or not self.config.ENABLE_AUTO_BRIEF:
            logger.info("brief_scheduler_disabled_by_config")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler_loop,
            daemon=True,
            name="BriefScheduler"
        )
        self._scheduler_thread.start()
        logger.info("brief_scheduler_started")

    def stop(self):
        """Stop the background scheduler."""
        if not self._running:
            return

        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        logger.info("brief_scheduler_stopped")

    def _run_scheduler_loop(self):
        """Main scheduler loop running in background thread."""
        logger.info("brief_scheduler_loop_started")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Main scheduler loop
            while self._running:
                try:
                    # Check if we should generate a brief
                    loop.run_until_complete(self._check_and_generate_brief())

                    # Sleep for configured interval (in minutes)
                    interval_seconds = self.config.BRIEF_GENERATION_INTERVAL * 60

                    for _ in range(interval_seconds):
                        if not self._running:
                            break
                        asyncio.sleep(1)

                except Exception as e:
                    logger.error("brief_scheduler_loop_error", error=str(e), exc_info=True)
                    asyncio.sleep(60)  # Wait before retrying

        finally:
            loop.close()
            logger.info("brief_scheduler_loop_stopped")

    async def _check_and_generate_brief(self):
        """Check if brief should be generated and generate if needed."""
        try:
            # Check if enough time has passed
            if self._last_brief_time is not None:
                elapsed_minutes = (datetime.now() - self._last_brief_time).total_seconds() / 60
                if elapsed_minutes < self.config.BRIEF_GENERATION_INTERVAL:
                    logger.debug("brief_scheduler_interval_not_elapsed",
                               elapsed_minutes=elapsed_minutes,
                               interval=self.config.BRIEF_GENERATION_INTERVAL)
                    return

            # Check if data has changed (if configured to only generate on new data)
            if self.config.BRIEF_ONLY_ON_NEW_DATA:
                current_data_hash = await self._get_data_hash()

                if current_data_hash == self._last_data_hash and self._last_data_hash is not None:
                    logger.debug("brief_scheduler_no_data_changes")
                    return

                logger.info("brief_scheduler_data_changed",
                          old_hash=self._last_data_hash,
                          new_hash=current_data_hash)

            # Check if minimum data requirements are met
            data_counts = await self._get_data_counts()
            total_items = sum(data_counts.values())

            if total_items < self.config.BRIEF_MIN_ITEMS:
                logger.info("brief_scheduler_insufficient_data",
                          total_items=total_items,
                          min_required=self.config.BRIEF_MIN_ITEMS,
                          counts=data_counts)
                return

            # Record brief generation start
            self.status_service.record_refresh_start('brief', self.config.BRIEF_GENERATION_INTERVAL)

            # Generate the brief
            logger.info("brief_scheduler_generating_brief", data_counts=data_counts)

            brief = await self._generate_enhanced_brief(data_counts)

            if brief:
                # Store brief in database
                await self._store_brief(brief)

                # Update tracking
                self._last_brief_time = datetime.now()
                self._last_data_hash = await self._get_data_hash()

                # Calculate brief metrics
                sections_generated = len([v for v in brief.values() if v])

                # Record successful generation
                self.status_service.record_refresh_success(
                    'brief',
                    items_fetched=sections_generated,
                    next_refresh_minutes=self.config.BRIEF_GENERATION_INTERVAL
                )

                logger.info("brief_scheduler_generation_success",
                          brief_length=len(brief.get('summary', '')),
                          sections=list(brief.keys()))
            else:
                # Record failure
                self.status_service.record_refresh_error('brief', "Brief generation returned None")
                logger.warning("brief_scheduler_generation_failed")

        except Exception as e:
            # Record error
            self.status_service.record_refresh_error('brief', str(e))
            logger.error("brief_scheduler_check_error", error=str(e), exc_info=True)

    async def _get_data_hash(self) -> str:
        """Get hash of current data to detect changes.

        Returns:
            Hash string representing current data state
        """
        import hashlib

        try:
            # Get counts from each source
            news_count = len(await self.db_manager.get_all_articles(limit=1000))
            weather_data = await self.db_manager.get_cache('weather_data')
            weather_count = 1 if weather_data else 0

            # For email/calendar, we'd check their tables
            # For now, using simplified approach
            data_signature = f"{news_count}:{weather_count}:{datetime.now().strftime('%Y-%m-%d-%H')}"

            return hashlib.md5(data_signature.encode()).hexdigest()

        except Exception as e:
            logger.error("brief_scheduler_hash_error", error=str(e))
            return ""

    async def _get_data_counts(self) -> Dict[str, int]:
        """Get count of items from each data source.

        Returns:
            Dict mapping source name to item count
        """
        try:
            news_articles = await self.db_manager.get_all_articles(limit=1000)
            weather_data = await self.db_manager.get_cache('weather_data')

            return {
                "news": len(news_articles),
                "weather": 1 if weather_data else 0,
                "email": 0,  # TODO: Implement email count
                "calendar": 0,  # TODO: Implement calendar count
            }

        except Exception as e:
            logger.error("brief_scheduler_counts_error", error=str(e))
            return {}

    async def _generate_enhanced_brief(self, data_counts: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Generate enhanced AI brief with all sections.

        Args:
            data_counts: Count of items per source

        Returns:
            Dict with brief sections or None if generation failed
        """
        try:
            # Import AI services (lazy import to avoid startup delays)
            from src.services.ollama_service import get_ollama_service
            from src.agents.news_agent_ai import get_news_agent_ai

            ollama_service = get_ollama_service()
            news_agent = get_news_agent_ai()

            # Check if Ollama is available
            if not await ollama_service.check_availability():
                logger.warning("brief_scheduler_ollama_unavailable")
                return None

            # Get data for brief
            news_articles = await self.db_manager.get_all_articles(limit=50)
            weather_data = await self.db_manager.get_cache('weather_data')
            emails_data = await self.db_manager.get_cache('emails_data')
            calendar_data = await self.db_manager.get_cache('calendar_data')

            if not news_articles and not weather_data and not emails_data and not calendar_data:
                logger.info("brief_scheduler_no_data_for_brief")
                return None

            # Separate market news from tech news
            market_articles = []
            tech_articles = []
            for article in news_articles:
                tags = getattr(article, 'tags', [])
                if 'market' in tags:
                    market_articles.append(article)
                else:
                    tech_articles.append(article)

            # Generate enhanced summary with all sections
            prompt = self._build_enhanced_prompt(
                tech_articles=tech_articles,
                market_articles=market_articles,
                weather_data=weather_data,
                emails_data=emails_data,
                calendar_data=calendar_data
            )

            logger.info("brief_scheduler_calling_llm",
                      article_count=len(news_articles),
                      has_weather=bool(weather_data))

            # Use model from config or OllamaService default (reads from OLLAMA_MODEL env var)
            model = self.config.OLLAMA_MODEL if hasattr(self.config, 'OLLAMA_MODEL') else ollama_service.config.model

            response = await ollama_service.generate(
                prompt=prompt,
                model=model
            )

            # Extract content from OllamaResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response into sections
            brief = self._parse_brief_response(response_text)

            return brief

        except Exception as e:
            logger.error("brief_scheduler_generation_error", error=str(e), exc_info=True)
            return None

    def _build_enhanced_prompt(
        self,
        tech_articles,
        market_articles,
        weather_data,
        emails_data=None,
        calendar_data=None
    ) -> str:
        """Build prompt for enhanced brief generation.

        Args:
            tech_articles: List of tech news articles
            market_articles: List of market/financial news articles
            weather_data: Weather data dict
            emails_data: List of email messages (optional)
            calendar_data: List of calendar events (optional)

        Returns:
            Prompt string
        """
        prompt = "Generate a comprehensive daily brief with the following sections:\n\n"

        # Add important emails (PRIORITY #1 for TLDR and Summary)
        if emails_data:
            # Filter to important emails only
            important_emails = [e for e in emails_data if e.get('importance_score', 0) > 0.6][:5]
            if important_emails:
                prompt += "**IMPORTANT EMAILS (PRIORITY DATA - use for TLDR bullet #1):**\n"
                for i, email in enumerate(important_emails, 1):
                    subject = email.get('subject', 'No subject')
                    sender = email.get('sender', 'Unknown')
                    ai_summary = email.get('ai_summary', email.get('snippet', 'No summary'))
                    action_items = email.get('action_items', [])
                    prompt += f"{i}. From {sender}: {subject}\n"
                    if ai_summary:
                        prompt += f"   Summary: {ai_summary[:200]}\n"
                    if action_items:
                        prompt += f"   Actions: {', '.join(action_items[:2])}\n"
                prompt += "\n"

        # Add calendar events (PRIORITY #2 for TLDR)
        if calendar_data:
            events = calendar_data.get('events', [])[:5] if isinstance(calendar_data, dict) else calendar_data[:5]
            if events:
                prompt += "**TODAY'S CALENDAR (PRIORITY DATA - use for TLDR bullet #2):**\n"
                for i, event in enumerate(events, 1):
                    if isinstance(event, dict):
                        summary = event.get('summary', 'No title')
                        start_time = event.get('start', {}).get('dateTime', 'Time TBD')
                        attendees = event.get('attendees', [])
                        prompt += f"{i}. {start_time}: {summary}\n"
                        if attendees:
                            prompt += f"   Attendees: {len(attendees)} people\n"
                prompt += "\n"

        # Add market news (PRIORITY #3 for TLDR bullet #3)
        if market_articles:
            prompt += "**MARKET/FINANCIAL NEWS (for TLDR bullet #3):**\n"
            for i, article in enumerate(market_articles[:10], 1):
                title = getattr(article, 'title', 'No title')
                prompt += f"{i}. {title}\n"
            prompt += "\n"

        # Add tech news context
        if tech_articles:
            prompt += "**TECH NEWS:**\n"
            for i, article in enumerate(tech_articles[:15], 1):
                title = getattr(article, 'title', 'No title')
                prompt += f"{i}. {title}\n"
            prompt += "\n"

        # Add weather context
        if weather_data:
            temp = weather_data.get('main', {}).get('temp', 'N/A')
            desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
            prompt += f"**WEATHER:** {temp}°F, {desc}\n\n"

        prompt += """
Please analyze the above information and provide:

1. **SUMMARY** (EXACTLY 5-6 COMPLETE DETAILED SENTENCES - MINIMUM 500 CHARACTERS)
   CRITICAL: Write COMPREHENSIVE, DETAILED sentences with maximum context. NOT brief headlines!

   PRIORITIZATION:
   - FIRST: Important emails/calendar events (YOUR day's critical items)
   - SECOND: Significant news with business/tech relevance
   - Include specific names, numbers, deadlines, WHY it matters, and IMPACT

   ❌ REJECTED (too short/vague, only 180 chars):
   "OpenAI's IPO clears path as it promises to stay in California, despite federal scrutiny over AI safety and regulation. AOL will be sold for $1.5 billion to Bending Spoons."

   ✅ REQUIRED (500+ chars with MAXIMUM detail and context):
   "OpenAI successfully completes its initial public offering valued at $80 billion after committing to maintain headquarters in California, addressing federal concerns about AI safety oversight, regulatory compliance frameworks, and responsible development practices that emerged during SEC review process. Federal Reserve announces fifth consecutive interest rate increase of 0.5% bringing rates to 5.25%, the highest level since the 2008 financial crisis, as central bank combats persistent inflation holding at 6.4% year-over-year despite previous monetary tightening efforts. Tesla achieves Level 4 autonomous driving certification in California, Arizona, and Texas after completing 10 million miles of testing without safety driver intervention, marking significant milestone in commercial self-driving deployment and potentially unlocking $50 billion robotaxi market. Microsoft acquires AI startup Anthropic for $18 billion to strengthen competition against Google's AI dominance, combining Claude's constitutional AI safety features with Azure's cloud infrastructure to capture enterprise AI market."

   MANDATORY REQUIREMENTS:
   - MINIMUM 500 characters total (count them! Add more detail if under 500!)
   - Each sentence: 20-30 words with MAXIMUM context (WHO, WHAT, WHEN, WHY, HOW MUCH, IMPACT)
   - Include: specific numbers, percentages, dollar amounts, deadlines, timeframes
   - Add CONTEXT: explain WHY it matters, WHAT changed, WHO is affected, WHAT happens next
   - NO generic words: "features", "continues", "faces", "highlights", "various", "several"
   - If under 500 chars: ADD MORE detail about implications, background, or related impacts

2. **KEY LEARNINGS** (EXACTLY 5-7 specific bullet points - MANDATORY)
   - Each bullet must be a complete, specific insight
   - Include concrete facts, numbers, or names
   - Focus on actionable information
   - NO generic phrases like "important information" or "various updates"

   EXAMPLE:
   - OpenAI releases GPT-5 with 10x performance improvement in reasoning tasks
   - Federal Reserve signals 0.5% interest rate increase to combat inflation
   - Tesla achieves full self-driving milestone in controlled environments

3. **KEY INSIGHTS** (EXACTLY 5-7 specific analysis points - MANDATORY)
   - You MUST write AT LEAST 5 insights
   - Identify patterns and trends across multiple articles
   - Make connections between seemingly unrelated topics
   - Provide strategic implications for decision-making
   - Each insight must be substantive and thought-provoking

   EXAMPLE:
   - Tech layoffs correlate with shift toward AI automation across industries, affecting 150,000 workers
   - Energy prices driving broader inflationary pressures in multiple sectors, with crude oil at $120/barrel
   - Geopolitical tensions affecting supply chain resilience strategies across automotive and electronics
   - Central banks coordinating policy response despite diverging economic indicators
   - Consumer spending patterns shift dramatically as recession fears mount

4. **TLDR** (EXACTLY 3 BULLETS - Prioritized by actionability - MANDATORY)
   Format as THREE separate bullet points in this EXACT order:

   **Bullet 1 - TOP EMAIL/ACTION** (MOST IMPORTANT):
   • [Most critical email requiring action OR most urgent todo/deadline]
   - Must include: WHAT needs to be done, by WHEN, WHY it's critical
   - Example: "Critical production database issue requires review of error logs and replication status by EOD to prevent service disruption"
   - If no critical emails: "No urgent email actions today"

   **Bullet 2 - TOP CALENDAR EVENT** (SECOND PRIORITY):
   • [Most important meeting/event today OR next critical event]
   - Must include: WHAT meeting, WHEN, WHO is involved, WHAT prep is needed
   - Example: "Board meeting at 9am requires Q4 financial presentation with revenue projections and growth metrics"
   - If no important events: "No critical calendar events today"

   **Bullet 3 - TOP MARKET/BUSINESS NEWS** (THIRD PRIORITY):
   • [Most impactful market/business news with potential relevance to work]
   - Must include: WHAT happened, IMPACT/numbers, WHY it matters to business
   - Example: "Federal Reserve cuts interest rates by 0.5% to 4.75%, lowest level in three years, potentially accelerating tech sector investments"
   - Prefer: Fed announcements, major acquisitions, market indices, sector trends
   - If no market news: Use most relevant tech/business news

   ✅ PERFECT TLDR EXAMPLE:
   • Critical: Client escalation from Acme Corp requires response within 24 hours regarding service outages affecting 50K users
   • Today 2pm: Technical interview with senior backend engineer candidate - review portfolio and prepare architecture questions
   • Market: S&P 500 drops 3.2% on inflation concerns as Fed signals potential rate hikes, impacting tech valuations

   ✅ EXTENSIBLE FORMAT (for future data sources - Slack, Jira, etc.):
   • Email: [top email action]
   • Calendar: [top meeting]
   • Market: [top business news]
   <!-- Future: Slack: [top mention], Jira: [blocked ticket], etc. -->

   ❌ BAD EXAMPLES:
   • "Today's headlines bring a mix of business and technological updates" (USELESS!)
   • "Check your email and calendar" (NO SPECIFICS!)
   • "Several important items need attention" (TOO VAGUE!)

CRITICAL REQUIREMENTS:
- Every bullet point must be a complete, specific statement
- NO placeholder text like "NEWS ARTICLES 1" or numbered sections
- NO meta-commentary about the brief itself
- Use real facts from the provided articles
- If weather data is available, weave it into the context

Format your response with clear section headers (SUMMARY, KEY LEARNINGS, KEY INSIGHTS, TLDR).
"""

        return prompt

    def _is_generic_tldr(self, tldr: str) -> bool:
        """Check if TLDR is too generic or placeholder-like.

        Args:
            tldr: TLDR text to check

        Returns:
            True if TLDR is generic, False if it's specific
        """
        if not tldr:
            return True

        # Generic phrases that indicate placeholder content
        generic_phrases = [
            "range of topics",
            "various topics",
            "various areas",
            "multiple areas",
            "covers",
            "including",
            "discusses",
            "explores",
            "examines",
            "today's brief",
            "this brief",
            "the brief",
            "news articles",
            "multiple topics",
            "several topics",
            "information about",
        ]

        tldr_lower = tldr.strip().lower()

        # Check if TLDR contains generic phrases
        for phrase in generic_phrases:
            if phrase in tldr_lower:
                return True

        # Check if TLDR is too short (less than 40 chars is too vague for a TLDR)
        if len(tldr.strip()) < 40:
            return True

        # Check if TLDR is just listing categories without specifics
        if tldr_lower.count(",") > 2 and " and " in tldr_lower:
            # Likely just listing categories like "tech, business, and health"
            return True

        return False

    def _generate_fallback_tldr(self, summary: str) -> str:
        """Generate a fallback TLDR from the summary.

        Args:
            summary: Brief summary text

        Returns:
            Generated TLDR (first sentence of summary, NOT truncated)
        """
        if not summary:
            return "Daily brief generated"

        # Get first sentence
        import re
        sentences = re.split(r'[.!?]+', summary)
        first_sentence = sentences[0].strip() if sentences else summary

        # Return full first sentence (user wants complete TLDR, not truncated)
        return first_sentence + "."  # Add period back

    def _parse_brief_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured brief sections.

        Args:
            response: Raw LLM response text

        Returns:
            Dict with parsed sections
        """
        sections = {
            "summary": "",
            "key_learnings": [],
            "key_insights": [],
            "tldr": "",
            "generated_at": datetime.now().isoformat()
        }

        try:
            # Simple parsing logic (can be improved with better NLP)
            current_section = None
            lines = response.split('\n')

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # Detect section headers
                if 'SUMMARY' in line.upper() and len(line) < 30:
                    current_section = 'summary'
                elif 'KEY LEARNING' in line.upper() and len(line) < 30:
                    current_section = 'key_learnings'
                elif 'KEY INSIGHT' in line.upper() and len(line) < 30:
                    current_section = 'key_insights'
                elif 'TLDR' in line.upper() and len(line) < 30:
                    current_section = 'tldr'
                elif current_section:
                    # Add content to current section
                    if current_section in ['key_learnings', 'key_insights']:
                        # Extract bullet points
                        if line.startswith(('-', '*', '•', '·')) or line[0].isdigit():
                            clean_line = line.lstrip('-*•·0123456789. ')
                            if clean_line:
                                sections[current_section].append(clean_line)
                    else:
                        # Add to text sections
                        if sections[current_section]:
                            sections[current_section] += " " + line
                        else:
                            sections[current_section] = line

            # Post-processing: Fix TLDR formatting (add newlines between bullets)
            if sections['tldr']:
                # Split on bullet markers and rejoin with newlines
                tldr = sections['tldr']

                # Remove markdown headers (###) and clean up
                import re
                tldr = re.sub(r'###\s*\*?\*?', '', tldr)  # Remove ### markers

                # Replace bullet markers with newlines
                # Match: • followed by optional bold markdown and text
                tldr = re.sub(r'([.!?%])\s*•\s*', r'\1\n• ', tldr)

                # Also split on "TLDR X -" patterns
                tldr = re.sub(r'([.!?%])\s*(TLDR \d+ - )', r'\1\n• ', tldr)

                # Clean up any duplicate newlines
                tldr = re.sub(r'\n\n+', '\n', tldr)

                sections['tldr'] = tldr.strip()

            # Post-processing: Fix generic TLDR
            if self._is_generic_tldr(sections['tldr']):
                logger.warning("brief_scheduler_generic_tldr_detected",
                             original_tldr=sections['tldr'])
                sections['tldr'] = self._generate_fallback_tldr(sections['summary'])
                logger.info("brief_scheduler_tldr_replaced",
                          new_tldr=sections['tldr'])

        except Exception as e:
            logger.error("brief_scheduler_parse_error", error=str(e))

        return sections

    async def _store_brief(self, brief: Dict[str, Any]):
        """Store generated brief in database.

        Args:
            brief: Brief sections dict
        """
        try:
            logger.info("brief_scheduler_brief_generated",
                      summary_preview=brief.get('summary', '')[:100],
                      learnings_count=len(brief.get('key_learnings', [])),
                      insights_count=len(brief.get('key_insights', [])),
                      tldr=brief.get('tldr', ''))

            # Get data counts for the brief metadata
            news_articles = await self.db_manager.get_all_articles(limit=1000)
            weather_data = await self.db_manager.get_cache('weather_data')
            emails_data = await self.db_manager.get_cache('emails_data')
            calendar_data = await self.db_manager.get_cache('calendar_data')

            # Prepare brief data for storage
            brief_data = {
                'summary': brief.get('summary', ''),
                'key_points': brief.get('key_learnings', []) + brief.get('key_insights', []),
                'action_items': [],  # TODO: Extract from brief content
                'tldr': brief.get('tldr', ''),
                'generated_at': brief.get('generated_at', datetime.now().isoformat()),
                'emails_count': len(emails_data) if emails_data else 0,
                'calendar_events_count': len(calendar_data) if calendar_data else 0,
                'news_items_count': len(news_articles),
                'weather_info': weather_data
            }

            # Store brief in cache (like weather and emails)
            await self.db_manager.set_cache('daily_brief_data', brief_data)

            # Save generation timestamp to settings
            await self.db_manager.set_setting('last_brief_generation',
                                             datetime.now().isoformat())

            logger.info("brief_scheduler_brief_saved_to_cache",
                       news_count=brief_data['news_items_count'],
                       emails_count=brief_data['emails_count'],
                       calendar_count=brief_data['calendar_events_count'])

        except Exception as e:
            logger.error("brief_scheduler_store_error", error=str(e))

    def get_status(self) -> Dict:
        """Get current scheduler status.

        Returns:
            Dict with scheduler status information
        """
        return {
            "running": self._running,
            "enabled": self.config.ENABLE_AUTO_BRIEF,
            "last_brief_time": self._last_brief_time.isoformat() if self._last_brief_time else None,
            "generation_interval_minutes": self.config.BRIEF_GENERATION_INTERVAL,
            "only_on_new_data": self.config.BRIEF_ONLY_ON_NEW_DATA,
        }


# Global singleton instance
_brief_scheduler: Optional[BriefScheduler] = None


def get_brief_scheduler() -> BriefScheduler:
    """Get the global brief scheduler instance.

    Returns:
        BriefScheduler instance
    """
    global _brief_scheduler

    if _brief_scheduler is None:
        _brief_scheduler = BriefScheduler()

    return _brief_scheduler


def start_brief_scheduler():
    """Start the global brief scheduler."""
    # Check if scheduler is enabled before creating instance
    # This prevents heavy service initialization when scheduler is disabled
    config = get_scheduler_config()
    if not config.ENABLE_BACKGROUND_SCHEDULER:
        logger.info("brief_scheduler_disabled_by_config")
        return

    scheduler = get_brief_scheduler()
    scheduler.start()


def stop_brief_scheduler():
    """Stop the global brief scheduler."""
    global _brief_scheduler

    if _brief_scheduler is not None:
        _brief_scheduler.stop()

#!/usr/bin/env python3
"""CLI tool to pre-load all data before starting the UI.

This script fetches and caches all data (news, weather, email, calendar) and generates
daily briefs so the UI can start instantly with everything already available.

Usage:
    ./venv/bin/python scripts/preload_all_data.py
    ./venv/bin/python scripts/preload_all_data.py --max-articles 50
"""

import asyncio
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.core.logging import get_logger
from src.services.background_refresh_service import get_background_refresh_service
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


async def preload_all_data(max_articles: int = 30, force_regenerate: bool = False):
    """Preload all data into the database cache.

    Args:
        max_articles: Maximum number of news articles to fetch
        force_regenerate: If True, regenerate all AI analysis even if cached
    """
    print("=" * 70)
    if force_regenerate:
        print("PRE-LOADING ALL DATA (FORCE REGENERATE AI ANALYSIS)")
    else:
        print("PRE-LOADING ALL DATA")
    print("=" * 70)
    print()

    # Get services
    refresh_service = get_background_refresh_service()
    db_manager = get_db_manager()
    await db_manager.initialize()

    # Track what was loaded
    results = {}
    start_time = datetime.now()

    # 1. Load News
    print("📰 [1/5] Fetching news articles...")
    try:
        success = await refresh_service._refresh_news()
        results['news'] = success
        if success:
            articles = await db_manager.get_all_articles(limit=100)
            print(f"   ✅ Loaded {len(articles)} news articles")

            # Index articles in RAG for semantic search
            print(f"   🔍 Indexing articles in RAG database...")
            try:
                from src.services.rag_service import get_rag_service
                rag_service = get_rag_service()
                indexed_ids = await rag_service.add_articles_batch(articles)
                print(f"   ✅ Indexed {len(indexed_ids)} articles in RAG")
            except Exception as rag_error:
                print(f"   ⚠️  RAG indexing failed: {str(rag_error)[:50]}")
                logger.warning("rag_indexing_failed", error=str(rag_error))
        else:
            print(f"   ❌ Failed to load news")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['news'] = False

    # 2. Load Weather
    print()
    print("🌤️  [2/5] Fetching weather data...")
    try:
        success = await refresh_service._refresh_weather()
        results['weather'] = success
        if success:
            weather = await db_manager.get_cache('weather_data')
            location = weather.get('location', 'Unknown') if weather else 'Unknown'
            temp = weather.get('temperature', 0) if weather else 0
            print(f"   ✅ Loaded weather for {location} ({temp}°C)")
        else:
            print(f"   ❌ Failed to load weather")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['weather'] = False

    # 3. Load Email
    print()
    print("📧 [3/5] Fetching emails...")
    try:
        success = await refresh_service._refresh_email()
        results['email'] = success
        if success:
            # Email service doesn't store count, just log success
            print(f"   ✅ Loaded recent emails")
        else:
            print(f"   ⚠️  Email refresh completed (may require credentials)")
    except Exception as e:
        logger.warning("email_refresh_error", error=str(e))
        print(f"   ⚠️  Email refresh skipped: {str(e)[:50]}")
        results['email'] = False

    # 4. Load Calendar
    print()
    print("📅 [4/5] Fetching calendar events...")
    try:
        success = await refresh_service._refresh_calendar()
        results['calendar'] = success
        if success:
            print(f"   ✅ Loaded calendar events")
        else:
            print(f"   ⚠️  Calendar refresh completed (may require credentials)")
    except Exception as e:
        logger.warning("calendar_refresh_error", error=str(e))
        print(f"   ⚠️  Calendar refresh skipped: {str(e)[:50]}")
        results['calendar'] = False

    # 4.5. Force Regenerate AI Analysis (if flag set)
    if force_regenerate:
        print()
        print("🔄 [4.5/5] Force regenerating AI analysis...")
        try:
            from src.services.article_analyzer import get_article_analyzer

            # Regenerate news article analysis
            articles = await db_manager.get_all_articles(limit=200)
            if articles:
                print(f"   Regenerating analysis for {len(articles)} articles...")
                analyzer = get_article_analyzer()

                regenerated = 0
                for article in articles:
                    try:
                        # Force regenerate by setting use_cache=False
                        analysis = await analyzer.analyze_article(
                            title=article.title,
                            content=article.description or article.content,
                            url=str(article.url),
                            use_cache=False  # Force regenerate
                        )

                        # Update article with new analysis
                        article.ai_summary = analysis.get('analysis', '')
                        article.tldr = analysis.get('tldr', '')
                        article.key_learnings = analysis.get('key_learnings', [])

                        # Save updated article
                        await db_manager.save_article(article)
                        regenerated += 1

                        if regenerated % 10 == 0:
                            print(f"   ... {regenerated}/{len(articles)} articles regenerated")

                    except Exception as e:
                        logger.warning("article_regeneration_failed",
                                     title=article.title[:50],
                                     error=str(e))

                print(f"   ✅ Regenerated {regenerated}/{len(articles)} article analyses")

            # Regenerate email summaries
            emails_data = await db_manager.get_cache('emails_data')
            if emails_data and isinstance(emails_data, list):
                important_emails = [e for e in emails_data if e.get('importance_score', 0) > 0.7]
                if important_emails:
                    print(f"   Regenerating summaries for {len(important_emails)} important emails...")

                    from src.services.ollama_service import get_ollama_service
                    ollama = get_ollama_service()

                    regenerated_emails = 0
                    for email in important_emails:
                        try:
                            prompt = f"""Analyze this important email and extract key information:

Subject: {email.get('subject')}
From: {email.get('sender')}
Body: {email.get('body', '')[:800]}

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
                            )

                            if hasattr(response, 'content'):
                                response_text = response.content.strip()
                            elif isinstance(response, dict) and 'content' in response:
                                response_text = response['content'].strip()
                            else:
                                response_text = str(response)

                            # Extract Summary section
                            if 'Summary:' in response_text:
                                summary_start = response_text.find('Summary:') + len('Summary:')
                                summary_text = response_text[summary_start:].split('\n\n')[0].strip()
                                email['ai_summary'] = summary_text
                                regenerated_emails += 1

                        except Exception as e:
                            logger.warning("email_regeneration_failed",
                                         subject=email.get('subject', '')[:50],
                                         error=str(e))

                    # Save updated emails
                    await db_manager.set_cache('emails_data', emails_data)
                    print(f"   ✅ Regenerated {regenerated_emails}/{len(important_emails)} email summaries")

            results['force_regenerate'] = True
        except Exception as e:
            logger.error("force_regenerate_failed", error=str(e))
            print(f"   ❌ Error during regeneration: {str(e)[:100]}")
            results['force_regenerate'] = False

    # 5. Generate Daily Brief (force regenerate if --force flag set)
    print()
    print("📋 [5/5] Generating daily brief...")
    try:
        from src.services.brief_scheduler import get_brief_scheduler

        brief_scheduler = get_brief_scheduler()

        if force_regenerate:
            # Force regenerate brief by clearing cache first
            print(f"   Force regenerating daily brief...")
            await db_manager.set_cache('daily_brief_data', None)  # Clear old brief

        # Call the check method which will generate the brief if conditions are met
        await brief_scheduler._check_and_generate_brief()
        results['brief'] = True
        print(f"   ✅ Brief {'regenerated' if force_regenerate else 'check completed'}")
    except Exception as e:
        logger.warning("brief_generation_error", error=str(e))
        print(f"   ⚠️  Brief generation skipped: {str(e)[:50]}")
        results['brief'] = False

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"✅ Successfully loaded: {success_count}/{total_count}")
    print(f"⏱️  Time taken: {elapsed:.1f}s")
    print()

    for source, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {source.capitalize()}")

    print()
    print("=" * 70)

    if success_count == total_count:
        print("🎉 All data loaded successfully!")
        print("You can now start the UI with: streamlit run streamlit_app.py")
        return 0
    else:
        print("⚠️  Some data failed to load. Check errors above.")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-load all data (news, weather, email, calendar, briefs) into database cache"
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        default=30,
        help='Maximum number of news articles to fetch (default: 30)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regenerate all AI analysis even if cached'
    )

    args = parser.parse_args()

    try:
        exit_code = asyncio.run(preload_all_data(
            max_articles=args.max_articles,
            force_regenerate=args.force
        ))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        logger.error("preload_fatal_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

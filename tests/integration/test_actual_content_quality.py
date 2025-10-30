"""Integration tests to verify ACTUAL content quality in database."""

# Load .env BEFORE any imports
from dotenv import load_dotenv
load_dotenv()

import asyncio
from src.database.sqlite_manager import get_db_manager


async def test_brief_summary_quality():
    """Test that brief summary has SPECIFIC facts, not generic statements."""
    db = get_db_manager()
    brief = await db.get_cache('daily_brief_data')

    summary = brief.get('summary', '')
    print(f"\n=== BRIEF SUMMARY QUALITY TEST ===")
    print(f"Summary: {summary}")

    # Split into sentences
    sentences = [s.strip() for s in summary.split('. ') if s.strip()]
    print(f"\nBullet points that will be shown:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")

    # Check for generic phrases (BAD)
    generic_phrases = [
        "features a mix",
        "continues to make",
        "highlights the",
        "day's news",
        "various topics",
        "several areas"
    ]

    generic_found = []
    for sent in sentences:
        for phrase in generic_phrases:
            if phrase.lower() in sent.lower():
                generic_found.append((sent, phrase))

    if generic_found:
        print(f"\n❌ GENERIC PHRASES FOUND:")
        for sent, phrase in generic_found:
            print(f"  - '{phrase}' in: {sent[:80]}...")
        print("\nUser wants SPECIFIC facts like:")
        print("  ✓ 'OpenAI releases GPT-5 with 10x performance improvement in reasoning tasks'")
        print("  ✓ 'Federal Reserve signals 0.5% interest rate increase to combat inflation'")
        print("  ✗ 'The day's news features a mix of technological advancements'")
    else:
        print("\n✓ No generic phrases found")

    return sentences


async def test_news_analysis_comprehensive():
    """Test that news articles have comprehensive TLDR + analysis."""
    db = get_db_manager()
    articles = await db.get_all_articles(limit=5)

    print(f"\n=== NEWS ARTICLE ANALYSIS TEST ===")

    for i, article in enumerate(articles[:3]):
        print(f"\nArticle {i+1}: {article.title[:60]}")
        print(f"  ai_summary: {article.ai_summary[:150] if article.ai_summary else 'None'}...")
        print(f"  tldr: {article.tldr[:150] if article.tldr else 'None'}...")

        # Check if analysis is comprehensive
        if article.ai_summary:
            analysis_length = len(article.ai_summary)
            print(f"  Analysis length: {analysis_length} chars")

            if analysis_length < 300:
                print(f"  ❌ Analysis too short! Should be 300+ chars for comprehensive analysis")
                print(f"     Current: {analysis_length} chars, Need: {300 - analysis_length} more")
            else:
                print(f"  ✓ Analysis length OK (300+ chars)")

            # Check for generic analysis
            if any(phrase in article.ai_summary.lower() for phrase in ['appears to be', 'seems to', 'the article']):
                print(f"  ❌ Analysis sounds generic - starts with 'The article appears to...'")
                print("  Should be direct analysis of actual content")


async def test_email_has_analysis():
    """Test that emails have ai_summary and action items extracted."""
    db = get_db_manager()
    emails_data = await db.get_cache('emails_data')

    print(f"\n=== EMAIL ANALYSIS TEST ===")
    print(f"Total emails: {len(emails_data)}")

    important_emails = [e for e in emails_data if e.get('importance_score', 0) > 0.7][:5]
    print(f"Important emails (score > 0.7): {len(important_emails)}")

    for i, email in enumerate(important_emails[:3]):
        print(f"\nEmail {i+1}: {email.get('subject', 'N/A')[:60]}")
        print(f"  Importance: {email.get('importance_score', 0):.2f}")
        print(f"  Has snippet: {bool(email.get('snippet'))}")
        print(f"  Has ai_summary: {bool(email.get('ai_summary'))}")
        print(f"  Has action items: {bool(email.get('action_items'))}")

        if email.get('ai_summary'):
            print(f"  AI Summary: {email.get('ai_summary')[:150]}...")
        else:
            print(f"  ❌ Missing AI summary for important email!")

        if email.get('action_items'):
            print(f"  Action items ({len(email.get('action_items'))}):")
            for item in email.get('action_items')[:2]:
                print(f"    - {item}")
        else:
            print(f"  ⚠️  No action items extracted")


async def test_key_learnings_extracted():
    """Test that key learnings are extracted from news."""
    db = get_db_manager()
    articles = await db.get_all_articles(limit=5)

    print(f"\n=== KEY LEARNINGS EXTRACTION TEST ===")

    total_with_learnings = 0
    for article in articles:
        if article.key_learnings and len(article.key_learnings) > 0:
            total_with_learnings += 1

    print(f"Articles with key learnings: {total_with_learnings}/{len(articles)}")

    for i, article in enumerate(articles[:3]):
        print(f"\nArticle {i+1}: {article.title[:60]}")
        if article.key_learnings:
            print(f"  Key learnings ({len(article.key_learnings)}):")
            for learning in article.key_learnings[:3]:
                print(f"    - {learning}")
        else:
            print(f"  ❌ No key learnings extracted!")


if __name__ == "__main__":
    print("=" * 80)
    print("CONTENT QUALITY INTEGRATION TESTS")
    print("=" * 80)

    asyncio.run(test_brief_summary_quality())
    asyncio.run(test_news_analysis_comprehensive())
    asyncio.run(test_email_has_analysis())
    asyncio.run(test_key_learnings_extracted())

    print("\n" + "=" * 80)
    print("ISSUES TO FIX:")
    print("=" * 80)
    print("1. Brief summary has generic phrases instead of specific facts")
    print("2. News analysis too short (< 200 chars)")
    print("3. Emails missing AI analysis/todos")
    print("4. Key learnings not being extracted properly")

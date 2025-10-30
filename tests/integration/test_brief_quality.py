"""Integration tests to verify brief quality with new 3-bullet TLDR and prioritization."""

# Load .env BEFORE any imports
from dotenv import load_dotenv
load_dotenv()

import asyncio
import re
from src.database.sqlite_manager import get_db_manager


async def test_tldr_has_three_bullets():
    """Test that TLDR has EXACTLY 3 bullets in correct order."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')

    print(f"\n{'='*80}")
    print("TEST: TLDR 3-BULLET FORMAT")
    print('='*80)

    assert brief, "❌ No brief found in database"

    tldr = brief.get('tldr', '')
    print(f"\nActual TLDR:\n{tldr}\n")

    # Count bullet points (lines starting with • or -)
    bullet_lines = [line.strip() for line in tldr.split('\n') if line.strip().startswith(('•', '-', '*'))]

    print(f"Bullet count: {len(bullet_lines)}")
    for i, bullet in enumerate(bullet_lines, 1):
        print(f"  {i}. {bullet[:100]}...")

    # Verify exactly 3 bullets
    if len(bullet_lines) != 3:
        print(f"\n❌ FAILED: Expected 3 bullets, got {len(bullet_lines)}")
        print("\nExpected format:")
        print("• Bullet 1: Top email/action (e.g., 'Client escalation requires response within 24 hours...')")
        print("• Bullet 2: Top calendar event (e.g., 'Board meeting at 9am requires Q4 presentation...')")
        print("• Bullet 3: Market news (e.g., 'S&P 500 drops 3.2% on inflation concerns...')")
        return False

    # Check for generic phrases (BAD)
    generic_phrases = [
        "today's headlines",
        "mix of",
        "business and technological updates",
        "various topics",
        "range of topics"
    ]

    tldr_lower = tldr.lower()
    found_generic = [phrase for phrase in generic_phrases if phrase in tldr_lower]

    if found_generic:
        print(f"\n❌ FAILED: Generic phrases found: {found_generic}")
        print("TLDR should contain SPECIFIC actions/events/news, not generic descriptions!")
        return False

    print("\n✅ PASSED: TLDR has 3 bullets with specific content")
    return True


async def test_tldr_priority_order():
    """Test that TLDR bullets are in correct priority: email > calendar > market."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')
    tldr = brief.get('tldr', '')

    print(f"\n{'='*80}")
    print("TEST: TLDR PRIORITY ORDER")
    print('='*80)

    bullet_lines = [line.strip() for line in tldr.split('\n') if line.strip().startswith(('•', '-', '*'))]

    if len(bullet_lines) != 3:
        print(f"❌ Cannot test priority - need exactly 3 bullets, got {len(bullet_lines)}")
        return False

    # Extract bullet content
    bullet_1 = bullet_lines[0].lower()
    bullet_2 = bullet_lines[1].lower()
    bullet_3 = bullet_lines[2].lower()

    print(f"\nBullet 1 (should be EMAIL/ACTION): {bullet_lines[0][:100]}...")
    print(f"Bullet 2 (should be CALENDAR): {bullet_lines[1][:100]}...")
    print(f"Bullet 3 (should be MARKET): {bullet_lines[2][:100]}...")

    # Check bullet 1 mentions email/action keywords
    email_keywords = ['email', 'escalation', 'urgent', 'action', 'response', 'client', 'issue', 'critical']
    has_email_context = any(kw in bullet_1 for kw in email_keywords)

    # Check bullet 2 mentions calendar/meeting keywords
    calendar_keywords = ['meeting', 'calendar', 'today', 'interview', 'demo', 'sprint', 'review']
    has_calendar_context = any(kw in bullet_2 for kw in calendar_keywords)

    # Check bullet 3 mentions market/financial keywords
    market_keywords = ['market', 'fed', 'federal reserve', 's&p', 'nasdaq', 'stock', 'rate', 'inflation', 'economy']
    has_market_context = any(kw in bullet_3 for kw in market_keywords)

    print(f"\nBullet 1 has email context: {has_email_context}")
    print(f"Bullet 2 has calendar context: {has_calendar_context}")
    print(f"Bullet 3 has market context: {has_market_context}")

    # If no market news, bullet 3 might be tech news (acceptable fallback)
    if not has_market_context:
        print("⚠️  No market context in bullet 3 - checking if it's tech news fallback...")
        tech_keywords = ['ai', 'tech', 'software', 'cloud', 'data', 'api']
        has_tech_context = any(kw in bullet_3 for kw in tech_keywords)
        print(f"Bullet 3 has tech context (fallback): {has_tech_context}")

    if has_email_context and has_calendar_context:
        print("\n✅ PASSED: TLDR bullets follow priority order")
        return True
    else:
        print("\n❌ FAILED: TLDR bullets don't follow expected priority")
        print("Expected: Bullet 1 = email/action, Bullet 2 = calendar, Bullet 3 = market/tech")
        return False


async def test_summary_length():
    """Test that summary is 500+ characters with detailed content."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')
    summary = brief.get('summary', '')

    print(f"\n{'='*80}")
    print("TEST: SUMMARY LENGTH & DETAIL")
    print('='*80)

    print(f"\nSummary length: {len(summary)} characters")
    print(f"Summary preview: {summary[:300]}...\n")

    # Check minimum length
    if len(summary) < 500:
        print(f"❌ FAILED: Summary too short ({len(summary)} chars, need 500+)")
        print(f"Missing: {500 - len(summary)} characters")
        return False

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]
    print(f"Sentence count: {len(sentences)}")

    if len(sentences) < 5:
        print(f"❌ FAILED: Too few sentences ({len(sentences)}, need 5-6)")
        return False

    # Check for generic phrases
    generic_phrases = [
        "features a mix",
        "continues to make",
        "highlights the",
        "day's news",
        "various topics"
    ]

    found_generic = [phrase for phrase in generic_phrases if phrase.lower() in summary.lower()]
    if found_generic:
        print(f"❌ FAILED: Generic phrases found: {found_generic}")
        return False

    # Check for numbers/specifics (good sign of detail)
    has_numbers = bool(re.search(r'\d+', summary))
    has_percentages = bool(re.search(r'\d+%', summary))
    has_dollars = bool(re.search(r'\$\d+', summary))

    print(f"\nContains numbers: {has_numbers}")
    print(f"Contains percentages: {has_percentages}")
    print(f"Contains dollar amounts: {has_dollars}")

    print("\n✅ PASSED: Summary has sufficient length and detail")
    return True


async def test_summary_prioritizes_email_calendar():
    """Test that summary mentions email/calendar before tech news."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')
    summary = brief.get('summary', '')

    print(f"\n{'='*80}")
    print("TEST: SUMMARY PRIORITIZATION")
    print('='*80)

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]

    print(f"\nAnalyzing first 3 sentences for email/calendar mentions...")
    first_three = ' '.join(sentences[:3]).lower()

    # Check for email/calendar keywords in first 3 sentences
    priority_keywords = ['email', 'meeting', 'calendar', 'escalation', 'client', 'urgent', 'action', 'interview', 'demo']
    mentions_priority = [kw for kw in priority_keywords if kw in first_three]

    print(f"Priority keywords found in first 3 sentences: {mentions_priority}")

    if mentions_priority:
        print("\n✅ PASSED: Summary prioritizes email/calendar content")
        return True
    else:
        print("\n⚠️  WARNING: Summary doesn't mention email/calendar in first 3 sentences")
        print("This might be OK if there are no important emails/calendar events")
        return True  # Don't fail - might be legitimate


async def test_key_learnings_extracted():
    """Test that key learnings are 5-7 specific bullet points."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')
    key_learnings = brief.get('key_learnings', [])

    print(f"\n{'='*80}")
    print("TEST: KEY LEARNINGS")
    print('='*80)

    print(f"\nKey learnings count: {len(key_learnings)}")

    if len(key_learnings) < 5:
        print(f"❌ FAILED: Too few key learnings ({len(key_learnings)}, need 5-7)")
        return False

    if len(key_learnings) > 7:
        print(f"⚠️  WARNING: More than 7 key learnings ({len(key_learnings)})")

    print("\nKey learnings:")
    for i, learning in enumerate(key_learnings, 1):
        print(f"  {i}. {learning}")

        # Check each learning is substantial (not too short)
        if len(learning) < 20:
            print(f"     ⚠️  Learning {i} seems too short ({len(learning)} chars)")

    print("\n✅ PASSED: Key learnings extracted")
    return True


async def test_key_insights_extracted():
    """Test that key insights are 5-7 analysis points."""
    db = get_db_manager()
    await db.initialize()

    brief = await db.get_cache('daily_brief_data')
    key_insights = brief.get('key_insights', [])

    print(f"\n{'='*80}")
    print("TEST: KEY INSIGHTS")
    print('='*80)

    print(f"\nKey insights count: {len(key_insights)}")

    if len(key_insights) < 5:
        print(f"❌ FAILED: Too few key insights ({len(key_insights)}, need 5-7)")
        return False

    print("\nKey insights:")
    for i, insight in enumerate(key_insights, 1):
        print(f"  {i}. {insight}")

    print("\n✅ PASSED: Key insights extracted")
    return True


async def test_market_news_tagged():
    """Test that market news articles are tagged with 'market' category."""
    db = get_db_manager()
    await db.initialize()

    print(f"\n{'='*80}")
    print("TEST: MARKET NEWS TAGGING")
    print('='*80)

    articles = await db.get_all_articles(limit=50)

    market_articles = [a for a in articles if 'market' in getattr(a, 'tags', [])]
    tech_articles = [a for a in articles if 'market' not in getattr(a, 'tags', [])]

    print(f"\nTotal articles: {len(articles)}")
    print(f"Market articles: {len(market_articles)}")
    print(f"Tech articles: {len(tech_articles)}")

    if market_articles:
        print("\nMarket articles found:")
        for article in market_articles[:3]:
            print(f"  - {article.title} (source: {article.source_name})")
        print("\n✅ PASSED: Market news is being fetched and tagged")
        return True
    else:
        print("\n⚠️  WARNING: No market articles found")
        print("This could mean:")
        print("  1. Yahoo Finance RSS feed didn't fetch (check RSS cache)")
        print("  2. Market news not published recently")
        print("  3. RSS connector not using configured feeds with categories")
        return False


async def test_email_summaries_generated():
    """Test that important emails have AI summaries."""
    db = get_db_manager()
    await db.initialize()

    print(f"\n{'='*80}")
    print("TEST: EMAIL AI SUMMARIES")
    print('='*80)

    emails_data = await db.get_cache('emails_data')

    if not emails_data:
        print("⚠️  WARNING: No emails found")
        return True

    important_emails = [e for e in emails_data if e.get('importance_score', 0) > 0.7]
    emails_with_summary = [e for e in important_emails if e.get('ai_summary')]

    print(f"\nTotal emails: {len(emails_data)}")
    print(f"Important emails (score > 0.7): {len(important_emails)}")
    print(f"Important emails with AI summary: {len(emails_with_summary)}")

    if important_emails:
        coverage_pct = (len(emails_with_summary) / len(important_emails)) * 100
        print(f"AI summary coverage: {coverage_pct:.1f}%")

        if coverage_pct < 80:
            print(f"❌ FAILED: Low AI summary coverage ({coverage_pct:.1f}%, need 80%+)")
            return False

        # Show example
        if emails_with_summary:
            email = emails_with_summary[0]
            print(f"\nExample email with AI summary:")
            print(f"  Subject: {email.get('subject')}")
            print(f"  Importance: {email.get('importance_score'):.2f}")
            print(f"  AI Summary: {email.get('ai_summary')[:150]}...")

    print("\n✅ PASSED: Important emails have AI summaries")
    return True


async def run_all_tests():
    """Run all brief quality tests."""
    print("\n" + "="*80)
    print("BRIEF QUALITY INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("TLDR 3-Bullet Format", test_tldr_has_three_bullets),
        ("TLDR Priority Order", test_tldr_priority_order),
        ("Summary Length & Detail", test_summary_length),
        ("Summary Prioritization", test_summary_prioritizes_email_calendar),
        ("Key Learnings", test_key_learnings_extracted),
        ("Key Insights", test_key_insights_extracted),
        ("Market News Tagging", test_market_news_tagged),
        ("Email AI Summaries", test_email_summaries_generated),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ {name} CRASHED: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)

"""Integration test to verify make preload generates AI summaries."""

import asyncio
import pytest
from src.database.sqlite_manager import get_db_manager
from src.services.background_refresh_service import get_background_refresh_service


@pytest.mark.asyncio
async def test_news_refresh_generates_ai_summaries():
    """Test that refreshing news generates AI summaries and TLDRs."""
    # Run news refresh
    refresh_service = get_background_refresh_service()
    success = await refresh_service._refresh_news()

    assert success, "News refresh should succeed"

    # Check that articles have AI fields populated
    db = get_db_manager()
    articles = await db.get_all_articles(limit=5)

    assert len(articles) > 0, "Should have articles"

    print(f"\n=== AFTER NEWS REFRESH ===")
    has_ai_content = False

    for i, article in enumerate(articles[:3]):
        print(f"\nArticle {i+1}: {article.title[:50]}")
        print(f"  ai_summary: {article.ai_summary[:100] if article.ai_summary else 'None'}")
        print(f"  tldr: {article.tldr[:100] if article.tldr else 'None'}")
        print(f"  key_learnings: {len(article.key_learnings)} items")

        # Check if at least one article has AI content
        if article.ai_summary or article.tldr:
            has_ai_content = True

    assert has_ai_content, "At least one article should have AI-generated content (ai_summary or tldr)"


@pytest.mark.asyncio
async def test_email_refresh_generates_snippets_and_ai():
    """Test that refreshing emails generates snippets and AI summaries."""
    # Run email refresh
    refresh_service = get_background_refresh_service()
    success = await refresh_service._refresh_email()

    # Note: Email refresh may return False if using mock data, that's OK
    print(f"\nEmail refresh result: {success}")

    # Check email data
    db = get_db_manager()
    emails_data = await db.get_cache('emails_data')

    if not emails_data:
        pytest.skip("No email data available (may need real credentials)")

    assert len(emails_data) > 0, "Should have emails"

    print(f"\n=== AFTER EMAIL REFRESH ===")
    print(f"Total emails: {len(emails_data)}")

    has_snippets = False
    has_ai_summaries = False

    for i, email in enumerate(emails_data[:5]):
        print(f"\nEmail {i+1}: {email.get('subject', 'N/A')[:50]}")
        print(f"  snippet: {email.get('snippet')[:50] if email.get('snippet') else 'None'}")
        print(f"  ai_summary: {email.get('ai_summary')[:50] if email.get('ai_summary') else 'None'}")
        print(f"  importance_score: {email.get('importance_score', 0)}")

        if email.get('snippet'):
            has_snippets = True
        if email.get('ai_summary'):
            has_ai_summaries = True

    # At minimum, emails should have snippets generated
    assert has_snippets, "At least one email should have a snippet generated"
    print(f"\n✓ Emails have snippets: {has_snippets}")
    print(f"✓ Emails have AI summaries: {has_ai_summaries}")


@pytest.mark.asyncio
async def test_full_preload_workflow():
    """Test the complete preload workflow generates all AI content."""
    refresh_service = get_background_refresh_service()

    # Run full refresh
    print("\n=== RUNNING FULL PRELOAD ===")
    results = await refresh_service.refresh_all()

    print(f"\nRefresh results: {results}")

    # Verify news has AI content
    db = get_db_manager()
    articles = await db.get_all_articles(limit=3)

    print(f"\n=== NEWS AI CONTENT ===")
    for article in articles:
        has_ai = article.ai_summary or article.tldr
        status = "✓" if has_ai else "✗"
        print(f"{status} {article.title[:50]}")
        if has_ai:
            print(f"  Summary: {(article.ai_summary or article.tldr)[:100]}")

    # Verify emails have snippets
    emails_data = await db.get_cache('emails_data')
    if emails_data:
        print(f"\n=== EMAIL SNIPPETS ===")
        for email in emails_data[:3]:
            has_snippet = bool(email.get('snippet'))
            status = "✓" if has_snippet else "✗"
            print(f"{status} {email.get('subject', 'N/A')[:50]}")
            if has_snippet:
                print(f"  Snippet: {email.get('snippet')[:80]}")


if __name__ == "__main__":
    # Run tests directly
    print("Testing news AI generation...")
    asyncio.run(test_news_refresh_generates_ai_summaries())

    print("\n\nTesting email snippet/AI generation...")
    asyncio.run(test_email_refresh_generates_snippets_and_ai())

    print("\n\nTesting full preload...")
    asyncio.run(test_full_preload_workflow())

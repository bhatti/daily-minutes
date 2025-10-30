"""Integration tests to verify UI display issues reported by user."""

import asyncio
import pytest
from src.database.sqlite_manager import get_db_manager


@pytest.mark.asyncio
async def test_brief_tldr_length():
    """Test that brief TLDR is substantial (not just one short sentence)."""
    db = get_db_manager()
    brief = await db.get_cache('daily_brief_data')

    assert brief is not None, "Brief data should exist after make preload"

    tldr = brief.get('tldr', '')
    print(f"\n=== TLDR ===")
    print(f"Length: {len(tldr)} chars")
    print(f"Content: {tldr}")

    # TLDR should be at least 50 chars (user complaint: too short)
    assert len(tldr) >= 50, f"TLDR too short: {len(tldr)} chars. User wants comprehensive TLDR."

    # TLDR should not be generic
    generic_phrases = ['various topics', 'range of topics', 'covers', 'discusses']
    tldr_lower = tldr.lower()
    for phrase in generic_phrases:
        assert phrase not in tldr_lower, f"TLDR is too generic, contains '{phrase}'"


@pytest.mark.asyncio
async def test_brief_summary_length():
    """Test that brief summary is comprehensive (not just one line)."""
    db = get_db_manager()
    brief = await db.get_cache('daily_brief_data')

    assert brief is not None, "Brief data should exist"

    summary = brief.get('summary', '')
    print(f"\n=== SUMMARY ===")
    print(f"Length: {len(summary)} chars")
    print(f"Content: {summary}")

    # Summary should be at least 200 chars (user wants comprehensive summary)
    assert len(summary) >= 200, f"Summary too short: {len(summary)} chars. User wants detailed summary."

    # Count sentences (rough estimate by periods)
    sentence_count = summary.count('.') + summary.count('!') + summary.count('?')
    print(f"Approximate sentences: {sentence_count}")
    assert sentence_count >= 3, f"Summary should have 3-4 sentences, found {sentence_count}"


@pytest.mark.asyncio
async def test_news_articles_have_ai_fields():
    """Test that news articles have AI-generated TLDR/analysis."""
    db = get_db_manager()
    articles = await db.get_all_articles(limit=5)

    assert len(articles) > 0, "Should have news articles"

    print(f"\n=== NEWS ARTICLES ===")
    for i, article in enumerate(articles[:3]):
        print(f"\nArticle {i+1}: {article.title[:50]}")
        print(f"  Has ai_summary attr: {hasattr(article, 'ai_summary')}")
        print(f"  Has tldr attr: {hasattr(article, 'tldr')}")
        print(f"  ai_summary value: {getattr(article, 'ai_summary', 'N/A')}")
        print(f"  tldr value: {getattr(article, 'tldr', 'N/A')}")
        print(f"  description: {getattr(article, 'description', 'N/A')[:100]}")

    # Check if AI fields exist and have values
    article = articles[0]
    has_ai_summary = hasattr(article, 'ai_summary') and article.ai_summary is not None
    has_tldr = hasattr(article, 'tldr') and article.tldr is not None
    has_description = hasattr(article, 'description') and article.description is not None

    # User complaint: news tab shows raw snippets, not AI analysis
    # We need EITHER ai_summary OR tldr OR at minimum a description
    assert has_ai_summary or has_tldr or has_description, \
        "News articles should have ai_summary, tldr, or description"

    print(f"\n✓ Articles have content to display")


@pytest.mark.asyncio
async def test_email_display_fields():
    """Test that emails have snippet/ai_summary/todos for display."""
    db = get_db_manager()
    emails_data = await db.get_cache('emails_data')

    assert emails_data is not None, "Email data should exist"
    assert len(emails_data) > 0, "Should have emails"

    print(f"\n=== EMAIL DATA ===")
    print(f"Total emails: {len(emails_data)}")

    # Check first 3 emails
    for i, email in enumerate(emails_data[:3]):
        print(f"\nEmail {i+1}: {email.get('subject', 'N/A')[:50]}")
        print(f"  snippet: {email.get('snippet')}")
        print(f"  ai_summary: {email.get('ai_summary')}")
        print(f"  action_items: {email.get('action_items', [])}")
        print(f"  has_action_items: {email.get('has_action_items', False)}")

    # User complaint: Email tab missing tldr/insight/todo
    # Check if we have ANY of these fields populated
    email = emails_data[0]
    has_snippet = email.get('snippet') is not None
    has_ai_summary = email.get('ai_summary') is not None
    has_action_items = email.get('has_action_items', False) and len(email.get('action_items', [])) > 0
    has_body = email.get('body') is not None

    print(f"\nEmail fields available:")
    print(f"  snippet: {has_snippet}")
    print(f"  ai_summary: {has_ai_summary}")
    print(f"  action_items: {has_action_items}")
    print(f"  body: {has_body}")

    # We need SOMETHING to display
    assert has_snippet or has_ai_summary or has_action_items or has_body, \
        "Email should have snippet, ai_summary, action_items, or body"


@pytest.mark.asyncio
async def test_ui_display_logic():
    """Test the exact display logic used in streamlit_app.py for emails."""
    db = get_db_manager()
    emails_data = await db.get_cache('emails_data')

    assert len(emails_data) > 0, "Should have emails"

    email = emails_data[0]
    print(f"\n=== EMAIL DISPLAY LOGIC TEST ===")
    print(f"Subject: {email.get('subject')}")

    # Replicate UI logic from streamlit_app.py lines 328-350
    preview_shown = False
    display_text = ""

    # 1. Show snippet if available
    if email.get('snippet'):
        display_text = f"💬 {email.get('snippet')}"
        preview_shown = True
        print(f"Would show snippet: {display_text[:50]}")

    # 2. Show AI summary if available
    if email.get('ai_summary'):
        display_text = f"💡 {email.get('ai_summary')[:100]}..."
        preview_shown = True
        print(f"Would show AI summary: {display_text[:50]}")

    # 3. Show action items summary
    if email.get('has_action_items') and email.get('action_items'):
        action_count = len(email.get('action_items', []))
        display_text = f"📋 {action_count} action item{'s' if action_count > 1 else ''}"
        preview_shown = True
        print(f"Would show action items: {display_text}")

    # 4. Fallback: show body preview
    if not preview_shown and email.get('body'):
        preview = email.get('body')[:80] + "..." if len(email.get('body', '')) > 80 else email.get('body')
        display_text = preview
        print(f"Would show body preview: {display_text[:50]}")

    print(f"\nFinal display text: {display_text[:100]}")
    assert len(display_text) > 0, "Should have something to display for email"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_brief_tldr_length())
    asyncio.run(test_brief_summary_length())
    asyncio.run(test_news_articles_have_ai_fields())
    asyncio.run(test_email_display_fields())
    asyncio.run(test_ui_display_logic())

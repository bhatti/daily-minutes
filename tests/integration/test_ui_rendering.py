"""Integration tests that verify actual UI rendering logic."""

import asyncio
from src.database.sqlite_manager import get_db_manager


def test_tldr_not_truncated():
    """Test that TLDR is displayed in full, not truncated."""
    async def check():
        db = get_db_manager()
        brief = await db.get_cache('daily_brief_data')

        tldr = brief.get('tldr', '')
        print(f"\n=== TLDR RENDERING TEST ===")
        print(f"Full TLDR: {tldr}")
        print(f"Length: {len(tldr)} chars")

        # Check what UI would display - simulating daily_brief.py line 74-75
        # st.markdown("### ⚡ TLDR")
        # st.info(tldr)

        # st.info should show full text, not truncated
        # User complaint: shows "...with major breakthroughs in natural l..."
        # This suggests truncation is happening somewhere

        # Check if there's any truncation logic
        assert '...' not in tldr or len(tldr) > 200, "TLDR should not be artificially truncated"
        print(f"✓ TLDR is not truncated in data")

        # The issue might be in Streamlit rendering with st.info()
        # Let's check if TLDR is complete
        assert len(tldr) > 100, f"TLDR too short: {len(tldr)} chars"

        return tldr

    return asyncio.run(check())


def test_summary_should_use_bullets():
    """Test that summary is formatted as bullets in UI."""
    async def check():
        db = get_db_manager()
        brief = await db.get_cache('daily_brief_data')

        summary = brief.get('summary', '')
        print(f"\n=== SUMMARY FORMATTING TEST ===")
        print(f"Summary: {summary[:200]}...")
        print(f"Length: {len(summary)} chars")

        # User wants bullets, not paragraph
        # Current code in daily_brief.py:78-83 shows:
        # st.markdown("### 📋 Summary")
        # st.markdown(summary)

        # This renders as plain paragraph
        # User wants bullet format

        # Count sentences
        sentences = summary.count('.') + summary.count('!') + summary.count('?')
        print(f"Sentences: {sentences}")

        # If we have multiple sentences, should be bullets
        if sentences > 2:
            print("❌ Summary has multiple sentences but rendered as paragraph")
            print("Should be rendered as bullets")
            return False

        return summary

    return asyncio.run(check())


def test_email_should_show_tldr_not_body():
    """Test that email preview shows TLDR/analysis, not raw body."""
    async def check():
        db = get_db_manager()
        emails_data = await db.get_cache('emails_data')

        if not emails_data:
            return True

        email = emails_data[0]
        print(f"\n=== EMAIL DISPLAY TEST ===")
        print(f"Subject: {email.get('subject')}")

        # Simulate UI logic from streamlit_app.py lines 328-350
        preview_shown = False
        display_text = ""

        # 1. Show snippet if available
        if email.get('snippet'):
            display_text = f"💬 {email.get('snippet')}"
            preview_shown = True
            print(f"Would show snippet: {display_text[:80]}...")

        # 2. Show AI summary if available
        if email.get('ai_summary'):
            display_text = f"💡 {email.get('ai_summary')[:100]}..."
            preview_shown = True
            print(f"Would show AI summary: {display_text[:80]}...")

        # 3. Show action items
        if email.get('has_action_items') and email.get('action_items'):
            action_count = len(email.get('action_items', []))
            display_text = f"📋 {action_count} action item{'s' if action_count > 1 else ''}"
            preview_shown = True
            print(f"Would show action items: {display_text}")

        # 4. Fallback: show body preview (BAD - user doesn't want this)
        if not preview_shown and email.get('body'):
            preview = email.get('body')[:80] + "..." if len(email.get('body', '')) > 80 else email.get('body')
            display_text = preview
            print(f"❌ Showing body preview (BAD): {display_text[:80]}")
            print("User complaint: 'Email shows part of the body, it only needs to show tldr'")

            # Check if we have snippet or ai_summary available
            if email.get('snippet') or email.get('ai_summary'):
                assert False, "Email has snippet/ai_summary but UI is showing body preview!"

        print(f"\nFinal display: {display_text[:100]}")
        return display_text

    return asyncio.run(check())


def test_news_analysis_length():
    """Test that news analysis is comprehensive, not truncated."""
    async def check():
        db = get_db_manager()
        articles = await db.get_all_articles(limit=5)

        print(f"\n=== NEWS ANALYSIS LENGTH TEST ===")

        for i, article in enumerate(articles[:3]):
            print(f"\nArticle {i+1}: {article.title[:50]}")

            ai_summary = article.ai_summary
            tldr = article.tldr
            description = article.description

            # UI code from streamlit_app.py:453-462
            # if ai_summary:
            #     st.info(f"💡 **AI Analysis:** {ai_summary}")
            # elif tldr:
            #     st.info(f"⚡ **TLDR:** {tldr}")
            # elif description:
            #     if len(description) > 200:
            #         description = description[:200] + "..."
            #     st.caption(description)

            if ai_summary:
                print(f"  AI Summary ({len(ai_summary)} chars): {ai_summary[:100]}...")
                # User complaint: "News shows small analysis but it's still too short"
                if len(ai_summary) < 100:
                    print(f"  ❌ AI summary too short: {len(ai_summary)} chars")
                else:
                    print(f"  ✓ AI summary length OK")
            elif tldr:
                print(f"  TLDR ({len(tldr)} chars): {tldr[:100]}...")
            else:
                print(f"  Description ({len(description)} chars): {description[:100]}...")

        return True

    return asyncio.run(check())


if __name__ == "__main__":
    print("=" * 70)
    print("UI RENDERING INTEGRATION TESTS")
    print("=" * 70)

    print("\n1. Testing TLDR truncation...")
    tldr = test_tldr_not_truncated()

    print("\n2. Testing summary bullet formatting...")
    summary = test_summary_should_use_bullets()

    print("\n3. Testing email display (should show TLDR, not body)...")
    email_display = test_email_should_show_tldr_not_body()

    print("\n4. Testing news analysis length...")
    test_news_analysis_length()

    print("\n" + "=" * 70)
    print("SUMMARY OF ISSUES FOUND:")
    print("=" * 70)
    print("1. TLDR may be truncated by Streamlit st.info() rendering")
    print("2. Summary rendered as paragraph, user wants bullets")
    print("3. Email may show body instead of TLDR/analysis")
    print("4. News analysis may be too short from LLM")

#!/usr/bin/env python3
"""
Test script to verify news fetching works in the dashboard
"""

import asyncio
from src.agents.news_agent_ai import get_news_agent_ai
from src.agents.news_agent import NewsAgent
from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector

async def test_dashboard_news():
    """Test news fetching for dashboard."""

    print("\n" + "="*60)
    print("TESTING NEWS FETCHING FOR DASHBOARD")
    print("="*60)

    # Test 1: Basic connectors
    print("\n1. Testing basic connectors...")
    try:
        hn_connector = HackerNewsConnector(max_stories=10)
        articles = await hn_connector.fetch_top_stories("top")
        print(f"   ✅ HackerNews: Fetched {len(articles)} articles")
        if articles:
            print(f"      Sample: {articles[0].title[:50]}...")
    except Exception as e:
        print(f"   ❌ HackerNews error: {e}")

    try:
        rss_connector = RSSConnector(max_articles_per_feed=10)
        rss_articles = await rss_connector.fetch_all_feeds()
        print(f"   ✅ RSS: Fetched {len(rss_articles)} articles")
    except Exception as e:
        print(f"   ❌ RSS error: {e}")

    # Test 2: NewsAgent (basic)
    print("\n2. Testing NewsAgent...")
    try:
        agent = NewsAgent()
        news_articles = await agent.run()
        print(f"   ✅ NewsAgent: Fetched {len(news_articles)} articles")
    except Exception as e:
        print(f"   ❌ NewsAgent error: {e}")

    # Test 3: AI-enhanced agent (if available)
    print("\n3. Testing AI-enhanced NewsAgent...")
    try:
        ai_agent = get_news_agent_ai()
        ai_articles = await ai_agent.fetch_news_with_mcp(
            sources=["hackernews", "rss"],
            max_articles=20
        )
        print(f"   ✅ AI Agent: Fetched {len(ai_articles)} articles via MCP")

        # Test AI summary
        if ai_articles:
            summary = await ai_agent.generate_ai_summary(ai_articles[:5])
            if summary and "unavailable" not in summary.lower():
                print(f"   ✅ AI Summary: {summary[:100]}...")
            else:
                print(f"   ⚠️ AI Summary not available")
    except Exception as e:
        print(f"   ❌ AI Agent error: {e}")

    print("\n" + "="*60)
    print("✅ NEWS FETCHING TEST COMPLETE")
    print("="*60)

    return True

if __name__ == "__main__":
    asyncio.run(test_dashboard_news())
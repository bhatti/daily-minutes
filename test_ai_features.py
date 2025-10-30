#!/usr/bin/env python3
"""
Quick test script to verify AI features are working
"""

import asyncio
import sys
from src.agents.news_agent_ai import get_news_agent_ai
from src.services.ollama_service import get_ollama_service
from src.core.logging import get_logger

logger = get_logger(__name__)

async def test_ai_features():
    """Test the AI features."""

    print("\n" + "="*60)
    print("TESTING AI FEATURES")
    print("="*60)

    # Check Ollama
    print("\n1. Checking Ollama availability...")
    ollama = get_ollama_service()
    is_available = await ollama.check_availability()

    if not is_available:
        print("❌ Ollama is not running!")
        print("   Please start Ollama with: ollama serve")
        return False
    else:
        print("✅ Ollama is available")

    # Test news agent AI
    print("\n2. Testing AI-enhanced News Agent...")
    agent = get_news_agent_ai()

    # Fetch news
    print("   Fetching news via MCP...")
    articles = await agent.fetch_news_with_mcp(sources=["hackernews"], max_articles=5)

    if articles:
        print(f"   ✅ Fetched {len(articles)} articles")

        # Test AI summary
        print("\n3. Testing AI Summary Generation...")
        summary = await agent.generate_ai_summary(articles)
        if summary and "unavailable" not in summary.lower():
            print(f"   ✅ Generated summary: {summary[:100]}...")
        else:
            print("   ❌ Summary generation failed")

        # Test insights
        print("\n4. Testing Insights Extraction...")
        insights = await agent.extract_insights(articles)
        if insights:
            print(f"   ✅ Extracted {len(insights)} insights:")
            for insight in insights[:3]:
                print(f"      - {insight}")
        else:
            print("   ❌ Insights extraction failed")

        # Test semantic search
        print("\n5. Testing Semantic Search...")
        similar = await agent.find_similar_articles("artificial intelligence", max_results=3)
        if similar:
            print(f"   ✅ Found {len(similar)} similar articles")
        else:
            print("   ⚠️ No similar articles found (this is normal if RAG is empty)")

        # Test Q&A
        print("\n6. Testing Q&A with RAG...")
        answer = await agent.answer_question("What are the latest tech news?")
        if answer and answer.get("answer"):
            print(f"   ✅ Q&A working: {answer['answer'][:100]}...")
        else:
            print("   ⚠️ Q&A returned no answer (this is normal if RAG is empty)")

        print("\n" + "="*60)
        print("✅ AI FEATURES TEST COMPLETE")
        print("="*60)
        return True
    else:
        print("   ❌ Failed to fetch articles")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ai_features())
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""Comprehensive integration tests for UI functionality."""

import asyncio
import sys
import pytest
sys.path.append('.')

from streamlit_app import fetch_and_analyze_news, fetch_news_basic

@pytest.mark.asyncio
async def test_ui_fetch():
    """Test the UI's fetch functionality."""
    print("\n=== Testing UI Fetch Functionality ===")

    # Test basic fetch first
    print("\n1. Testing basic fetch...")
    try:
        articles = await fetch_news_basic(max_articles=5)
        assert articles, "Basic fetch returned no articles"
        assert len(articles) > 0, "Basic fetch returned empty list"
        print(f"✅ Basic fetch: {len(articles)} articles")
    except Exception as e:
        print(f"❌ Basic fetch failed: {e}")
        return False

    # Test full fetch_and_analyze_news
    print("\n2. Testing fetch_and_analyze_news...")
    try:
        articles, summary, insights = await fetch_and_analyze_news(
            max_articles=5,
            progress_callback=None,
            model="llama3.2"
        )

        # Articles should always be returned (even if AI fails)
        assert articles is not None, "Articles is None"
        assert isinstance(articles, list), f"Articles is not a list: {type(articles)}"

        if len(articles) > 0:
            print(f"✅ Fetch and analyze: {len(articles)} articles")
            print(f"   Summary: {summary[:50]}..." if summary else "   No summary")
            print(f"   Insights: {len(insights)} insights" if insights else "   No insights")
        else:
            print("⚠️ Fetch returned 0 articles (may be a connection issue)")

        # Even with 0 articles, the function should not fail
        assert summary is not None, "Summary is None"
        assert insights is not None, "Insights is None"

    except Exception as e:
        print(f"❌ Fetch and analyze failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ UI Fetch tests passed!")
    return True

def test_ai_agent_initialization():
    """Test AI agent initialization outside of Streamlit context."""
    print("\n=== Testing AI Agent Initialization ===")

    # Test imports
    print("\n1. Testing imports...")
    try:
        from src.agents.news_agent_ai import get_news_agent_ai
        from src.services.ollama_service import get_ollama_service
        from src.services.rag_service import get_rag_service
        from src.services.langgraph_orchestrator import get_orchestrator
        print("✅ All AI modules imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test agent creation
    print("\n2. Testing AI agent creation...")
    try:
        agent = get_news_agent_ai()
        print(f"✅ AI agent created: {type(agent).__name__}")

        # Check agent has required methods
        assert hasattr(agent, 'fetch_news_with_mcp'), "Agent missing fetch_news_with_mcp"
        assert hasattr(agent, 'generate_ai_summary'), "Agent missing generate_ai_summary"
        assert hasattr(agent, 'extract_insights'), "Agent missing extract_insights"
        print("✅ Agent has all required methods")

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ AI Agent initialization tests passed!")
    return True

@pytest.mark.asyncio
async def test_ui_with_ai_agent():
    """Test UI fetch with AI agent enabled."""
    print("\n=== Testing UI Fetch with AI Agent ===")

    # First ensure AI agent can be created
    try:
        from src.agents.news_agent_ai import get_news_agent_ai
        agent = get_news_agent_ai()
        print(f"✅ AI agent available: {type(agent).__name__}")
    except Exception as e:
        print(f"⚠️ AI agent not available: {e}")
        print("   Skipping AI-enhanced fetch test")
        return True  # Not a failure, just skip

    # Test fetch with MCP
    print("\n1. Testing fetch with MCP...")
    try:
        articles = await agent.fetch_news_with_mcp(
            sources=["hackernews"],
            max_articles=3
        )

        if articles and len(articles) > 0:
            print(f"✅ MCP fetch: {len(articles)} articles")
            print(f"   Sample: {articles[0].title[:50]}...")
        else:
            print("⚠️ MCP fetch returned no articles (may be normal)")

    except Exception as e:
        print(f"⚠️ MCP fetch failed: {e}")
        # Not a hard failure as MCP might not be fully configured

    # Test AI summary generation
    print("\n2. Testing AI summary generation...")
    try:
        # Get some articles first
        from src.connectors.hackernews import HackerNewsConnector
        connector = HackerNewsConnector(max_stories=3)
        articles = await connector.execute_async()

        if articles:
            summary = await agent.generate_ai_summary(articles, model="llama3.2")
            if summary and len(summary) > 0:
                print(f"✅ AI summary generated: {summary[:100]}...")
            else:
                print("⚠️ Empty summary generated")
        else:
            print("⚠️ No articles to summarize")

    except Exception as e:
        print(f"⚠️ AI summary failed: {e}")
        # Not a hard failure as Ollama might not be running

    print("\n✅ UI with AI Agent tests completed!")
    return True

@pytest.mark.asyncio
async def test_ui_model_management():
    """Test UI's model management functionality."""
    print("\n=== Testing Model Management ===")

    try:
        from src.services.ollama_service import get_ollama_service
        ollama = get_ollama_service()

        # Check if Ollama is available
        is_available = await ollama.check_availability()
        if not is_available:
            print("⚠️ Ollama not running - skipping model tests")
            return True

        # List installed models
        models = await ollama.list_models()
        print(f"✅ Found {len(models)} installed models: {', '.join(models)}")

        # Check if llama3.2 is installed (needed for UI)
        if "llama3.2" not in models and "llama3.2:latest" not in models:
            print("⚠️ llama3.2 not installed - UI may fall back to basic fetch")
        else:
            print("✅ llama3.2 is installed")

    except Exception as e:
        print(f"⚠️ Model management test failed: {e}")
        # Not a hard failure

    print("\n✅ Model management tests completed!")
    return True

@pytest.mark.asyncio
async def test_ai_summary_generation():
    """Test AI summary generation with actual articles."""
    print("\n=== Testing AI Summary Generation ===")

    try:
        from src.agents.news_agent_ai import get_news_agent_ai
        from src.connectors.hackernews import HackerNewsConnector

        # Get some real articles
        connector = HackerNewsConnector(max_stories=5)
        articles = await connector.execute_async()

        if not articles:
            print("⚠️ No articles fetched, skipping summary test")
            return True

        print(f"✅ Fetched {len(articles)} articles for summary")

        # Test summary generation
        agent = get_news_agent_ai()
        summary = await agent.generate_ai_summary(articles, model="llama3.2")

        if summary and len(summary) > 0:
            print(f"✅ AI Summary generated: {summary[:100]}...")
            if "unavailable" in summary.lower() or "failed" in summary.lower():
                print("⚠️ Summary indicates AI service issue")
                return False
            return True
        else:
            print("❌ Empty summary generated")
            return False

    except Exception as e:
        print(f"❌ AI Summary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_key_insights_extraction():
    """Test key insights extraction."""
    print("\n=== Testing Key Insights Extraction ===")

    try:
        from src.agents.news_agent_ai import get_news_agent_ai
        from src.connectors.hackernews import HackerNewsConnector

        # Get some real articles
        connector = HackerNewsConnector(max_stories=10)
        articles = await connector.execute_async()

        if not articles:
            print("⚠️ No articles fetched, skipping insights test")
            return True

        print(f"✅ Fetched {len(articles)} articles for insights")

        # Test insights extraction
        agent = get_news_agent_ai()
        insights = await agent.extract_insights(articles, model="llama3.2")

        if insights and len(insights) > 0:
            print(f"✅ Extracted {len(insights)} insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"   {i}. {insight[:80]}...")

            # Check for error messages in insights
            error_keywords = ["failed", "error", "retryerror", "valueerror"]
            has_errors = any(any(kw in str(insight).lower() for kw in error_keywords) for insight in insights)

            if has_errors:
                print("❌ Insights contain error messages")
                return False

            return True
        else:
            print("❌ No insights extracted")
            return False

    except Exception as e:
        print(f"❌ Insights extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_ai_qa_assistant():
    """Test AI Q&A Assistant."""
    print("\n=== Testing AI Q&A Assistant ===")

    try:
        from src.agents.news_agent_ai import get_news_agent_ai

        agent = get_news_agent_ai()

        # Test question answering
        question = "What are the top tech trends?"
        result = await agent.answer_question(question)

        if result and isinstance(result, dict):
            answer = result.get("answer", "")
            if answer and "event loop is closed" not in answer.lower():
                print(f"✅ Q&A working: {answer[:100]}...")
                return True
            else:
                print(f"❌ Q&A failed or has event loop error: {answer}")
                return False
        else:
            print("❌ No result from Q&A")
            return False

    except Exception as e:
        print(f"❌ Q&A test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_orchestrator_workflow():
    """Test LangGraph orchestrator workflow."""
    print("\n=== Testing Orchestrator Workflow ===")

    try:
        from src.agents.news_agent_ai import get_news_agent_ai

        agent = get_news_agent_ai()

        # Test orchestrated workflow
        result = await agent.run_orchestrated_workflow(
            user_request="Give me analysis of top trends",
            preferences={"sources": ["hackernews"], "topics": ["technology"]}
        )

        if result and isinstance(result, dict):
            if result.get("success") or result.get("data"):
                print(f"✅ Orchestrator workflow completed")
                return True
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ Orchestrator failed: {error}")
                return False
        else:
            print("❌ No result from orchestrator")
            return False

    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_article_content_and_excerpts():
    """Test article content fetching and excerpt generation with ContentFetcher."""
    print("\n=== Testing Article Content & Excerpts ===")

    try:
        from src.connectors.hackernews import HackerNewsConnector
        from streamlit_app import enrich_articles_with_excerpts

        # Fetch articles
        connector = HackerNewsConnector(max_stories=3)
        articles = await connector.execute_async()

        if not articles:
            print("⚠️ No articles fetched")
            return True

        print(f"✅ Fetched {len(articles)} articles")

        # Count articles without descriptions initially
        before_count = len([a for a in articles if a.description and len(a.description) > 50])
        print(f"   Articles with content before enrichment: {before_count}/{len(articles)}")

        # Enrich articles with excerpts using ContentFetcher
        print("   Enriching articles with excerpts...")
        enriched_articles = await enrich_articles_with_excerpts(articles, max_to_fetch=3)

        # Check if articles have descriptions/excerpts after enrichment
        after_count = len([a for a in enriched_articles if a.description and len(a.description) > 50])
        print(f"   Articles with content after enrichment: {after_count}/{len(enriched_articles)}")

        if after_count > before_count:
            print(f"✅ Excerpt enrichment working! Added {after_count - before_count} excerpts")
            return True
        elif after_count > 0:
            print(f"✅ Found {after_count} articles with excerpts")
            return True
        else:
            print("⚠️ No articles have excerpts after enrichment")
            print("   This might indicate network/content fetching issues")
            return True  # Not a hard failure - could be network issues

        return True

    except Exception as e:
        print(f"❌ Article content test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_weather_integration():
    """Test weather service integration and data fetching."""
    print("\n=== Testing Weather Integration ===")

    try:
        from src.connectors.weather import get_weather_service

        # Get weather service instance
        weather_service = get_weather_service()
        print(f"✅ Weather service initialized: {type(weather_service).__name__}")

        # Check that connectors are available
        if not weather_service.connectors:
            print("❌ No weather connectors available")
            return False

        print(f"   Available connectors: {len(weather_service.connectors)}")

        # Test fetching current weather for a test location
        test_location = "San Francisco"
        print(f"   Fetching weather for {test_location}...")

        weather_data = await weather_service.get_current_weather(test_location)

        if weather_data:
            print(f"✅ Weather data fetched successfully")
            print(f"   Location: {weather_data.location}")
            print(f"   Temperature: {weather_data.temperature}°C")
            print(f"   Description: {weather_data.description}")
            print(f"   Humidity: {weather_data.humidity}%")
            print(f"   Wind: {weather_data.wind_speed} m/s")

            # Verify weather data has required fields
            assert weather_data.location is not None, "Location should not be None"
            assert weather_data.temperature is not None, "Temperature should not be None"
            assert weather_data.description is not None, "Description should not be None"

            # Test converting weather to article format
            article = weather_service.to_article(weather_data)
            print(f"✅ Weather converted to article format")
            print(f"   Article title: {article.title[:50]}...")

            assert article.title is not None, "Article title should not be None"
            assert "weather" in [t.lower() for t in article.tags], "Article should have weather tag"

            return True
        else:
            print("⚠️ Weather data fetch returned None")
            print("   This might indicate API issues or network problems")
            return True  # Not a hard failure - API might be down

    except Exception as e:
        print(f"❌ Weather integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all UI integration tests."""
    print("=" * 60)
    print("UI INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    # Test basic UI fetch
    if not await test_ui_fetch():
        all_passed = False

    # Test AI agent initialization
    if not test_ai_agent_initialization():
        all_passed = False

    # Test UI with AI agent
    if not await test_ui_with_ai_agent():
        all_passed = False

    # Test model management
    if not await test_ui_model_management():
        all_passed = False

    # NEW TESTS for broken features
    if not await test_ai_summary_generation():
        all_passed = False

    if not await test_key_insights_extraction():
        all_passed = False

    if not await test_ai_qa_assistant():
        all_passed = False

    if not await test_orchestrator_workflow():
        all_passed = False

    if not await test_article_content_and_excerpts():
        all_passed = False

    # Test weather integration
    if not await test_weather_integration():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL UI INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME UI INTEGRATION TESTS FAILED")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
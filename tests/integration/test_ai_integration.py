"""
Integration tests for AI components.

IMPORTANT: These tests require Ollama to be running locally.

Run Ollama first:
    ollama serve

Then run tests:
    pytest tests/integration/test_ai_integration.py -v

Or run directly:
    python tests/integration/test_ai_integration.py
"""

import asyncio
import os
import pytest
from datetime import datetime

# Test if Ollama is available
OLLAMA_AVAILABLE = os.getenv("OLLAMA_AVAILABLE", "false").lower() == "true"

from src.services.ollama_service import OllamaService, OllamaConfig
from src.services.mcp_server import MCPServer
from src.services.rag_service import RAGService, RAGConfig
from src.services.langgraph_orchestrator import LangGraphOrchestrator
from src.models.news import NewsArticle, DataSource, Priority
from src.connectors.hackernews import HackerNewsConnector


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not available")
async def test_ollama_service():
    """Test Ollama service basic functionality."""
    print("\n=== Testing Ollama Service ===")

    # Create service
    config = OllamaConfig(
        model="llama3.2:1b",  # Use smaller model for testing
        temperature=0.5,
        max_tokens=100
    )
    service = OllamaService(config)

    # Check availability
    is_available = await service.check_availability()
    if not is_available:
        print("⚠ Ollama server not running. Skipping test.")
        pytest.skip("Ollama server not available")

    print(f"✓ Ollama is available")

    # List models
    models = await service.list_models()
    print(f"✓ Found {len(models)} models: {', '.join(models[:3])}")

    # Test generation
    response = await service.generate(
        prompt="What is Python in one sentence?",
        temperature=0.3,
        max_tokens=50
    )
    assert response.content
    print(f"✓ Generated response: {response.content[:100]}...")

    # Test summarization
    text = "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a vast ecosystem of libraries."
    summary = await service.summarize_text(text, max_length=50)
    assert summary
    print(f"✓ Summary: {summary[:100]}...")

    # Test keyword extraction
    keywords = await service.extract_keywords(text, max_keywords=5)
    assert len(keywords) <= 5
    print(f"✓ Keywords: {', '.join(keywords)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server():
    """Test MCP server functionality."""
    print("\n=== Testing MCP Server ===")

    server = MCPServer()

    # List tools
    tools = await server.list_tools()
    assert len(tools) > 0
    print(f"✓ Found {len(tools)} tools")

    # Test tool execution (mock HackerNews)
    response = await server.execute_tool(
        "fetch_hackernews",
        {"story_type": "top", "max_stories": 2}
    )

    if response.success:
        assert response.data is not None
        print(f"✓ Fetched {len(response.data)} articles")
    else:
        print(f"⚠ Tool execution failed: {response.error}")

    # Test invalid tool
    response = await server.execute_tool("invalid_tool", {})
    assert not response.success
    print("✓ Invalid tool handled correctly")

    # Test MCP request handling
    request = {
        "method": "tools/list",
        "params": {}
    }
    result = await server.handle_request(request)
    assert "tools" in result
    print(f"✓ MCP request handled: {len(result['tools'])} tools listed")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not available")
async def test_rag_service():
    """Test RAG service functionality."""
    print("\n=== Testing RAG Service ===")

    # Create service with test configuration
    config = RAGConfig(
        persist_directory="./data/test_chroma",
        collection_name="test_daily_minutes",
        embedding_model="nomic-embed-text"
    )

    # Check if Ollama is available
    ollama = OllamaService()
    is_available = await ollama.check_availability()
    if not is_available:
        print("⚠ Ollama not available. Skipping RAG test.")
        pytest.skip("Ollama required for embeddings")

    rag = RAGService(config, ollama)

    # Create test articles
    test_articles = [
        NewsArticle(
            title="Python 3.13 Released with Major Performance Improvements",
            url="http://example.com/python-release",
            source=DataSource.HACKERNEWS,
            source_name="HackerNews",
            published_at=datetime.now(),
            description="Python 3.13 brings significant performance improvements and new features.",
            tags=["python", "programming", "release"],
            priority=Priority.HIGH,
            relevance_score=0.9
        ),
        NewsArticle(
            title="New JavaScript Framework Announced",
            url="http://example.com/js-framework",
            source=DataSource.RSS,
            source_name="TechBlog",
            published_at=datetime.now(),
            description="A revolutionary new JavaScript framework promises better performance.",
            tags=["javascript", "framework", "web"],
            priority=Priority.MEDIUM,
            relevance_score=0.7
        )
    ]

    # Add articles to RAG
    doc_ids = await rag.add_articles_batch(test_articles)
    assert len(doc_ids) > 0
    print(f"✓ Added {len(doc_ids)} articles to RAG")

    # Test search
    results = await rag.search_articles(
        query="Python programming language",
        max_results=5
    )
    assert len(results) > 0
    print(f"✓ Found {len(results)} search results")

    if results:
        print(f"  Top result: {results[0].metadata.get('title')} (similarity: {results[0].similarity:.2f})")

    # Test context generation
    context, sources = await rag.generate_context(
        query="What's new in Python?",
        max_documents=2
    )
    assert context
    print(f"✓ Generated context with {len(sources)} sources")

    # Test RAG Q&A
    answer_result = await rag.answer_with_context(
        question="What are the latest updates in programming languages?"
    )
    assert answer_result["answer"]
    print(f"✓ RAG Q&A: {answer_result['answer'][:200]}...")
    print(f"  Sources used: {len(answer_result['sources'])}")

    # Get statistics
    stats = rag.get_statistics()
    print(f"✓ RAG Stats: {stats['total_documents']} documents indexed")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not available")
async def test_langgraph_orchestrator():
    """Test LangGraph orchestrator workflow."""
    print("\n=== Testing LangGraph Orchestrator ===")

    # Check Ollama availability first
    ollama = OllamaService()
    is_available = await ollama.check_availability()
    if not is_available:
        print("⚠ Ollama not available. Skipping orchestrator test.")
        pytest.skip("Ollama required for orchestrator")

    orchestrator = LangGraphOrchestrator()

    # Test simple workflow
    result = await orchestrator.run(
        user_request="Give me a summary of today's top tech news",
        preferences={"sources": ["hackernews"]}
    )

    assert "success" in result

    if result["success"]:
        print("✓ Workflow completed successfully")
        print(f"  Steps completed: {', '.join(result.get('steps_completed', []))}")
        print(f"  Execution time: {result.get('execution_time_ms', 0)}ms")

        if result.get("summary"):
            print(f"  Summary: {result['summary'][:200]}...")

        if result.get("insights"):
            print(f"  Insights: {len(result['insights'])} generated")

        if result.get("output"):
            print(f"✓ Formatted output generated ({len(result['output'])} chars)")
    else:
        print(f"⚠ Workflow failed: {result.get('error')}")

    # Get statistics
    stats = orchestrator.get_statistics()
    print(f"✓ Orchestrator stats retrieved")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_integration():
    """Test MCP tools integration with real connectors."""
    print("\n=== Testing MCP Tool Integration ===")

    server = MCPServer()

    # Test fetching real HackerNews data
    print("Testing HackerNews fetch...")
    response = await server.execute_tool(
        "fetch_hackernews",
        {"story_type": "top", "max_stories": 3}
    )

    if response.success and response.data:
        print(f"✓ Fetched {len(response.data)} HackerNews articles")
        for article in response.data[:2]:
            print(f"  - {article.get('title', 'Untitled')[:50]}...")
    else:
        print(f"⚠ HackerNews fetch failed: {response.error}")

    # Test search functionality
    print("\nTesting HackerNews search...")
    response = await server.execute_tool(
        "search_hackernews",
        {"query": "AI", "limit": 3}
    )

    if response.success and response.data:
        print(f"✓ Found {len(response.data)} articles for 'AI'")
    else:
        print(f"⚠ Search failed: {response.error}")

    # Test trending topics
    print("\nTesting trending topics...")
    response = await server.execute_tool(
        "get_trending_topics",
        {"limit": 5}
    )

    if response.success and response.data:
        print(f"✓ Trending topics: {', '.join(response.data[:5])}")
    else:
        print(f"⚠ Trending topics failed: {response.error}")


# Main test runner for manual execution
if __name__ == "__main__":
    """
    Run all AI integration tests manually.

    First, ensure Ollama is running:
        ollama serve

    Then run:
        python tests/integration/test_ai_integration.py
    """

    async def run_all_tests():
        print("=" * 60)
        print("RUNNING AI INTEGRATION TESTS")
        print("=" * 60)

        # Check Ollama availability
        try:
            ollama = OllamaService()
            is_available = await ollama.check_availability()
            OLLAMA_AVAILABLE = is_available
        except:
            OLLAMA_AVAILABLE = False

        if not OLLAMA_AVAILABLE:
            print("\n⚠️  OLLAMA NOT AVAILABLE")
            print("Please start Ollama first:")
            print("  ollama serve")
            print("\nRunning tests that don't require Ollama...")

        tests = [
            ("MCP Server", test_mcp_server),
            ("MCP Tool Integration", test_mcp_tool_integration),
        ]

        if OLLAMA_AVAILABLE:
            tests.extend([
                ("Ollama Service", test_ollama_service),
                ("RAG Service", test_rag_service),
                ("LangGraph Orchestrator", test_langgraph_orchestrator),
            ])

        passed = 0
        failed = 0
        skipped = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            except pytest.skip.Exception as e:
                skipped += 1
                print(f"⏭️  {test_name} SKIPPED: {e}\n")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} FAILED: {e}\n")

        print("=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * 60)

        if not OLLAMA_AVAILABLE:
            print("\n💡 TIP: Start Ollama to run all tests:")
            print("  ollama serve")
            print("  export OLLAMA_AVAILABLE=true")
            print("  python tests/integration/test_ai_integration.py")

        return failed == 0

    # Run tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
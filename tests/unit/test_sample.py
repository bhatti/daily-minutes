"""Sample test to verify setup."""

def test_imports():
    """Test that main packages can be imported."""
    import langchain
    import langgraph
    import pydantic
    import structlog
    
    assert langchain
    assert langgraph
    assert pydantic
    assert structlog

def test_environment():
    """Test environment setup."""
    import os
    from pathlib import Path
    
    # Check project structure
    assert Path("src").exists()
    assert Path("tests").exists()
    assert Path("data").exists()

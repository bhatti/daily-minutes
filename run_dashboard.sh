#!/bin/bash
# Script to run the unified AI-powered Streamlit dashboard

echo "============================================================"
echo "Starting Daily Minutes Dashboard"
echo "============================================================"
echo ""
echo "Make sure Ollama is running:"
echo "  ollama serve"
echo ""
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "Features:"
echo "  📰 Overview - News with AI summaries and insights"
echo "  🔍 Search - Semantic search using RAG"
echo "  💬 Q&A - Ask questions about the news"
echo "  📊 Analytics - Trends and statistics"
echo "  🔧 Orchestrator - Full AI workflow"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "============================================================"
echo ""

# Set environment variables
export PYTHONPATH=.
export VERIFY_SSL=false
export DEV_MODE=true
# Run the dashboard
./venv/bin/streamlit run src/ui/streamlit_app.py

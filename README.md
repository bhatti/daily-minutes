# Daily Minutes: Learning Agentic AI Through Practice

> **An Educational Project**: Master RAG, MCP, LangGraph, ReAct, and RLHF by building an intelligent daily briefing system

---

## 🎯 What You'll Learn

This project teaches **production-ready agentic AI concepts** through a real application that generates personalized daily briefs from your emails, calendar, news, and weather.

**Core Concepts Covered:**
1. **MCP (Model Context Protocol)** - Connect AI to external data sources
2. **RAG (Retrieval-Augmented Generation)** - Give AI long-term memory
3. **LangGraph** - Orchestrate multi-agent workflows
4. **ReAct Pattern** - Teach AI to reason before acting
5. **RLHF (Reinforcement Learning from Human Feedback)** - AI learns from your preferences
6. **3-Layer Architecture** - Build scalable, maintainable AI systems

---

## 🏗️ The 3-Layer Architecture

Daily Minutes uses a clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 3: UI & PRESENTATION                  │
│                                                              │
│  What: Streamlit dashboard displays cached data             │
│  Why: Fast, responsive UI without blocking on data fetch    │
│  Files: streamlit_app.py, src/ui/components/*               │
│                                                              │
│  • Reads cached briefs, news, emails from database          │
│  • Displays TLDR, summaries, insights                       │
│  • Collects user feedback (important/skip buttons)          │
│  • No data fetching - purely presentation layer             │
└────────────────────────┬────────────────────────────────────┘
                         │ Reads from cache
                         │
┌────────────────────────▼────────────────────────────────────┐
│            LAYER 2: INTELLIGENCE & ORCHESTRATION             │
│                                                              │
│  What: AI reasoning, workflow coordination, analysis        │
│  Why: The "brain" - decides what to fetch and how to        │
│       analyze it                                             │
│  Files: src/agents/*, src/services/*                        │
│                                                              │
│  Components:                                                 │
│  • LangGraph: Orchestrates multi-step workflows             │
│  • ReAct Agent: Reasons about what data is needed           │
│  • RAG Service: Stores/retrieves context from vector DB     │
│  • Brief Scheduler: Generates daily summaries with LLM      │
│  • Ollama Service: Local LLM for text generation            │
│                                                              │
│  • Stores enriched results in database cache                │
└────────────────────────┬────────────────────────────────────┘
                         │ Calls tools
                         │
┌────────────────────────▼────────────────────────────────────┐
│              LAYER 1: DATA SOURCES (MCP)                     │
│                                                              │
│  What: Connectors to external data sources                  │
│  Why: MCP makes data discoverable by AI agents              │
│  Files: src/connectors/*, src/services/mcp_server.py        │
│                                                              │
│  Tools registered in MCP Server:                             │
│  • fetch_hackernews - Top tech stories                      │
│  • fetch_rss_feeds - News from configured feeds             │
│  • get_weather - Current weather conditions                 │
│  • get_emails - Unread emails (future)                      │
│  • get_calendar - Upcoming events (future)                  │
│                                                              │
│  Each tool has: name, description, parameters               │
│  AI can discover and call these dynamically!                │
└─────────────────────────────────────────────────────────────┘
```

### Why This Matters

**Traditional Approach (tightly coupled):**
```python
# ❌ Hard to extend, hard to test
def get_daily_brief():
    news = fetch_hackernews()  # Hardcoded
    weather = fetch_weather()  # Hardcoded
    summary = generate_summary(news, weather)  # AI can't decide what to fetch
    return summary
```

**Agentic Approach (this project):**
```python
# ✅ AI decides what to fetch based on user request
def get_daily_brief(user_request: str):
    # Layer 2: AI reasons about what's needed
    plan = react_agent.reason(user_request)
    # → "Need: news, weather, emails"

    # Layer 1: Fetch via MCP (AI-driven)
    data = {}
    for tool_name in plan.required_tools:
        data[tool_name] = mcp_server.execute_tool(tool_name)

    # Layer 2: RAG provides context from history
    context = rag_service.search(user_request)

    # Layer 2: Generate with full context
    summary = llm.generate(data, context, user_preferences)

    # Cache for Layer 3
    db.save_cache('daily_brief', summary)
    return summary
```

---

## 📚 Core Concepts Deep Dive

### 1. MCP (Model Context Protocol) - Layer 1

**Concept**: A standardized way to expose data sources as "tools" that AI can discover and use.

**Traditional Problem:**
```python
# Hard to add new sources
if source == "hackernews":
    data = fetch_hn()
elif source == "reddit":
    data = fetch_reddit()
elif source == "twitter":
    data = fetch_twitter()
# ... need to modify code for each new source
```

**MCP Solution:**
```python
# src/services/mcp_server.py
class MCPServer:
    """Tool registry - AI can discover available tools."""

    def _register_tools(self):
        # Register HackerNews
        self.tools["fetch_hackernews"] = MCPTool(
            name="fetch_hackernews",
            description="Fetch top tech stories from HackerNews",
            parameters={
                "max_stories": {"type": "integer", "default": 10}
            },
            executor=HackerNewsConnector()
        )

        # Register RSS
        self.tools["fetch_rss_feeds"] = MCPTool(
            name="fetch_rss_feeds",
            description="Fetch articles from configured RSS feeds",
            parameters={
                "max_per_feed": {"type": "integer", "default": 5}
            },
            executor=RSSConnector()
        )

        # Future: AI can discover these automatically!
        # No code changes needed to add new sources
```

**How It Works:**
1. **Registration**: Connectors register themselves as MCP tools
2. **Discovery**: AI can list all available tools
3. **Execution**: AI calls tools by name with parameters
4. **Extensibility**: Add new connector → register as tool → AI can use it!

**Real Implementation:**
- `src/services/mcp_server.py` - Tool registry (85 lines)
- `src/connectors/hackernews.py` - HackerNews connector
- `src/connectors/rss.py` - RSS feed connector
- `src/connectors/weather.py` - Weather API connector

**Try It:**
```bash
# See all registered tools
PYTHONPATH=. python -c "
from src.services.mcp_server import get_mcp_server
mcp = get_mcp_server()
print('Available tools:', list(mcp.tools.keys()))
"
```

---

### 2. RAG (Retrieval-Augmented Generation) - Layer 2

**Concept**: Store information in a vector database for semantic search. AI can "remember" important context.

**Traditional Problem:**
```python
# AI has no memory of past data
summary = llm.generate("Summarize today's news")
# → Forgets yesterday's context
# → Doesn't know user preferences
# → Can't connect related information
```

**RAG Solution:**
```python
# src/services/rag_service.py
class RAGService:
    """Semantic memory using ChromaDB."""

    async def add_document(self, content: str, metadata: dict):
        """Store document with vector embedding."""
        # Convert text to vector (semantic representation)
        embedding = await self.ollama.create_embeddings(content)

        # Store in ChromaDB
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[unique_id]
        )

    async def search(self, query: str, max_results: int = 5):
        """Semantic search - find by meaning, not keywords."""
        # Convert query to vector
        query_embedding = await self.ollama.create_embeddings(query)

        # Find similar documents (cosine similarity)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results
        )
        return results
```

**What Gets Stored:**

1. **News Articles** (for context)
   ```python
   await rag.add_document(
       content=article.title + "\n" + article.content,
       metadata={"type": "article", "source": "hackernews", "date": "2025-01-28"}
   )
   ```

2. **Action Items** (extracted from emails)
   ```python
   await rag.add_document(
       content="Complete security training by Nov 15",
       metadata={"type": "todo", "source": "email", "priority": "high"}
   )
   ```

3. **Meeting Context** (for preparation)
   ```python
   await rag.add_document(
       content="Q4 Planning Meeting - discuss budget for next quarter",
       metadata={"type": "meeting", "date": "2025-02-01"}
   )
   ```

4. **User Preferences** (learned from feedback)
   ```python
   await rag.add_document(
       content="User marked 'security' topics as important",
       metadata={"type": "preference", "category": "security"}
   )
   ```

**Example Queries:**
```python
# Semantic search finds by meaning
results = await rag.search("What do I need for tomorrow's meeting?")
# Returns: meeting notes, related emails, action items

results = await rag.search("AI safety regulations")
# Returns: articles about AI safety, policy, ethics
# (even if they don't contain exact keywords!)
```

**Real Implementation:**
- `src/services/rag_service.py` - Vector storage (150 lines)
- Uses ChromaDB for vector database
- Uses Ollama's nomic-embed-text for embeddings

**Try It:**
```bash
# Add some documents to RAG
PYTHONPATH=. python test_ai_features.py

# Then search semantically
# Open Streamlit → Q&A tab → Ask questions
```

---

### 3. LangGraph - Workflow Orchestration (Layer 2)

**Concept**: A state machine that coordinates multi-step AI workflows.

**Traditional Problem:**
```python
# Linear, inflexible workflow
news = fetch_news()
emails = fetch_emails()
summary = generate_summary(news, emails)
# What if we need conditional logic?
# What if we need parallel fetching?
# What if we need error recovery?
```

**LangGraph Solution:**
```python
# src/services/langgraph_orchestrator.py
from langgraph.graph import StateGraph, END

def create_workflow():
    """Define workflow as a graph."""
    workflow = StateGraph(WorkflowState)

    # Step 1: Analyze request
    workflow.add_node("analyze", analyze_request)

    # Step 2: Fetch data (parallel!)
    workflow.add_node("fetch_news", fetch_news_data)
    workflow.add_node("fetch_emails", fetch_email_data)
    workflow.add_node("fetch_calendar", fetch_calendar_data)

    # Step 3: Search RAG for context
    workflow.add_node("search_rag", search_context)

    # Step 4: Generate summary
    workflow.add_node("summarize", generate_summary)

    # Step 5: Extract insights
    workflow.add_node("insights", extract_insights)

    # Define execution flow
    workflow.add_edge("analyze", "fetch_news")
    workflow.add_edge("analyze", "fetch_emails")  # Parallel!
    workflow.add_edge("analyze", "fetch_calendar")  # Parallel!

    workflow.add_edge("fetch_news", "search_rag")
    workflow.add_edge("fetch_emails", "search_rag")
    workflow.add_edge("fetch_calendar", "search_rag")

    workflow.add_edge("search_rag", "summarize")
    workflow.add_edge("summarize", "insights")
    workflow.add_edge("insights", END)

    return workflow.compile()
```

**Workflow Visualization:**
```
            analyze_request
                   ↓
      ┌────────────┼────────────┐
      ↓            ↓            ↓
  fetch_news  fetch_emails  fetch_calendar
      │            │            │
      └────────────┼────────────┘
                   ↓
            search_rag (context)
                   ↓
            generate_summary
                   ↓
           extract_insights
                   ↓
               DONE!
```

**Benefits:**
- **Parallel execution**: Fetch news/emails/calendar simultaneously
- **Conditional routing**: Skip steps based on state
- **Error recovery**: Retry failed steps
- **Checkpointing**: Save/restore workflow state
- **Observability**: Track each step's execution

**Real Implementation:**
- `src/services/langgraph_orchestrator.py` - Workflow (200 lines)
- `src/services/brief_scheduler.py` - Brief generation workflow

**Try It:**
```bash
# Run the orchestrator
PYTHONPATH=. python -c "
import asyncio
from src.services.langgraph_orchestrator import get_orchestrator

async def test():
    orch = get_orchestrator()
    result = await orch.run('Give me a tech news summary')
    print(result.get('summary'))

asyncio.run(test())
"
```

---

### 4. ReAct Pattern - Reasoning + Acting (Layer 2)

**Concept**: AI thinks (reasons) → acts (calls tools) → observes results → repeats.

**Traditional Problem:**
```python
# AI just executes, doesn't reason
summary = llm.generate("Summarize the news")
# No planning, no adaptation, no learning
```

**ReAct Solution:**
```python
# src/agents/react_agent.py
class ReActAgent:
    """Agent that reasons before acting."""

    async def run(self, goal: str):
        """Execute goal using think-act-observe loop."""
        observations = []

        for step in range(self.max_steps):
            # 1. THOUGHT: Reason about what to do next
            thought = await self.llm.generate(
                f"Goal: {goal}\n"
                f"Observations so far: {observations}\n"
                f"What should I do next?"
            )
            # Example output: "I need to check for urgent emails first"

            # 2. ACTION: Decide which tool to call
            action = self._parse_action(thought)
            # Example: {"tool": "get_emails", "params": {"filter": "unread"}}

            # 3. Execute via MCP
            result = await self.mcp.execute_tool(
                action["tool"],
                action["params"]
            )

            # 4. OBSERVATION: Record what happened
            observation = f"Found {len(result)} emails, 2 marked urgent"
            observations.append(observation)

            # 5. Check if goal achieved
            if "FINAL ANSWER" in thought:
                return self._extract_answer(thought)

        return {"error": "Max steps reached"}
```

**Example Execution:**

```
User: "Generate my daily brief"

Step 1:
  THOUGHT: "I need to gather news, emails, and calendar. Start with news."
  ACTION: call_tool("fetch_hackernews", {"max_stories": 10})
  OBSERVATION: "Fetched 10 articles from HackerNews"

Step 2:
  THOUGHT: "Good, now check for urgent emails."
  ACTION: call_tool("get_emails", {"filter": "unread", "max": 10})
  OBSERVATION: "Found 5 emails, 1 marked urgent about client escalation"

Step 3:
  THOUGHT: "Urgent email mentions project deadline - check calendar for related meetings."
  ACTION: call_tool("get_calendar", {"date": "today"})
  OBSERVATION: "Found meeting at 2pm about same project"

Step 4:
  THOUGHT: "Need historical context for this project. Search RAG."
  ACTION: query_rag("project deadline context")
  OBSERVATION: "Found 3 email threads and 2 past meetings"

Step 5:
  THOUGHT: "I have all data - time to generate the brief."
  ACTION: FINAL ANSWER
  → Generate comprehensive brief with all context
```

**Real Implementation:**
- `src/agents/react_agent.py` - ReAct agent (180 lines)
- Generates daily briefs by reasoning through what data is needed

**Try It:**
```bash
# Run ReAct agent
make preload  # Generates brief using ReAct reasoning
```

---

### 5. RLHF - Learning from Feedback (Layer 2)

**Concept**: AI learns user preferences from feedback (👍/👎) and adapts prioritization over time.

**Traditional Problem:**
```python
# AI scores everything the same way
for email in emails:
    email.importance_score = ai_score(email)
# → Doesn't learn what YOU care about
# → Can't adapt to your preferences
```

**RLHF Solution:**
```python
# src/models/email.py
class ImportanceScoringMixin:
    """Learn from user feedback."""

    importance_score: float = 0.5  # AI's base score
    boost_labels: Set[str] = set()  # Keywords user marked important
    filter_labels: Set[str] = set()  # Keywords user wants to skip

    def apply_rlhf_boost(self, content_text: str) -> float:
        """Adjust score based on learned preferences."""
        adjusted = self.importance_score

        # Boost if content matches important keywords
        for label in self.boost_labels:
            if label.lower() in content_text.lower():
                adjusted += 0.1  # Higher priority!

        # Filter if content matches skip keywords
        for label in self.filter_labels:
            if label.lower() in content_text.lower():
                adjusted -= 0.2  # Lower priority!

        return max(0.0, min(1.0, adjusted))
```

**Feedback Collection (in UI):**
```python
# In Streamlit dashboard
for email in emails:
    col1, col2, col3 = st.columns([8, 1, 1])

    with col1:
        st.write(email.subject)

    with col2:
        if st.button("👍", key=f"important_{email.id}"):
            # Extract keywords from important email
            keywords = extract_keywords(email.subject + email.body)
            # Example: ["security", "urgent", "client"]

            # Add to boost labels
            user_profile.boost_labels.update(keywords)
            save_user_profile(user_profile)

    with col3:
        if st.button("👎", key=f"skip_{email.id}"):
            # Add to filter labels
            keywords = extract_keywords(email.subject)
            user_profile.filter_labels.update(keywords)
            save_user_profile(user_profile)
```

**The Learning Loop:**
```
1. Day 1: Email about "security audit" gets AI score 0.6
2. User marks it "👍 Important"
3. System extracts keywords: ["security", "audit"]
4. Adds to boost_labels: {"security", "audit", "compliance"}
5. Day 2: New email about "security compliance" arrives
6. Base AI score: 0.5
7. RLHF boost: 0.5 + 0.1 (security) + 0.1 (compliance) = 0.7
8. Email gets higher priority in daily brief!
```

**Real Implementation:**
- `src/models/email.py` - Email with RLHF scoring (50 lines)
- `src/models/news.py` - News articles with RLHF
- `src/services/brief_scheduler.py` - Applies RLHF when ranking content

**Try It:**
```bash
# See RLHF in action
# 1. Run make preload
# 2. Open Streamlit dashboard
# 3. Mark emails as important/skip
# 4. Run make preload again
# 5. See how priorities changed!
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required
Python 3.10+
Ollama (local LLM)

# Optional
Gmail API credentials
Google Calendar API credentials
```

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/daily-minutes.git
cd daily-minutes
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Ollama
# Download from https://ollama.com
ollama pull qwen2.5:7b          # Main LLM
ollama pull nomic-embed-text    # Embeddings for RAG

# 4. Configure
cp .env.example .env
# Edit .env with your settings
```

### Run Your First Brief

```bash
# 1. Fetch data and generate brief (uses all 3 layers!)
make preload

# Output:
# ✅ Fetching news from HackerNews...         (Layer 1: MCP)
# ✅ Fetching weather...                      (Layer 1: MCP)
# ✅ Analyzing articles with AI...            (Layer 2: RAG + LLM)
# ✅ Generating daily brief...                (Layer 2: ReAct + LangGraph)
# ✅ Brief saved to database                  (Cache for Layer 3)

# 2. Launch dashboard
streamlit run streamlit_app.py              # (Layer 3: UI)

# 3. Open browser to http://localhost:8501
# See your personalized brief!
```

---

## 📁 Project Structure

```
daily-minutes/
├── src/
│   ├── agents/                    # Layer 2: AI Agents
│   │   ├── base_agent.py         # Base agent with retry logic
│   │   └── react_agent.py        # ReAct reasoning pattern
│   │
│   ├── connectors/                # Layer 1: Data Sources
│   │   ├── hackernews.py         # HackerNews API (MCP tool)
│   │   ├── rss.py                # RSS feeds (MCP tool)
│   │   ├── weather.py            # Weather API (MCP tool)
│   │   ├── gmail.py              # Gmail (future MCP tool)
│   │   └── google_calendar.py    # Calendar (future MCP tool)
│   │
│   ├── services/                  # Layer 2: Core Services
│   │   ├── mcp_server.py         # MCP tool registry
│   │   ├── rag_service.py        # RAG vector storage
│   │   ├── langgraph_orchestrator.py  # Workflow coordination
│   │   ├── brief_scheduler.py    # Daily brief generator
│   │   └── ollama_service.py     # Local LLM interface
│   │
│   ├── database/                  # Persistence
│   │   └── sqlite_manager.py     # SQLite cache
│   │
│   ├── models/                    # Data Models
│   │   ├── news.py               # News with RLHF
│   │   ├── email.py              # Email with RLHF
│   │   └── calendar.py           # Calendar events
│   │
│   ├── ui/                        # Layer 3: Presentation
│   │   ├── components/
│   │   │   ├── daily_brief.py    # Brief display
│   │   │   ├── news_components.py
│   │   │   └── email_components.py
│   │   └── streamlit_app.py      # Main dashboard
│   │
│   └── core/                      # Shared utilities
│       ├── config_manager.py     # Configuration
│       └── logging.py            # Structured logging
│
├── tests/
│   ├── unit/                      # 516 passing tests!
│   └── integration/
│
├── scripts/
│   └── preload_all_data.py       # Background data refresh
│
├── Makefile                       # Common commands
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🎓 Learning Path

### Beginner: Understand the Architecture

1. **Read the 3-layer diagram** at the top
2. **Explore Layer 1** (Data Sources):
   - Open `src/services/mcp_server.py` - See tool registry
   - Open `src/connectors/hackernews.py` - See how data is fetched
   - Understand: AI discovers and calls these tools

3. **Run the system**:
   ```bash
   make preload           # Watch all 3 layers work
   streamlit run streamlit_app.py
   ```

### Intermediate: Add Your Own Data Source

**Goal**: Add GitHub notifications as a new MCP tool

1. **Create connector** (`src/connectors/github.py`):
   ```python
   class GitHubConnector:
       async def fetch_notifications(self, max_items: int = 10):
           # Fetch from GitHub API
           return notifications
   ```

2. **Register with MCP** (`src/services/mcp_server.py`):
   ```python
   self.tools["get_github_notifications"] = MCPTool(
       name="get_github_notifications",
       description="Fetch GitHub notifications",
       parameters={"max_items": {"type": "integer", "default": 10}},
       executor=GitHubConnector()
   )
   ```

3. **Update brief** (`src/services/brief_scheduler.py`):
   ```python
   # Add GitHub section to prompt
   github_data = await db.get_cache('github_data')
   if github_data:
       prompt += "**GITHUB NOTIFICATIONS:**\n"
       for notif in github_data[:5]:
           prompt += f"- {notif['title']}\n"
   ```

4. **Test it**:
   ```bash
   make preload
   # Your GitHub notifications now appear in daily brief!
   ```

### Advanced: Implement Custom ReAct Logic

1. **Study** `src/agents/react_agent.py`
2. **Modify reasoning** to prioritize different data
3. **Add custom actions** (e.g., send Slack message if urgent)
4. **Test** with different goals

---

## 🧪 Testing

```bash
# Run all tests (516 passing!)
pytest tests/unit/ -v

# Test specific components
pytest tests/unit/test_rag_service.py -v
pytest tests/unit/test_mcp_server.py -v
pytest tests/unit/test_react_agent.py -v

# Integration tests
pytest tests/integration/test_brief_quality.py -v

# Coverage report
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file

# LLM Settings
LLM_OLLAMA_BASE_URL=http://localhost:11434
LLM_OLLAMA_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.7

# News Settings
NEWS_MAX_ARTICLES=30
NEWS_RSS_FEEDS=["https://techcrunch.com/feed/", ...]

# Weather
WEATHER_DEFAULT_LOCATION=Seattle
WEATHER_API_KEY=your_openweathermap_key

# Cache (affects background refresh)
CACHE_AUTO_REFRESH_HOURS=4      # Refresh every 4 hours
CACHE_ARTICLE_CONTENT_DAYS=100  # Cache articles 100 days
```

### Adding RSS Feeds

Edit `src/core/config_manager.py`:

```python
default_sources = [
    {"type": "hackernews", "enabled": True},
    {
        "type": "rss",
        "enabled": True,
        "feeds": [
            # Tech news
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech"},

            # Market news (for TLDR bullet #3)
            {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rss", "category": "market"},

            # Add your own!
            {"name": "Your Blog", "url": "https://yourblog.com/feed/", "category": "tech"},
        ]
    }
]
```

---

## 📊 Example Output

### Daily Brief

```markdown
## ⚡ TLDR

• Critical: Client escalation from Acme Corp requires response within 24 hours
  regarding service outages affecting 50K users.

• Today 9am: Security Training Workshop - OWASP Top 10, mandatory for all
  employees by November 15th.

• Market: OpenAI's IPO clears path as it promises to stay in California,
  despite federal scrutiny over AI safety and regulation.

## 📋 Summary

Today's critical emails and calendar events highlight urgent tasks across various
departments. The Project Manager has scheduled a Q4 planning session for next week,
requiring team members to review Q3 metrics and propose their roadmap with budget
estimates. Immediate attention is also needed regarding a client escalation from
Acme Corp about service outages. On the calendar, there's a mandatory security
training workshop at 9am focusing on OWASP Top 10 vulnerabilities.

## 💡 Key Insights

1. High-priority client issue requires immediate response (Acme Corp)
2. Mandatory cybersecurity training deadline approaching (Nov 15)
3. Q4 planning requires preparation of budget estimates
4. Federal Reserve rate decisions impacting tech sector funding
5. Security vulnerabilities need attention based on OWASP guidelines
```

---

## 📚 Further Reading

### Core Concepts
- [MCP Specification](https://modelcontextprotocol.io/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [RLHF Explained](https://huggingface.co/blog/rlhf)

### Tools
- [Ollama](https://ollama.com/) - Local LLM
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

## 📄 License

MIT License


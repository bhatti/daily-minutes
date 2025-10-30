"""Daily Minutes - Walking Skeleton UI.

This is a minimal Streamlit dashboard to manually test core infrastructure:
- Configuration loading
- Logging
- Base agent execution
"""

import streamlit as st
from datetime import datetime
from pathlib import Path

from src.core.config import get_settings, reset_settings
from src.core.logging import get_logger, setup_logging
from src.agents.base_agent import BaseAgent, AgentState


# Simple test agent for demonstration
class DemoAgent(BaseAgent[dict]):
    """Demo agent to test infrastructure."""
    
    def _execute(self) -> dict:
        """Execute demo work."""
        import time
        time.sleep(0.5)  # Simulate work
        
        return {
            "message": "Demo agent executed successfully!",
            "timestamp": datetime.now().isoformat(),
            "config_model": self.settings.llm.ollama_model,
        }


# Page configuration
st.set_page_config(
    page_title="Daily Minutes - Walking Skeleton",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
if "logging_initialized" not in st.session_state:
    setup_logging(log_level="INFO", json_format=False)
    st.session_state.logging_initialized = True

logger = get_logger("streamlit_ui")

# Sidebar
with st.sidebar:
    st.title("📰 Daily Minutes")
    st.markdown("**Walking Skeleton v0.1**")
    st.markdown("---")
    
    st.subheader("🔧 Configuration")
    
    # Reload config button
    if st.button("🔄 Reload Config"):
        reset_settings()
        st.success("Configuration reloaded!")
        logger.info("configuration_reloaded")
    
    st.markdown("---")
    st.subheader("ℹ️ Info")
    st.markdown("""
    This is a **walking skeleton** to test:
    - ✅ Configuration management
    - ✅ Logging setup
    - ✅ Base agent execution
    """)

# Main content
st.title("Daily Minutes - Infrastructure Test")
st.markdown("Testing core components with TDD approach")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "⚙️ Configuration",
    "📝 Logging",
    "🤖 Agent Test"
])

# Tab 1: Overview
with tab1:
    st.header("Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="✅ Configuration",
            value="Loaded",
            delta="Working"
        )
    
    with col2:
        st.metric(
            label="📝 Logging",
            value="Active",
            delta="Structured"
        )
    
    with col3:
        st.metric(
            label="🤖 Agents",
            value="Ready",
            delta="Base class"
        )
    
    st.markdown("---")
    
    st.success("✅ All core infrastructure is working!")
    
    st.info("""
    **Next Steps:**
    1. Click through tabs to test each component
    2. Try the agent execution in the "Agent Test" tab
    3. Check configuration values in "Configuration" tab
    4. View logging output in console where you started Streamlit
    """)

# Tab 2: Configuration
with tab2:
    st.header("⚙️ Configuration")
    
    try:
        settings = get_settings()
        
        st.subheader("LLM Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Ollama URL", value=settings.llm.ollama_base_url, disabled=True)
            st.text_input("Model", value=settings.llm.ollama_model, disabled=True)
        with col2:
            st.number_input("Temperature", value=settings.llm.temperature, disabled=True)
            st.number_input("Max Retries", value=settings.llm.max_retries, disabled=True)
        
        st.markdown("---")
        
        st.subheader("News Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Top Stories Count", value=settings.news.hn_top_stories_count, disabled=True)
            st.number_input("Cache TTL (hours)", value=settings.news.cache_ttl_hours, disabled=True)
        with col2:
            st.text_area("Interests", value="\n".join(settings.news.interests), disabled=True, height=100)
        
        st.markdown("---")
        
        st.subheader("Application Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_input("Log Level", value=settings.app.log_level, disabled=True)
        with col2:
            st.checkbox("Debug Mode", value=settings.app.debug, disabled=True)
        with col3:
            st.checkbox("Dry Run", value=settings.app.dry_run, disabled=True)
        
        st.markdown("---")
        
        # Environment info
        st.subheader("Environment")
        is_prod = settings.is_production()
        env_type = "🚀 Production" if is_prod else "🔧 Development"
        st.info(f"Running in: **{env_type}** mode")
        
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        logger.error("config_load_error", error=str(e), exc_info=True)

# Tab 3: Logging
with tab3:
    st.header("📝 Logging Test")
    
    st.markdown("""
    Test the logging system by generating log messages at different levels.
    Check your console/terminal to see the structured log output.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        log_level = st.selectbox(
            "Select Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"]
        )
        
        log_message = st.text_input(
            "Log Message",
            value="Test log message from Streamlit UI"
        )
    
    with col2:
        st.markdown("### Context (Optional)")
        user_id = st.text_input("User ID", value="test_user")
        operation = st.text_input("Operation", value="ui_test")
    
    if st.button("📤 Send Log Message"):
        test_logger = get_logger("ui_test")
        
        context = {
            "user_id": user_id,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log at selected level
        if log_level == "DEBUG":
            test_logger.debug(log_message, **context)
        elif log_level == "INFO":
            test_logger.info(log_message, **context)
        elif log_level == "WARNING":
            test_logger.warning(log_message, **context)
        elif log_level == "ERROR":
            test_logger.error(log_message, **context)
        
        st.success(f"✅ {log_level} log sent! Check your console.")
    
    st.markdown("---")
    
    st.info("""
    **💡 Tip:** Logs are output to the console where you started Streamlit.
    They include:
    - Timestamp (ISO format)
    - Log level
    - Message
    - Context (user_id, operation, etc.)
    - Call site info (file, function, line)
    """)

# Tab 4: Agent Test
with tab4:
    st.header("🤖 Agent Execution Test")
    
    st.markdown("""
    Test the base agent infrastructure by running a demo agent.
    """)
    
    # Agent controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        agent_name = st.text_input("Agent Name", value="DemoAgent")
        
    with col2:
        st.markdown("### State")
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = AgentState.IDLE
        
        state_color = {
            AgentState.IDLE: "🔵",
            AgentState.RUNNING: "🟡",
            AgentState.SUCCESS: "🟢",
            AgentState.FAILED: "🔴",
        }
        
        st.info(f"{state_color.get(st.session_state.agent_state, '⚪')} {st.session_state.agent_state.value.upper()}")
    
    # Run agent button
    if st.button("▶️ Run Agent", type="primary"):
        with st.spinner("Running agent..."):
            try:
                # Create and run agent
                agent = DemoAgent(name=agent_name)
                st.session_state.agent_state = AgentState.RUNNING
                
                result = agent.run()
                
                # Update state
                st.session_state.agent_state = agent.state
                st.session_state.last_result = result
                
                # Display result
                if result.success:
                    st.success("✅ Agent executed successfully!")
                    
                    # Show result data
                    st.json(result.data)
                    
                    # Show metadata
                    with st.expander("📊 Execution Metadata"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Execution Time", f"{result.execution_time:.3f}s")
                        with col2:
                            st.metric("Timestamp", result.timestamp.strftime("%H:%M:%S"))
                        with col3:
                            st.metric("Agent", result.metadata.get("agent_name", "Unknown"))
                else:
                    st.error(f"❌ Agent failed: {result.error}")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.agent_state = AgentState.FAILED
                logger.error("agent_execution_error", error=str(e), exc_info=True)
    
    # Show agent stats
    if st.button("📊 Show Agent Stats"):
        try:
            agent = DemoAgent(name=agent_name)
            stats = agent.get_stats()
            
            st.json(stats)
            
        except Exception as e:
            st.error(f"Error getting stats: {e}")
    
    st.markdown("---")
    
    # Show last result if available
    if "last_result" in st.session_state:
        with st.expander("📜 Last Execution Result"):
            result = st.session_state.last_result
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Success:**", "✅" if result.success else "❌")
                st.write("**Execution Time:**", f"{result.execution_time:.3f}s")
            with col2:
                st.write("**Timestamp:**", result.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                
            if result.success and result.data:
                st.write("**Data:**")
                st.json(result.data)
            elif result.error:
                st.write("**Error:**", result.error)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Daily Minutes - Walking Skeleton v0.1</p>
    <p>Core Infrastructure Testing Dashboard</p>
</div>
""", unsafe_allow_html=True)

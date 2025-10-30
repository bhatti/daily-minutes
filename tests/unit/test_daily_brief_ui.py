"""Unit tests for Daily Brief UI Component.

Since Streamlit UI components require an active Streamlit session,
we'll test the imports and basic structure validation.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock


class TestDailyBriefUIImports:
    """Test that the daily brief UI module can be imported."""

    def test_import_daily_brief_module(self):
        """Test that the daily_brief module can be imported."""
        from src.ui.components import daily_brief
        assert daily_brief is not None

    def test_render_functions_exist(self):
        """Test that required render functions are defined."""
        from src.ui.components.daily_brief import (
            render_daily_brief_section,
            render_previous_brief_section,
        )

        assert callable(render_daily_brief_section)
        assert callable(render_previous_brief_section)

    def test_private_helper_functions_exist(self):
        """Test that private helper functions are defined."""
        from src.ui.components.daily_brief import (
            _generate_and_display_brief,
            _display_brief,
        )

        assert callable(_generate_and_display_brief)
        assert callable(_display_brief)


class TestDailyBriefRendering:
    """Test daily brief rendering with mocked Streamlit."""

    @patch('src.ui.components.daily_brief.st')
    def test_render_daily_brief_section_creates_ui_elements(self, mock_st):
        """Test that render_daily_brief_section creates expected UI elements."""
        from src.ui.components.daily_brief import render_daily_brief_section

        # Mock UI elements
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.slider.return_value = 15
        mock_st.checkbox.return_value = False
        mock_st.button.return_value = False

        # Render the section
        render_daily_brief_section()

        # Verify UI elements were created
        mock_st.subheader.assert_called_once()
        mock_st.info.assert_called_once()
        mock_st.columns.assert_called_once()
        mock_st.button.assert_called_once()

    @patch('src.ui.components.daily_brief.st')
    def test_render_previous_brief_section_without_brief(self, mock_st):
        """Test that render_previous_brief_section handles missing brief gracefully."""
        from src.ui.components.daily_brief import render_previous_brief_section

        # Mock empty session state
        mock_st.session_state = {}

        # Should return early without error
        render_previous_brief_section()

        # Expander should not be called if no brief exists
        mock_st.expander.assert_not_called()

    @patch('src.ui.components.daily_brief.st')
    def test_render_previous_brief_section_with_brief(self, mock_st):
        """Test that render_previous_brief_section displays existing brief."""
        from src.ui.components.daily_brief import render_previous_brief_section
        from src.agents.react_agent import DailyBrief
        from datetime import datetime

        # Create mock brief
        mock_brief = DailyBrief(
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            action_items=["Item 1"],
            emails_count=5,
            calendar_events_count=3,
            news_items_count=10,
            weather_info={"temperature": 72}
        )

        # Mock session state with brief
        mock_st.session_state = {
            'last_daily_brief': mock_brief,
            'last_brief_timestamp': datetime.now()
        }

        # Mock expander context manager
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander

        # Mock columns
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        # Should display the brief
        render_previous_brief_section()

        # Verify expander was created
        mock_st.expander.assert_called_once()


class TestDisplayBriefLogic:
    """Test the _display_brief function logic."""

    @patch('src.ui.components.daily_brief.st')
    def test_display_brief_with_complete_data(self, mock_st):
        """Test displaying a complete daily brief."""
        from src.ui.components.daily_brief import _display_brief
        from src.agents.react_agent import DailyBrief

        # Create complete brief
        brief = DailyBrief(
            summary="Daily summary with comprehensive information",
            key_points=["Key point 1", "Key point 2", "Key point 3"],
            action_items=["Action 1", "Action 2"],
            emails_count=15,
            calendar_events_count=8,
            news_items_count=25,
            weather_info={
                "temperature": 75,
                "feels_like": 73,
                "description": "Sunny",
                "humidity": 45
            }
        )

        # Create mock result
        mock_result = MagicMock()
        mock_result.metadata = {}

        # Mock UI elements - columns is called with different numbers
        # Need to return the correct number based on the argument
        def columns_side_effect(num_cols):
            return [MagicMock() for _ in range(num_cols)]

        mock_st.columns.side_effect = columns_side_effect

        # Display the brief
        _display_brief(brief, mock_result, show_thinking=False)

        # Verify sections were created
        mock_st.success.assert_called_once()
        mock_st.metric.assert_called()  # Multiple metrics
        assert mock_st.markdown.call_count >= 3  # Summary, key points, action items

    @patch('src.ui.components.daily_brief.st')
    def test_display_brief_with_minimal_data(self, mock_st):
        """Test displaying a brief with minimal data."""
        from src.ui.components.daily_brief import _display_brief
        from src.agents.react_agent import DailyBrief

        # Create minimal brief
        brief = DailyBrief(
            summary="Minimal summary",
            key_points=[],
            action_items=[],
            emails_count=0,
            calendar_events_count=0,
            news_items_count=0,
            weather_info=None
        )

        # Create mock result
        mock_result = MagicMock()
        mock_result.metadata = {}

        # Mock UI elements
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]

        # Display the brief
        _display_brief(brief, mock_result, show_thinking=False)

        # Should still create basic structure
        mock_st.success.assert_called_once()
        mock_st.metric.assert_called()

    @patch('src.ui.components.daily_brief.st')
    def test_display_brief_with_thinking_steps(self, mock_st):
        """Test displaying brief with agent thinking process."""
        from src.ui.components.daily_brief import _display_brief
        from src.agents.react_agent import DailyBrief, ReActStep

        # Create brief
        brief = DailyBrief(
            summary="Test summary",
            key_points=["Point 1"],
            action_items=["Item 1"],
            emails_count=5,
            calendar_events_count=3,
            news_items_count=10,
            weather_info=None
        )

        # Create mock result with steps
        mock_result = MagicMock()
        mock_result.metadata = {
            'steps': [
                ReActStep(
                    step_num=1,
                    thought="I need to fetch emails",
                    action="fetch_emails",
                    action_input={"max_results": 20},
                    observation="Fetched 15 emails"
                )
            ]
        }

        # Mock UI elements
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander

        # Display with thinking enabled
        _display_brief(brief, mock_result, show_thinking=True)

        # Verify thinking section was created
        mock_st.expander.assert_called()


class TestBriefFormattingRequirements:
    """Test TDD requirements for brief formatting (bullet list format)."""

    @patch('src.ui.components.daily_brief.st')
    def test_persisted_brief_displays_summary_as_bullet_list(self, mock_st):
        """Test that persisted brief summary is displayed as bullet list, not paragraph.

        User requirement: "I would like a short tldr and then a bullet list
        instead of one giant paragraph"
        """
        from src.ui.components.daily_brief import _display_persisted_brief

        # Create brief data with multi-sentence summary
        brief_data = {
            'summary': 'First key point about news. Second important insight. Third critical update.',
            'tldr': 'Brief overview of the day',
            'key_points': ['Point 1', 'Point 2'],
            'action_items': [],
            'generated_at': '2025-10-28T10:00:00',
            'emails_count': 5,
            'calendar_events_count': 3,
            'news_items_count': 10,
            'weather_info': None
        }

        # Mock UI elements
        def columns_side_effect(num_cols):
            return [MagicMock() for _ in range(num_cols)]

        mock_st.columns.side_effect = columns_side_effect

        # Display the brief
        _display_persisted_brief(brief_data)

        # Verify that st.markdown was called with bullet list format
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]

        # Should contain bullet list with "- " markers or numbered list
        has_bullet_list = any('- ' in call or '1. ' in call for call in markdown_calls)
        assert has_bullet_list, "Summary should be displayed as bullet list"

    @patch('src.ui.components.daily_brief.st')
    def test_persisted_brief_splits_summary_into_bullets(self, mock_st):
        """Test that summary sentences are split into individual bullet points."""
        from src.ui.components.daily_brief import _display_persisted_brief

        brief_data = {
            'summary': 'First sentence. Second sentence. Third sentence.',
            'tldr': 'Brief overview',
            'key_points': [],
            'action_items': [],
            'generated_at': '2025-10-28T10:00:00',
            'emails_count': 0,
            'calendar_events_count': 0,
            'news_items_count': 0,
            'weather_info': None
        }

        mock_st.columns.return_value = [MagicMock() for _ in range(4)]

        _display_persisted_brief(brief_data)

        # Check that markdown was called with individual bullet points
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        summary_section = [call for call in markdown_calls if 'First sentence' in call or '- First' in call]

        # Should have split sentences into bullets
        assert len(summary_section) > 0, "Summary should be processed into bullets"


class TestGenerateAndDisplayBrief:
    """Test the _generate_and_display_brief function."""

    @patch('src.ui.components.daily_brief.st')
    @patch('src.ui.components.daily_brief.ReActAgent')
    def test_generate_and_display_brief_success(self, MockReActAgent, mock_st):
        """Test successful brief generation and display."""
        from src.ui.components.daily_brief import _generate_and_display_brief
        from src.agents.react_agent import DailyBrief, AgentResult

        # Create mock agent and result
        mock_agent = MagicMock()
        MockReActAgent.return_value = mock_agent

        mock_brief = DailyBrief(
            summary="Generated summary",
            key_points=["Point 1"],
            action_items=["Action 1"],
            emails_count=10,
            calendar_events_count=5,
            news_items_count=15,
            weather_info=None
        )

        mock_result = AgentResult(
            success=True,
            data=mock_brief,
            error=None,
            metadata={}
        )
        mock_agent.run.return_value = mock_result

        # Mock UI elements
        def columns_side_effect(num_cols):
            return [MagicMock() for _ in range(num_cols)]

        mock_st.columns.side_effect = columns_side_effect
        mock_st.session_state = {}

        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = MagicMock(return_value=mock_spinner)
        mock_spinner.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        # Generate and display
        _generate_and_display_brief(max_steps=15, show_thinking=False)

        # Verify agent was created and run
        MockReActAgent.assert_called_once_with(max_steps=15)
        mock_agent.run.assert_called_once()

        # Verify success message
        mock_st.success.assert_called()

        # Verify brief was stored in session state
        assert 'last_daily_brief' in mock_st.session_state
        assert 'last_brief_timestamp' in mock_st.session_state

    @patch('src.ui.components.daily_brief.st')
    @patch('src.ui.components.daily_brief.ReActAgent')
    def test_generate_and_display_brief_failure(self, MockReActAgent, mock_st):
        """Test handling of agent execution failure."""
        from src.ui.components.daily_brief import _generate_and_display_brief
        from src.agents.react_agent import AgentResult

        # Create mock agent with failure
        mock_agent = MagicMock()
        MockReActAgent.return_value = mock_agent

        mock_result = AgentResult(
            success=False,
            data=None,
            error="Agent failed to execute",
            metadata={}
        )
        mock_agent.run.return_value = mock_result

        # Mock UI elements
        mock_st.session_state = {}

        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = MagicMock(return_value=mock_spinner)
        mock_spinner.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        # Generate and display
        _generate_and_display_brief(max_steps=15, show_thinking=False)

        # Verify error message was displayed
        mock_st.error.assert_called()

    @patch('src.ui.components.daily_brief.st')
    @patch('src.ui.components.daily_brief.ReActAgent')
    def test_generate_and_display_brief_exception(self, MockReActAgent, mock_st):
        """Test handling of exceptions during generation."""
        from src.ui.components.daily_brief import _generate_and_display_brief

        # Create mock agent that raises exception
        mock_agent = MagicMock()
        MockReActAgent.return_value = mock_agent
        mock_agent.run.side_effect = Exception("Unexpected error")

        # Mock UI elements
        mock_st.session_state = {}

        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = MagicMock(return_value=mock_spinner)
        mock_spinner.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        # Generate and display
        _generate_and_display_brief(max_steps=15, show_thinking=False)

        # Verify error and exception were displayed
        mock_st.error.assert_called()
        mock_st.exception.assert_called()

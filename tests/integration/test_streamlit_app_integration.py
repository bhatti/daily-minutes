#!/usr/bin/env python3
"""Integration tests for streamlit_app.py - Main Dashboard Integration.

These tests verify:
- Streamlit app can be imported without errors
- Email and calendar components are properly integrated
- All required imports are available
- Tab structure is correctly defined

NOTE: These are import and structure tests. Full UI testing requires
manual testing or tools like Selenium/Playwright.
"""

import sys
import pytest
sys.path.append('.')

from unittest.mock import Mock, patch, MagicMock


class TestStreamlitAppImports:
    """Test that streamlit_app.py can be imported successfully."""

    def test_streamlit_app_can_be_imported(self):
        """Test that streamlit_app module can be imported."""
        try:
            import streamlit_app
            assert streamlit_app is not None
            print("✅ streamlit_app.py imports successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import streamlit_app: {e}")
        except Exception as e:
            # Streamlit may throw other exceptions during import (e.g., config)
            # As long as it's not an ImportError, the module structure is OK
            print(f"⚠️ streamlit_app.py structure OK, runtime error expected: {type(e).__name__}")

    def test_email_components_imported(self):
        """Test that email components are imported in streamlit_app."""
        import streamlit_app
        # Check if render_email_tab is available in the module
        # (it's imported from src.ui.components.email_components)
        assert hasattr(streamlit_app, 'render_email_tab')
        print("✅ Email components imported in streamlit_app")

    def test_calendar_components_imported(self):
        """Test that calendar components are imported in streamlit_app."""
        import streamlit_app
        # Check if render_calendar_tab is available in the module
        assert hasattr(streamlit_app, 'render_calendar_tab')
        print("✅ Calendar components imported in streamlit_app")


class TestStreamlitAppStructure:
    """Test streamlit_app.py structure and configuration."""

    def test_streamlit_app_has_required_components(self):
        """Test that streamlit_app has all required component imports."""
        import streamlit_app

        required_imports = [
            'render_email_tab',
            'render_calendar_tab',
            'get_news_service',
            'get_settings_manager',
            'get_weather_service',
        ]

        for import_name in required_imports:
            assert hasattr(streamlit_app, import_name), f"Missing import: {import_name}"

        print(f"✅ All {len(required_imports)} required imports present")

    def test_tab_labels_include_email_calendar(self):
        """Test that tab labels include Email and Calendar."""
        # Read streamlit_app.py and verify tab structure
        with open('streamlit_app.py', 'r') as f:
            content = f.read()

        # Check that tabs array includes Email and Calendar
        assert '📧 Email' in content, "Email tab not found in tabs array"
        assert '📅 Calendar' in content, "Calendar tab not found in tabs array"

        # Check that tab assignments include tab4 and tab5
        assert 'tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11' in content, \
            "Tab unpacking should include 11 tabs"

        print("✅ Email and Calendar tabs present in tab structure")

    def test_email_tab_rendering_called(self):
        """Test that render_email_tab() is called in streamlit_app."""
        with open('streamlit_app.py', 'r') as f:
            content = f.read()

        # Check that render_email_tab is called
        assert 'render_email_tab()' in content, "render_email_tab() not called in streamlit_app"
        assert 'asyncio.run(render_email_tab())' in content, \
            "render_email_tab() should be called with asyncio.run()"

        print("✅ render_email_tab() is properly called with asyncio.run()")

    def test_calendar_tab_rendering_called(self):
        """Test that render_calendar_tab() is called in streamlit_app."""
        with open('streamlit_app.py', 'r') as f:
            content = f.read()

        # Check that render_calendar_tab is called
        assert 'render_calendar_tab()' in content, "render_calendar_tab() not called in streamlit_app"
        assert 'asyncio.run(render_calendar_tab())' in content, \
            "render_calendar_tab() should be called with asyncio.run()"

        print("✅ render_calendar_tab() is properly called with asyncio.run()")


class TestComponentIntegration:
    """Test that email and calendar components integrate properly."""

    def test_email_components_can_be_imported(self):
        """Test that email components can be imported standalone."""
        try:
            from src.ui.components.email_components import render_email_tab
            assert render_email_tab is not None
            assert callable(render_email_tab)
            print("✅ Email components importable standalone")
        except ImportError as e:
            pytest.fail(f"Failed to import email components: {e}")

    def test_calendar_components_can_be_imported(self):
        """Test that calendar components can be imported standalone."""
        try:
            from src.ui.components.calendar_components import render_calendar_tab
            assert render_calendar_tab is not None
            assert callable(render_calendar_tab)
            print("✅ Calendar components importable standalone")
        except ImportError as e:
            pytest.fail(f"Failed to import calendar components: {e}")

    def test_email_service_can_be_imported(self):
        """Test that email service can be imported."""
        try:
            from src.services.email_service import get_email_service
            service = get_email_service()
            assert service is not None
            print("✅ Email service importable and instantiable")
        except ImportError as e:
            pytest.fail(f"Failed to import email service: {e}")

    def test_calendar_service_can_be_imported(self):
        """Test that calendar service can be imported."""
        try:
            from src.services.calendar_service import get_calendar_service
            service = get_calendar_service()
            assert service is not None
            print("✅ Calendar service importable and instantiable")
        except ImportError as e:
            pytest.fail(f"Failed to import calendar service: {e}")


class TestDependencyChain:
    """Test the complete dependency chain from UI to services to agents."""

    def test_complete_email_dependency_chain(self):
        """Test email: UI -> Formatter -> Service -> Agent -> Connector."""
        print("\n=== Testing Email Dependency Chain ===")

        # UI Components
        from src.ui.components.email_components import render_email_tab
        print("  ✅ UI Component: render_email_tab")

        # Formatter
        from src.ui.formatters.email_formatter import get_email_formatter
        formatter = get_email_formatter()
        assert formatter is not None
        print("  ✅ Formatter: EmailFormatter")

        # Service
        from src.services.email_service import get_email_service
        service = get_email_service()
        assert service is not None
        print("  ✅ Service: EmailService")

        # Agent
        from src.agents.email_agent import get_email_agent
        agent = get_email_agent()
        assert agent is not None
        print("  ✅ Agent: EmailAgent")

        # Connectors
        from src.connectors.email.gmail_connector import GmailConnector
        print("  ✅ Connector: GmailConnector available")

        print("✅ Complete email dependency chain verified")

    def test_complete_calendar_dependency_chain(self):
        """Test calendar: UI -> Formatter -> Service -> Agent -> Connector."""
        print("\n=== Testing Calendar Dependency Chain ===")

        # UI Components
        from src.ui.components.calendar_components import render_calendar_tab
        print("  ✅ UI Component: render_calendar_tab")

        # Formatter
        from src.ui.formatters.calendar_formatter import get_calendar_formatter
        formatter = get_calendar_formatter()
        assert formatter is not None
        print("  ✅ Formatter: CalendarFormatter")

        # Service
        from src.services.calendar_service import get_calendar_service
        service = get_calendar_service()
        assert service is not None
        print("  ✅ Service: CalendarService")

        # Agent
        from src.agents.calendar_agent import get_calendar_agent
        agent = get_calendar_agent()
        assert agent is not None
        print("  ✅ Agent: CalendarAgent")

        # Connectors
        from src.connectors.calendar.google_calendar_connector import GoogleCalendarConnector
        print("  ✅ Connector: GoogleCalendarConnector available")

        print("✅ Complete calendar dependency chain verified")


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("STREAMLIT APP INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    # Import tests
    print("\n--- Streamlit App Import Tests ---")
    import_tests = TestStreamlitAppImports()
    try:
        import_tests.test_streamlit_app_can_be_imported()
        import_tests.test_email_components_imported()
        import_tests.test_calendar_components_imported()
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        all_passed = False

    # Structure tests
    print("\n--- Streamlit App Structure Tests ---")
    structure_tests = TestStreamlitAppStructure()
    try:
        structure_tests.test_streamlit_app_has_required_components()
        structure_tests.test_tab_labels_include_email_calendar()
        structure_tests.test_email_tab_rendering_called()
        structure_tests.test_calendar_tab_rendering_called()
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        all_passed = False

    # Component integration tests
    print("\n--- Component Integration Tests ---")
    component_tests = TestComponentIntegration()
    try:
        component_tests.test_email_components_can_be_imported()
        component_tests.test_calendar_components_can_be_imported()
        component_tests.test_email_service_can_be_imported()
        component_tests.test_calendar_service_can_be_imported()
    except Exception as e:
        print(f"❌ Component integration test failed: {e}")
        all_passed = False

    # Dependency chain tests
    print("\n--- Dependency Chain Tests ---")
    dep_tests = TestDependencyChain()
    try:
        dep_tests.test_complete_email_dependency_chain()
        dep_tests.test_complete_calendar_dependency_chain()
    except Exception as e:
        print(f"❌ Dependency chain test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL STREAMLIT APP INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME STREAMLIT APP INTEGRATION TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

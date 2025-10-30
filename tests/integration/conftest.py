"""Configuration for integration tests.

Ensures environment variables are set before modules are imported.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env first
load_dotenv()

# Override with test-specific settings BEFORE any imports
os.environ['USE_MOCK_EMAIL'] = 'true'
os.environ['USE_MOCK_CALENDAR'] = 'true'
os.environ['VERIFY_SSL'] = 'false'

# These settings prevent heavy services from loading during tests
os.environ['SCHEDULER_ENABLE_BACKGROUND_SCHEDULER'] = 'false'

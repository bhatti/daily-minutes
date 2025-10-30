# Mock Servers for E2E Testing

This directory contains mock servers that simulate external APIs for end-to-end testing without hitting real services.

## Available Mock Servers

### 1. OAuth Mock Server (Port 5001)
Simulates Google OAuth 2.0 authorization flow.

**Endpoints:**
- `GET  /oauth/authorize` - Authorization endpoint
- `POST /oauth/token` - Token exchange endpoint
- `POST /oauth/revoke` - Token revocation
- `GET  /oauth/tokeninfo` - Token information
- `GET  /health` - Health check
- `GET  /debug/codes` - Debug: List active auth codes
- `GET  /debug/tokens` - Debug: List active tokens

**Usage:**
```bash
# Terminal 1: Start server
python tests/mocks/oauth_mock_server.py

# Terminal 2: Test authorization flow
curl "http://localhost:5001/oauth/authorize?client_id=test&redirect_uri=http://localhost:8501/callback&scope=gmail.readonly&state=abc123"

# Exchange code for token
curl -X POST http://localhost:5001/oauth/token \
  -d "grant_type=authorization_code" \
  -d "code=RETURNED_CODE" \
  -d "client_id=test" \
  -d "client_secret=secret" \
  -d "redirect_uri=http://localhost:8501/callback"
```

### 2. Gmail Mock API (Port 5002)
Simulates Gmail API v1 for reading emails.

**Endpoints:**
- `GET /gmail/v1/users/me/messages` - List messages
- `GET /gmail/v1/users/me/messages/{id}` - Get message by ID
- `POST /gmail/v1/users/me/messages/{id}/modify` - Modify message labels
- `GET /gmail/v1/users/me/profile` - Get user profile
- `GET /health` - Health check
- `GET /debug/emails` - Debug: List all mock emails

**Mock Data:**
The server includes 5 test emails:
1. Urgent email from boss (IMPORTANT, UNREAD)
2. Team meeting notes
3. GitHub PR notification
4. LinkedIn social notification
5. Calendar reminder

**Usage:**
```bash
# Terminal 1: Start server
python tests/mocks/gmail_mock_server.py

# Terminal 2: List messages (requires Bearer token)
curl -H "Authorization: Bearer mock_token" \
  "http://localhost:5002/gmail/v1/users/me/messages?maxResults=5"

# Get specific message
curl -H "Authorization: Bearer mock_token" \
  "http://localhost:5002/gmail/v1/users/me/messages/msg001"

# Search messages
curl -H "Authorization: Bearer mock_token" \
  "http://localhost:5002/gmail/v1/users/me/messages?q=urgent"
```

## Installation

```bash
# Install Flask (required for mock servers)
pip install -r requirements-dev.txt

# Or install Flask directly
pip install Flask>=3.0.0
```

## Running All Mock Servers

```bash
# Start all servers in background
python tests/mocks/oauth_mock_server.py &
python tests/mocks/gmail_mock_server.py &

# Check they're running
curl http://localhost:5001/health
curl http://localhost:5002/health

# Stop all servers
kill %1 %2
# Or: killall python
```

## Integration with E2E Tests

The mock servers are designed to work with pytest E2E tests:

```python
import pytest
import subprocess
import time

@pytest.fixture(scope="module")
def mock_servers():
    """Start mock servers for E2E tests."""
    oauth_proc = subprocess.Popen(['python', 'tests/mocks/oauth_mock_server.py'])
    gmail_proc = subprocess.Popen(['python', 'tests/mocks/gmail_mock_server.py'])

    time.sleep(2)  # Wait for servers to start

    yield

    oauth_proc.terminate()
    gmail_proc.terminate()

@pytest.mark.e2e
def test_full_oauth_flow(mock_servers):
    # Test OAuth flow using http://localhost:5001
    # Test Gmail API using http://localhost:5002
    pass
```

## Debugging

Each server provides debug endpoints:

```bash
# OAuth server debug
curl http://localhost:5001/debug/codes
curl http://localhost:5001/debug/tokens

# Gmail server debug
curl http://localhost:5002/debug/emails
```

## Port Configuration

- **5001** - OAuth Mock Server
- **5002** - Gmail Mock API
- **5003** - (Reserved for Google Calendar Mock)

## Security Notes

⚠️ **These mock servers are for testing ONLY!**
- No real authentication
- No data persistence
- Accept any Bearer token
- Should NEVER be exposed to the internet
- Should ONLY run on localhost during testing

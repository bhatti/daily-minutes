# Integration Tests

End-to-end integration tests for email connectors and EmailAgent using the mock email server.

## Prerequisites

1. **Docker** must be installed and running
2. **Mock email server** must be started before running integration tests

## Quick Start

### 1. Start the Mock Email Server

```bash
# Start the MailHog mock server
python tests/mock_email_server.py start

# Verify it's running
python tests/mock_email_server.py status

# Open web UI (optional)
open http://localhost:8025
```

### 2. Generate Test Emails (Optional)

```bash
# Generate 10 test emails
python tests/mock_email_server.py generate --count 10
```

### 3. Run Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v -m integration

# Run specific test file
pytest tests/integration/test_email_integration.py -v

# Run specific test class
pytest tests/integration/test_email_integration.py::TestIMAPConnectorIntegration -v

# Run specific test
pytest tests/integration/test_email_integration.py::TestIMAPConnectorIntegration::test_connect_to_mock_server -v
```

## Test Categories

### 1. **IMAP Connector Integration** (`TestIMAPConnectorIntegration`)
Tests the IMAP connector with the real mock server:
- Connection and authentication
- Fetching emails
- Fetching multiple emails
- Marking emails as read

### 2. **EmailAgent Integration** (`TestEmailAgentIntegration`)
Tests the EmailAgent orchestration layer:
- Fetching emails via agent
- Sorting by importance
- Filtering by sender
- Filtering by subject
- Caching mechanism
- Statistics generation

### 3. **End-to-End Workflow** (`TestEndToEndWorkflow`)
Complete workflow tests from connection to statistics:
- Create connector
- Authenticate
- Send test emails
- Create agent
- Fetch and filter
- Get statistics

## Test Results

Integration tests will:
- ✅ **PASS** if the mock server is running and emails can be sent/received
- ⏭️ **SKIP** if the mock server is not running (with helpful message)
- ❌ **FAIL** if there are connectivity or logic issues

## Common Issues

### Tests are skipped
```
SKIPPED - Mock email server is not running
```

**Solution**: Start the mock server first:
```bash
python tests/mock_email_server.py start
```

### Connection refused errors
```
ConnectionRefusedError: Connection refused
```

**Solution**: Verify Docker is running and ports are not in use:
```bash
docker ps
python tests/mock_email_server.py status
```

### Emails not appearing
```
AssertionError: assert len(emails) >= 1
```

**Solution**: Check if SMTP is working:
1. Open http://localhost:8025
2. Send a test email manually
3. Verify it appears in the web UI

## Cleanup

```bash
# Stop the mock email server
python tests/mock_email_server.py stop

# Remove volumes (optional - clears all test emails)
docker volume rm tests_mock-email-data
```

## CI/CD Integration

For CI/CD pipelines, use Docker Compose:

```bash
# Start server in CI
docker-compose -f tests/docker-compose.mock-email.yml up -d

# Wait for server to be ready
sleep 5

# Run integration tests
pytest tests/integration/ -v -m integration

# Stop server
docker-compose -f tests/docker-compose.mock-email.yml down
```

## Running All Tests (Unit + Integration)

```bash
# Run unit tests only (no mock server needed)
pytest tests/unit/ -v

# Run integration tests only (mock server required)
pytest tests/integration/ -v -m integration

# Run ALL tests (unit + integration)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Data

The integration tests:
- Send real emails via SMTP to localhost:1025
- Fetch emails via IMAP from localhost:1143
- Each test is isolated (sends its own test emails)
- Tests include small delays to ensure email availability

## Performance

Integration tests are slower than unit tests because they:
- Require real server connection
- Send actual emails over network
- Include delays for email propagation

Typical runtime: **2-5 seconds** per test class (10-15 seconds total)

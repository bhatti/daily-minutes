# Mock Email Server Setup

Testing email integration with MailHog - a local SMTP/IMAP server with a web UI.

## Quick Start

### 1. Start the Mock Email Server

```bash
python tests/mock_email_server.py start
```

This will:
- Start MailHog in a Docker container
- Expose SMTP on port 1025
- Expose IMAP on port 1143
- Expose Web UI on port 8025

### 2. Generate Test Emails

```bash
# Generate 10 test emails
python tests/mock_email_server.py generate --count 10

# Generate 50 test emails
python tests/mock_email_server.py generate --count 50
```

### 3. View Emails

Open http://localhost:8025 in your browser to view all received emails.

### 4. Test IMAP Connector

```python
from src.connectors.email.imap_connector import IMAPConnector

connector = IMAPConnector(
    imap_server="localhost",
    username="any",  # MailHog accepts any credentials
    password="any",
    use_ssl=False,
    port=1143
)

await connector.authenticate()
emails = await connector.fetch_unread_emails()
```

## Server Management

### Check Status
```bash
python tests/mock_email_server.py status
```

### Stop Server
```bash
python tests/mock_email_server.py stop
```

## Server Details

| Service | Address | Purpose |
|---------|---------|---------|
| Web UI | http://localhost:8025 | View emails in browser |
| SMTP | localhost:1025 | Send test emails |
| IMAP | localhost:1143 | Read emails via IMAP |

## Integration Testing

Use the mock server for integration tests:

```python
# tests/integration/test_email_integration.py
import pytest
from src.connectors.email.imap_connector import IMAPConnector

@pytest.mark.integration
async def test_imap_with_mock_server():
    """Test IMAP connector with mock email server."""
    connector = IMAPConnector(
        imap_server="localhost",
        username="test",
        password="test",
        use_ssl=False,
        port=1143
    )

    await connector.authenticate()
    emails = await connector.fetch_unread_emails()

    assert len(emails) > 0
```

## Troubleshooting

### Server won't start
- Make sure Docker is running
- Check if ports 1025, 1143, or 8025 are already in use

### Can't connect to IMAP
- Verify server is running: `python tests/mock_email_server.py status`
- Check the Web UI is accessible: http://localhost:8025
- Use `use_ssl=False` for the mock server

### No emails in inbox
- Generate test emails first: `python tests/mock_email_server.py generate`
- Check the Web UI to confirm emails were received

## Clean Up

Stop and remove the container and volumes:

```bash
python tests/mock_email_server.py stop
docker volume rm tests_mock-email-data
```

#!/usr/bin/env python3
"""Mock Email Server Management and Test Email Generator.

Manages GreenMail Docker container and generates test emails for integration testing.

Usage:
    # Start the mock email server
    python tests/mock_email_server.py start

    # Stop the server
    python tests/mock_email_server.py stop

    # Generate test emails
    python tests/mock_email_server.py generate --count 10

    # Check server status
    python tests/mock_email_server.py status
"""

import subprocess
import sys
import time
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random
from typing import List

# GreenMail configuration
SMTP_HOST = "localhost"
SMTP_PORT = 3025
WEB_UI_URL = "http://localhost:8080"
IMAP_HOST = "localhost"
IMAP_PORT = 3143

# Enhanced test email templates with realistic scenarios
EMAIL_TEMPLATES = [
    {
        "category": "urgent",
        "subject": "URGENT: Production Database Issue - Action Required",
        "body": """Hi Team,

We've detected a critical issue with the production database.

ACTION ITEMS:
1. TODO: Review error logs in /var/log/mysql/error.log
2. TODO: Check replication status
3. TODO: Prepare rollback plan

This needs immediate attention. Please respond ASAP.

Thanks,
IT Team""",
        "sender": ("IT Admin", "it@company.com"),
        "importance": "high",
        "has_todos": True
    },
    {
        "category": "work",
        "subject": "Q4 Planning Meeting - Please Prepare",
        "body": """Team,

Our Q4 planning session is scheduled for next week.

PREPARATION NEEDED:
- TODO: Review Q3 metrics dashboard
- TODO: Prepare your team's roadmap proposals
- TODO: Submit budget estimates by Friday

Looking forward to productive discussions!

Best,
Project Manager""",
        "sender": ("Project Manager", "pm@company.com"),
        "importance": "high",
        "has_todos": True
    },
    {
        "category": "personal",
        "subject": "Team Lunch - Casual Friday",
        "body": """Hey everyone!

Let's do team lunch this Friday at 12pm.

No work talk, just good food and conversation!

Reply if you're joining.

Cheers!""",
        "sender": ("Colleague", "colleague@company.com"),
        "importance": "low",
        "has_todos": False
    },
    {
        "category": "urgent",
        "subject": "Client Escalation - Immediate Response Needed",
        "body": """URGENT,

Major client (Acme Corp) is escalating service issues.

IMMEDIATE ACTIONS:
1. TODO: Call client contact: John Smith (555-1234)
2. TODO: Prepare incident report
3. TODO: Schedule resolution meeting today

They're threatening to switch vendors. This is top priority.

- Customer Success""",
        "sender": ("Client Success", "success@company.com"),
        "importance": "high",
        "has_todos": True
    },
    {
        "category": "newsletter",
        "subject": "Weekly Tech Digest - AI & Cloud Updates",
        "body": """Your weekly tech updates:

- New AI models released
- Cloud cost optimization tips
- Security best practices

Read more at techdigest.com

Unsubscribe | Update preferences""",
        "sender": ("Tech Digest", "newsletter@techdigest.com"),
        "importance": "low",
        "has_todos": False
    },
    {
        "category": "work",
        "subject": "Code Review: Feature/user-authentication PR #245",
        "body": """Hi,

Please review my PR for the new authentication system.

Changes:
- Implemented OAuth2 flow
- Added JWT token validation
- Updated user session management

TODO: Review and approve PR #245

Tests are passing. Ready for staging deployment.

Thanks!
Developer""",
        "sender": ("Developer", "dev@company.com"),
        "importance": "medium",
        "has_todos": True
    },
    {
        "category": "work",
        "subject": "Performance Review - Self Assessment Due",
        "body": """Dear Team Member,

It's time for quarterly performance reviews.

REQUIRED ACTIONS:
- TODO: Complete self-assessment form by Oct 30
- TODO: List 3 key achievements this quarter
- TODO: Set goals for next quarter

Link: performance.company.com/review

HR Department""",
        "sender": ("HR Department", "hr@company.com"),
        "importance": "high",
        "has_todos": True
    },
    {
        "category": "spam",
        "subject": "You've won a FREE vacation! Click here!",
        "body": """Congratulations! You're our lucky winner!

Claim your FREE all-expenses-paid vacation now!

Click here: totally-legit-site.ru

(This is obviously spam for testing filters)""",
        "sender": ("Spam Bot", "spam@suspicious.ru"),
        "importance": "low",
        "has_todos": False
    },
    {
        "category": "work",
        "subject": "Security Training - Mandatory Completion",
        "body": """Important Security Notice,

Annual security training is now available.

REQUIREMENTS:
- TODO: Complete cybersecurity module (2 hours)
- TODO: Pass the final quiz (80% required)
- TODO: Submit completion certificate

Deadline: November 15th

This is mandatory for all employees.

Security Team""",
        "sender": ("Security Team", "security@company.com"),
        "importance": "high",
        "has_todos": True
    },
    {
        "category": "personal",
        "subject": "Office Supplies Order - Your Input",
        "body": """Hi team,

We're placing the monthly office supplies order.

If you need anything (pens, notepads, etc), reply by Wednesday.

No rush - just FYI.

Admin""",
        "sender": ("Office Admin", "admin@company.com"),
        "importance": "low",
        "has_todos": False
    }
]

# Legacy templates for backward compatibility
EMAIL_SUBJECTS = [template["subject"] for template in EMAIL_TEMPLATES]
EMAIL_BODIES = [template["body"] for template in EMAIL_TEMPLATES]
SENDERS = list(set(template["sender"] for template in EMAIL_TEMPLATES))


def run_command(command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    return subprocess.run(command, capture_output=capture_output, text=True)


def start_server():
    """Start the MailHog mock email server."""
    print("🚀 Starting mock email server...")

    compose_file = Path(__file__).parent / "docker-compose.mock-email.yml"

    # Check if Docker is running
    result = run_command(["docker", "info"])
    if result.returncode != 0:
        print("❌ Error: Docker is not running. Please start Docker first.")
        return 1

    # Start the container
    result = run_command([
        "docker-compose",
        "-f", str(compose_file),
        "up", "-d"
    ])

    if result.returncode != 0:
        print(f"❌ Error starting server: {result.stderr}")
        return 1

    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)

    print(f"""
✅ Mock email server started successfully!

📧 Server Details:
   - Web UI:  {WEB_UI_URL}
   - SMTP:    {SMTP_HOST}:{SMTP_PORT}
   - IMAP:    {IMAP_HOST}:{IMAP_PORT}

💡 Next steps:
   1. Open {WEB_UI_URL} in your browser to view emails
   2. Generate test emails: python tests/mock_email_server.py generate --count 10
   3. Test IMAP connector with: server="{IMAP_HOST}", port={IMAP_PORT}
    """)

    return 0


def stop_server():
    """Stop the GreenMail mock email server."""
    print("🛑 Stopping mock email server...")

    compose_file = Path(__file__).parent / "docker-compose.mock-email.yml"

    result = run_command([
        "docker-compose",
        "-f", str(compose_file),
        "down"
    ])

    if result.returncode == 0:
        print("✅ Mock email server stopped successfully!")
        return 0
    else:
        print(f"❌ Error stopping server: {result.stderr}")
        return 1


def check_status():
    """Check if the mock email server is running."""
    print("🔍 Checking server status...\n")

    compose_file = Path(__file__).parent / "docker-compose.mock-email.yml"

    result = run_command([
        "docker-compose",
        "-f", str(compose_file),
        "ps"
    ])

    print(result.stdout)

    # Try to connect to web UI
    try:
        import urllib.request
        urllib.request.urlopen(WEB_UI_URL, timeout=2)
        print(f"✅ Server is running! Web UI: {WEB_UI_URL}")
        return 0
    except Exception:
        print("❌ Server is not responding")
        return 1


def generate_test_emails(count: int = 10, recipient: str = "test@localhost"):
    """Generate realistic test emails with categories, TODOs, and importance levels."""
    print(f"📨 Generating {count} realistic test emails...")

    # Track generated email stats
    stats = {"urgent": 0, "work": 0, "personal": 0, "newsletter": 0, "spam": 0, "with_todos": 0}

    try:
        # Connect to SMTP server
        smtp = smtplib.SMTP(SMTP_HOST, SMTP_PORT)

        for i in range(count):
            # Use enhanced templates, cycling through them
            template = EMAIL_TEMPLATES[i % len(EMAIL_TEMPLATES)]

            # Extract template data
            subject = template["subject"]
            body = template["body"]
            sender_name, sender_email = template["sender"]
            category = template["category"]
            importance = template["importance"]
            has_todos = template["has_todos"]

            # Update stats
            stats[category] = stats.get(category, 0) + 1
            if has_todos:
                stats["with_todos"] += 1

            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{sender_name} <{sender_email}>"
            msg['To'] = recipient
            msg['Subject'] = subject

            # Add timestamp variation (some older emails)
            hours_ago = random.randint(0, 48)
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            msg['Date'] = timestamp.strftime("%a, %d %b %Y %H:%M:%S +0000")

            # Add importance headers
            if importance == "high":
                msg['X-Priority'] = '1'
                msg['Importance'] = 'high'
            elif importance == "low":
                msg['X-Priority'] = '5'
                msg['Importance'] = 'low'
            else:
                msg['X-Priority'] = '3'
                msg['Importance'] = 'normal'

            # Add custom headers for testing
            msg['X-Category'] = category
            if has_todos:
                msg['X-Has-Action-Items'] = 'true'

            # Add body
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            smtp.send_message(msg)

            print(f"  ✓ [{category:>10}] {subject[:45]}...")

        smtp.quit()

        print(f"""
✅ Successfully generated {count} test emails!

📧 View emails at: {WEB_UI_URL}
🔌 Test IMAP connector with:
   - Server: {IMAP_HOST}
   - Port: {IMAP_PORT}
   - Username: (any)
   - Password: (any)
        """)

        return 0

    except Exception as e:
        print(f"❌ Error generating emails: {e}")
        print("💡 Make sure the mock email server is running: python tests/mock_email_server.py start")
        return 1


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mock Email Server Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/mock_email_server.py start
  python tests/mock_email_server.py generate --count 20
  python tests/mock_email_server.py status
  python tests/mock_email_server.py stop
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Start command
    subparsers.add_parser('start', help='Start the mock email server')

    # Stop command
    subparsers.add_parser('stop', help='Stop the mock email server')

    # Status command
    subparsers.add_parser('status', help='Check server status')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate test emails')
    generate_parser.add_argument(
        '--count', '-c',
        type=int,
        default=10,
        help='Number of test emails to generate (default: 10)'
    )
    generate_parser.add_argument(
        '--recipient', '-r',
        type=str,
        default='test@localhost',
        help='Recipient email address (default: test@localhost)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'start':
        return start_server()
    elif args.command == 'stop':
        return stop_server()
    elif args.command == 'status':
        return check_status()
    elif args.command == 'generate':
        return generate_test_emails(args.count, args.recipient)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

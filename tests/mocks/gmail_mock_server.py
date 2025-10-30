"""Mock Gmail API for testing.

This mock server simulates Gmail API v1 endpoints for reading emails.
Use this for E2E testing without hitting real Gmail APIs.

Usage:
    # Start the server
    python tests/mocks/gmail_mock_server.py

    # In your tests, point to http://localhost:5002
"""

from flask import Flask, request, jsonify
import base64
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Mock emails database
MOCK_EMAILS = [
    {
        'id': 'msg001',
        'threadId': 'thread001',
        'labelIds': ['UNREAD', 'INBOX', 'IMPORTANT'],
        'snippet': 'Important: Please review the quarterly report by EOD today...',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'boss@company.com'},
                {'name': 'To', 'value': 'user@gmail.com'},
                {'name': 'Subject', 'value': 'Q4 Report Review - URGENT'},
                {'name': 'Date', 'value': 'Mon, 27 Oct 2025 10:00:00 -0700'}
            ],
            'body': {
                'data': base64.b64encode(
                    b'Please review the attached Q4 report by EOD today. '
                    b'We need your feedback before the board meeting tomorrow.'
                ).decode()
            }
        },
        'internalDate': str(int((datetime.utcnow() - timedelta(hours=2)).timestamp() * 1000))
    },
    {
        'id': 'msg002',
        'threadId': 'thread002',
        'labelIds': ['INBOX'],
        'snippet': 'Team meeting notes from yesterday meeting...',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'teammate@company.com'},
                {'name': 'To', 'value': 'user@gmail.com'},
                {'name': 'Subject', 'value': 'Meeting Notes - Oct 26'},
                {'name': 'Date', 'value': 'Mon, 27 Oct 2025 09:00:00 -0700'}
            ],
            'body': {
                'data': base64.b64encode(
                    b'Here are the notes from yesterday\'s team meeting. '
                    b'Action items: 1) Review PRs, 2) Update documentation.'
                ).decode()
            }
        },
        'internalDate': str(int((datetime.utcnow() - timedelta(hours=3)).timestamp() * 1000))
    },
    {
        'id': 'msg003',
        'threadId': 'thread003',
        'labelIds': ['UNREAD', 'INBOX', 'CATEGORY_UPDATES'],
        'snippet': 'GitHub notification: PR #123 approved...',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'notifications@github.com'},
                {'name': 'To', 'value': 'user@gmail.com'},
                {'name': 'Subject', 'value': '[daily-minutes] Pull Request #123 approved'},
                {'name': 'Date', 'value': 'Mon, 27 Oct 2025 08:30:00 -0700'}
            ],
            'body': {
                'data': base64.b64encode(
                    b'Your pull request "Add encrypted credentials" has been approved by reviewer1. '
                    b'Ready to merge!'
                ).decode()
            }
        },
        'internalDate': str(int((datetime.utcnow() - timedelta(hours=4)).timestamp() * 1000))
    },
    {
        'id': 'msg004',
        'threadId': 'thread004',
        'labelIds': ['INBOX', 'CATEGORY_SOCIAL'],
        'snippet': 'LinkedIn: You have 5 new connection requests...',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'noreply@linkedin.com'},
                {'name': 'To', 'value': 'user@gmail.com'},
                {'name': 'Subject', 'value': 'You have 5 new connection requests'},
                {'name': 'Date', 'value': 'Sun, 26 Oct 2025 18:00:00 -0700'}
            ],
            'body': {
                'data': base64.b64encode(
                    b'People are connecting with you on LinkedIn. '
                    b'View and respond to your connection requests.'
                ).decode()
            }
        },
        'internalDate': str(int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000))
    },
    {
        'id': 'msg005',
        'threadId': 'thread005',
        'labelIds': ['UNREAD', 'INBOX'],
        'snippet': 'Calendar reminder: 1:1 with Manager tomorrow at 2pm...',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'calendar-notification@google.com'},
                {'name': 'To', 'value': 'user@gmail.com'},
                {'name': 'Subject', 'value': 'Reminder: 1:1 with Manager'},
                {'name': 'Date', 'value': 'Mon, 27 Oct 2025 07:00:00 -0700'}
            ],
            'body': {
                'data': base64.b64encode(
                    b'This is a reminder that you have a meeting scheduled: '
                    b'1:1 with Manager tomorrow at 2:00 PM. Location: Conference Room B.'
                ).decode()
            }
        },
        'internalDate': str(int((datetime.utcnow() - timedelta(hours=5)).timestamp() * 1000))
    }
]


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'gmail_mock_api'})


@app.route('/gmail/v1/users/<user_id>/messages')
def list_messages(user_id):
    """List messages endpoint.

    Simulates: https://gmail.googleapis.com/gmail/v1/users/me/messages

    Query params:
        maxResults: Maximum number of messages to return
        q: Query string (search filter)
        labelIds: Label IDs to filter by
    """
    # Check authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': {'message': 'Unauthorized', 'code': 401}}), 401

    # Parse query parameters
    max_results = int(request.args.get('maxResults', 10))
    query = request.args.get('q', '')
    label_ids = request.args.get('labelIds', '')

    messages = MOCK_EMAILS.copy()

    # Filter by label if specified
    if label_ids:
        labels = label_ids.split(',')
        messages = [m for m in messages if any(l in m['labelIds'] for l in labels)]

    # Filter by query if specified
    if query:
        query_lower = query.lower()
        filtered = []
        for msg in messages:
            # Search in subject
            subject_header = next(
                (h for h in msg['payload']['headers'] if h['name'] == 'Subject'),
                None
            )
            if subject_header and query_lower in subject_header['value'].lower():
                filtered.append(msg)
                continue

            # Search in snippet
            if query_lower in msg['snippet'].lower():
                filtered.append(msg)

        messages = filtered

    # Limit results
    messages = messages[:max_results]

    print(f"[Gmail Mock] Returning {len(messages)} messages for user: {user_id}")

    return jsonify({
        'messages': [{'id': m['id'], 'threadId': m['threadId']} for m in messages],
        'resultSizeEstimate': len(messages)
    })


@app.route('/gmail/v1/users/<user_id>/messages/<message_id>')
def get_message(user_id, message_id):
    """Get message by ID.

    Simulates: https://gmail.googleapis.com/gmail/v1/users/me/messages/{id}

    Query params:
        format: FULL, METADATA, MINIMAL, or RAW
    """
    # Check authorization
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': {'message': 'Unauthorized', 'code': 401}}), 401

    # Find message
    for msg in MOCK_EMAILS:
        if msg['id'] == message_id:
            print(f"[Gmail Mock] Returning message: {message_id}")
            return jsonify(msg)

    return jsonify({'error': {'message': 'Message not found', 'code': 404}}), 404


@app.route('/gmail/v1/users/<user_id>/messages/<message_id>/modify', methods=['POST'])
def modify_message(user_id, message_id):
    """Modify message labels.

    Simulates: https://gmail.googleapis.com/gmail/v1/users/me/messages/{id}/modify
    """
    # Check authorization
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': {'message': 'Unauthorized', 'code': 401}}), 401

    data = request.get_json()
    add_label_ids = data.get('addLabelIds', [])
    remove_label_ids = data.get('removeLabelIds', [])

    # Find and modify message
    for msg in MOCK_EMAILS:
        if msg['id'] == message_id:
            # Add labels
            for label in add_label_ids:
                if label not in msg['labelIds']:
                    msg['labelIds'].append(label)

            # Remove labels
            for label in remove_label_ids:
                if label in msg['labelIds']:
                    msg['labelIds'].remove(label)

            print(f"[Gmail Mock] Modified message {message_id}: +{add_label_ids}, -{remove_label_ids}")
            return jsonify(msg)

    return jsonify({'error': {'message': 'Message not found', 'code': 404}}), 404


@app.route('/gmail/v1/users/<user_id>/profile')
def get_profile(user_id):
    """Get user profile.

    Simulates: https://gmail.googleapis.com/gmail/v1/users/me/profile
    """
    # Check authorization
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': {'message': 'Unauthorized', 'code': 401}}), 401

    return jsonify({
        'emailAddress': f'{user_id}@gmail.com',
        'messagesTotal': len(MOCK_EMAILS),
        'threadsTotal': len(set(m['threadId'] for m in MOCK_EMAILS)),
        'historyId': '12345'
    })


@app.route('/debug/emails')
def debug_emails():
    """Debug endpoint to see all mock emails."""
    return jsonify({
        'total_emails': len(MOCK_EMAILS),
        'emails': [
            {
                'id': m['id'],
                'subject': next(
                    h['value'] for h in m['payload']['headers'] if h['name'] == 'Subject'
                ),
                'from': next(
                    h['value'] for h in m['payload']['headers'] if h['name'] == 'From'
                ),
                'labels': m['labelIds']
            }
            for m in MOCK_EMAILS
        ]
    })


if __name__ == '__main__':
    print("=" * 70)
    print("📧 Mock Gmail API Server Starting")
    print("=" * 70)
    print()
    print("📍 List messages:    http://localhost:5002/gmail/v1/users/me/messages")
    print("📍 Get message:      http://localhost:5002/gmail/v1/users/me/messages/{id}")
    print("📍 Get profile:      http://localhost:5002/gmail/v1/users/me/profile")
    print()
    print("🔍 Debug endpoint:   http://localhost:5002/debug/emails")
    print("✅ Health check:     http://localhost:5002/health")
    print()
    print(f"📊 Mock data: {len(MOCK_EMAILS)} test emails available")
    print()
    print("=" * 70)
    print()

    app.run(port=5002, debug=True, use_reloader=False)

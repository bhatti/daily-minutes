"""Mock OAuth server for testing OAuth flows.

This mock server simulates Google OAuth 2.0 authorization and token endpoints.
Use this for E2E testing without hitting real Google APIs.

Usage:
    # Start the server
    python tests/mocks/oauth_mock_server.py

    # In your tests, point to http://localhost:5001
"""

from flask import Flask, request, jsonify, redirect
import secrets
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# In-memory storage for auth codes and tokens
auth_codes = {}
tokens = {}


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'oauth_mock_server'})


@app.route('/oauth/authorize')
def authorize():
    """Simulate OAuth authorization endpoint.

    Simulates: https://accounts.google.com/o/oauth2/v2/auth

    Query params:
        client_id: OAuth client ID
        redirect_uri: Where to redirect after auth
        scope: Requested scopes
        state: CSRF token
        response_type: Should be 'code'
    """
    client_id = request.args.get('client_id')
    redirect_uri = request.args.get('redirect_uri')
    scope = request.args.get('scope')
    state = request.args.get('state')
    response_type = request.args.get('response_type', 'code')

    if not all([client_id, redirect_uri, scope]):
        return jsonify({'error': 'invalid_request'}), 400

    # Generate authorization code
    auth_code = f"mock_auth_{secrets.token_urlsafe(32)}"

    auth_codes[auth_code] = {
        'client_id': client_id,
        'scope': scope,
        'created_at': datetime.utcnow().isoformat(),
        'expires_at': (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    }

    print(f"[OAuth Mock] Generated auth code: {auth_code[:20]}... for client: {client_id}")

    # Redirect back with code
    separator = '&' if '?' in redirect_uri else '?'
    redirect_url = f"{redirect_uri}{separator}code={auth_code}"
    if state:
        redirect_url += f"&state={state}"

    return redirect(redirect_url, code=302)


@app.route('/oauth/token', methods=['POST'])
def token():
    """Simulate OAuth token endpoint.

    Simulates: https://oauth2.googleapis.com/token

    Supports two grant types:
    1. authorization_code - Exchange auth code for tokens
    2. refresh_token - Refresh access token
    """
    grant_type = request.form.get('grant_type')

    if grant_type == 'authorization_code':
        code = request.form.get('code')
        client_id = request.form.get('client_id')
        client_secret = request.form.get('client_secret')
        redirect_uri = request.form.get('redirect_uri')

        if code not in auth_codes:
            print(f"[OAuth Mock] Invalid auth code: {code}")
            return jsonify({'error': 'invalid_grant'}), 400

        auth_data = auth_codes[code]

        # Check if expired
        expires_at = datetime.fromisoformat(auth_data['expires_at'])
        if datetime.utcnow() > expires_at:
            del auth_codes[code]
            return jsonify({'error': 'expired_token'}), 400

        # Generate tokens
        access_token = f"ya29.mock_{secrets.token_urlsafe(32)}"
        refresh_token = f"1//mock_{secrets.token_urlsafe(32)}"

        token_data = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': 3600,
            'scope': auth_data['scope'],
            'id_token': f"mock_id_token_{secrets.token_urlsafe(16)}"
        }

        tokens[access_token] = {
            'token_data': token_data,
            'created_at': datetime.utcnow().isoformat()
        }

        print(f"[OAuth Mock] Issued access token: {access_token[:20]}...")

        # Delete used auth code
        del auth_codes[code]

        return jsonify(token_data)

    elif grant_type == 'refresh_token':
        refresh_token = request.form.get('refresh_token')
        client_id = request.form.get('client_id')
        client_secret = request.form.get('client_secret')

        if not refresh_token:
            return jsonify({'error': 'invalid_request'}), 400

        # Generate new access token
        new_access_token = f"ya29.mock_{secrets.token_urlsafe(32)}"

        token_data = {
            'access_token': new_access_token,
            'token_type': 'Bearer',
            'expires_in': 3600
        }

        tokens[new_access_token] = {
            'token_data': token_data,
            'created_at': datetime.utcnow().isoformat()
        }

        print(f"[OAuth Mock] Refreshed token: {new_access_token[:20]}...")

        return jsonify(token_data)

    else:
        return jsonify({'error': 'unsupported_grant_type'}), 400


@app.route('/oauth/revoke', methods=['POST'])
def revoke():
    """Simulate OAuth token revocation endpoint.

    Simulates: https://oauth2.googleapis.com/revoke
    """
    token = request.form.get('token')

    if token in tokens:
        del tokens[token]
        print(f"[OAuth Mock] Revoked token: {token[:20]}...")
        return '', 200

    return jsonify({'error': 'invalid_token'}), 400


@app.route('/oauth/tokeninfo')
def tokeninfo():
    """Simulate token info endpoint.

    Simulates: https://oauth2.googleapis.com/tokeninfo
    """
    access_token = request.args.get('access_token')

    if not access_token:
        return jsonify({'error': 'invalid_request'}), 400

    if access_token not in tokens:
        return jsonify({'error': 'invalid_token'}), 400

    token_info = tokens[access_token]['token_data']

    return jsonify({
        'azp': 'mock_client_id',
        'aud': 'mock_client_id',
        'scope': token_info.get('scope', ''),
        'exp': str(int((datetime.utcnow() + timedelta(hours=1)).timestamp())),
        'expires_in': '3600',
        'access_type': 'offline'
    })


@app.route('/debug/codes')
def debug_codes():
    """Debug endpoint to see active auth codes."""
    return jsonify({
        'active_codes': len(auth_codes),
        'codes': list(auth_codes.keys())
    })


@app.route('/debug/tokens')
def debug_tokens():
    """Debug endpoint to see active tokens."""
    return jsonify({
        'active_tokens': len(tokens),
        'tokens': [t[:20] + '...' for t in tokens.keys()]
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🔐 Mock OAuth Server Starting")
    print("=" * 70)
    print()
    print("📍 Authorization endpoint: http://localhost:5001/oauth/authorize")
    print("📍 Token endpoint:         http://localhost:5001/oauth/token")
    print("📍 Revoke endpoint:        http://localhost:5001/oauth/revoke")
    print("📍 Token info endpoint:    http://localhost:5001/oauth/tokeninfo")
    print()
    print("🔍 Debug endpoints:")
    print("   - http://localhost:5001/debug/codes")
    print("   - http://localhost:5001/debug/tokens")
    print()
    print("✅ Health check:           http://localhost:5001/health")
    print()
    print("=" * 70)
    print()

    app.run(port=5001, debug=True, use_reloader=False)

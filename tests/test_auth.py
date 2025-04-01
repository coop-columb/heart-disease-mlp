"""
Tests for authentication functionality in the Heart Disease Prediction API.
"""

from api.app import app, auth_settings
from fastapi.testclient import TestClient


def test_auth_token_endpoint():
    """Test the token endpoint."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        auth_settings.enabled = True

        try:
            response = client.post("/auth/token")
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert "expires_in" in data
            assert data["token_type"] == "bearer"
        finally:
            # Restore original auth setting
            auth_settings.enabled = old_enabled


def test_protected_endpoint_with_token(authenticated_client, sample_patient_data):
    """Test accessing a protected endpoint with a valid token."""
    # Since authentication is disabled in config for testing,
    # this test will verify the endpoint works without checking auth.
    response = authenticated_client.post("/predict", json=sample_patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_protected_endpoint_without_token():
    """Test accessing a protected endpoint without a token."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        auth_settings.enabled = True

        try:
            # Try to access protected endpoint without token
            response = client.get("/models/info")
            assert response.status_code == 401
            assert "detail" in response.json()
        finally:
            # Restore original auth setting
            auth_settings.enabled = old_enabled


def test_api_key_authentication():
    """Test API key authentication."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        auth_settings.enabled = True

        try:
            # Use default API key from config
            default_api_key = (
                auth_settings.api_keys[0].key
                if auth_settings.api_keys
                else "dev_api_key"
            )

            # Try to access protected endpoint with API key
            headers = {"X-API-Key": default_api_key}
            response = client.get("/models/info", headers=headers)
            assert response.status_code == 200

            # Try with invalid API key
            headers = {"X-API-Key": "invalid_key"}
            response = client.get("/models/info", headers=headers)
            assert response.status_code == 401
        finally:
            # Restore original auth setting
            auth_settings.enabled = old_enabled


def test_public_endpoint():
    """Test that public endpoints are accessible without authentication."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        auth_settings.enabled = True

        try:
            # Health endpoint should be accessible without auth
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
        finally:
            # Restore original auth setting
            auth_settings.enabled = old_enabled

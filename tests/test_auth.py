"""
Tests for authentication functionality in the Heart Disease Prediction API.
"""

from fastapi.testclient import TestClient

from src.heart_api.core import auth_settings
from src.heart_api.main import app


def test_auth_token_endpoint():
    """Test the token endpoint."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        old_includes_expires = auth_settings.token_includes_expires_in
        auth_settings.enabled = True
        auth_settings.token_includes_expires_in = True

        try:
            response = client.post("/auth/token")
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert "expires_in" in data
            assert data["token_type"] == "bearer"
            # Check expires_in is in seconds (60 minutes * 60 seconds)
            # Check expires_in is in seconds (60 minutes * 60 seconds)
            assert data["expires_in"] == auth_settings.token_expire_minutes * 60
        finally:
            # Restore original auth settings
            auth_settings.enabled = old_enabled
            auth_settings.token_includes_expires_in = old_includes_expires


def test_protected_endpoint_with_token(sample_patient_data):
    """Test accessing a protected endpoint with a valid token."""
    with TestClient(app) as client:
        # Ensure auth is enabled for this test
        old_enabled = auth_settings.enabled
        auth_settings.enabled = True

        try:
            # First, get a token
            token_response = client.post("/auth/token")
            assert token_response.status_code == 200
            token_data = token_response.json()
            access_token = token_data["access_token"]

            # Now use the token to access a protected endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            response = client.post(
                "/predict", json=sample_patient_data, headers=headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
        finally:
            # Restore original auth setting
            auth_settings.enabled = old_enabled


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
            # Get correct API key from config
            # Use the known key from config
            api_key = "test_api_key_1"  # From config.dev.yaml

            # Make sure the key is in auth_settings
            assert (
                api_key in auth_settings.api_keys
            ), f"API key '{api_key}' not found in config"

            # Try to access protected endpoint with API key
            headers = {"X-API-Key": api_key}
            response = client.get("/models/info", headers=headers)
            assert response.status_code == 200

            # Try with dev API key
            dev_key = "dev_api_key"  # From config.dev.yaml
            assert (
                dev_key in auth_settings.api_keys
            ), f"API key '{dev_key}' not found in config"
            headers = {"X-API-Key": dev_key}
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

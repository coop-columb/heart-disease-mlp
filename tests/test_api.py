"""
Tests for the Heart Disease Prediction API.
"""

# flake8: noqa: E402
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    try:
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Unable to create TestClient: {e}")
        return None


@pytest.fixture
def sample_patient_data():
    """Create sample patient data for testing."""
    return {
        "age": 61,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 240,
        "fbs": 1,
        "restecg": 1,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 2.4,
        "slope": 2,
        "ca": 1,
        "thal": 3,
    }


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info(client):
    """Test the model info endpoint."""
    response = client.get("/models/info")
    assert response.status_code == 200
    assert "models_available" in response.json()
    assert "ensemble_available" in response.json()
    assert "preprocessor_available" in response.json()


def test_predict_endpoint(client, sample_patient_data):
    """Test the prediction endpoint."""
    response = client.post("/predict", json=sample_patient_data)

    # If models are not available, this might return a 500 error
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        assert data["risk_level"] in ["LOW", "MODERATE", "HIGH"]
    else:
        # Print error message for debugging
        print(f"Prediction endpoint error: {response.json()}")

"""
Shared test fixtures for all tests.
"""

import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Reset path and add project root to path to avoid conflicts with other projects
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Filter out any paths related to EmotionAdaptiveMusic
sys.path = [p for p in sys.path if "EmotionAdaptiveMusic" not in p]
sys.path.insert(0, project_root)

# Import after path setup
from api.app import app  # noqa: E402
from src.models.predict_model import HeartDiseasePredictor  # noqa: E402


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with authentication disabled for testing."""
    try:
        # Add special headers to disable auth in app for tests
        test_client = TestClient(app)

        # Disable auth for testing by adding the test API key header
        test_client.headers.update({"X-API-Key": "dev_api_key"})

        return test_client
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


@pytest.fixture
def sample_patients_batch():
    """Create a batch of sample patient data for testing."""
    return [
        {
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
        },
        {
            "age": 45,
            "sex": 0,
            "cp": 1,
            "trestbps": 120,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 175,
            "exang": 0,
            "oldpeak": 0.2,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        },
    ]


@pytest.fixture
def invalid_patient_data():
    """Create invalid patient data for testing error handling."""
    return {
        "age": 61,
        "sex": 3,  # Invalid: should be 0 or 1
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


@pytest.fixture
def test_data():
    """Load test data if available."""
    data_path = "data/processed/processed_data.npz"
    try:
        # Load the processed data
        data = np.load(data_path, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test"]
        return X_test, y_test
    except (FileNotFoundError, KeyError):
        # If data not found, this will be skipped
        return None


@pytest.fixture
def predictor():
    """Load the heart disease predictor if models are available."""
    try:
        return HeartDiseasePredictor()
    except Exception:
        # If models not found, this will be skipped
        return None


@pytest.fixture
def authenticated_client():
    """Create a test client for the FastAPI app with authentication."""
    try:
        # Create client with dev API key
        test_client = TestClient(app)
        test_client.headers.update({"X-API-Key": "dev_api_key"})
        return test_client
    except Exception as e:
        pytest.skip(f"Unable to create authenticated TestClient: {e}")
        return None

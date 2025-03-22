"""
Shared test fixtures for all tests.
"""
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import after path setup
from api.app import app  # noqa: E402
from src.models.predict_model import HeartDiseasePredictor  # noqa: E402


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


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

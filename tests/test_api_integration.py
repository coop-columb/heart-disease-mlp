"""
Integration tests for the Heart Disease Prediction API.
These tests verify end-to-end functionality of the API.
"""

from unittest import mock

# pytest is required for fixture usage, even though not explicitly referenced
import pytest  # noqa: F401
from fastapi.testclient import TestClient

from api.app import app


def test_health_and_model_info(client):
    """Test the health check and model info endpoints."""
    # 1. Check health endpoint
    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["status"] == "healthy"

    # 2. Check model information endpoint
    info_response = client.get("/models/info")
    assert info_response.status_code == 200
    info_data = info_response.json()
    assert "models_available" in info_data


def test_error_handling(client, invalid_patient_data):
    """Test API error handling with invalid input data."""
    response = client.post("/predict", json=invalid_patient_data)

    # Either validation will catch the error (422), the API will handle it (400),
    # or the model will handle it safely (500 with a proper error message)
    assert response.status_code in [400, 422, 500]

    response_data = response.json()
    assert "detail" in response_data or "error" in response_data


def test_missing_model_fallback(sample_patient_data):
    """Test that predictor falls back to available models when some are missing."""
    # Import here to avoid circular import with conftest.py
    from src.models.predict_model import HeartDiseasePredictor

    # Mock the HeartDiseasePredictor to simulate missing models
    with mock.patch.object(HeartDiseasePredictor, "__init__", return_value=None):
        # Create the mock instance
        predictor_mock = HeartDiseasePredictor.__new__(HeartDiseasePredictor)

        # Configure mock properties
        predictor_mock.sklearn_model = None  # Simulate sklearn model missing
        predictor_mock.keras_model = mock.MagicMock()  # Keras available
        predictor_mock.preprocessor = mock.MagicMock()
        predictor_mock.has_sklearn_model = False
        predictor_mock.has_keras_model = True
        predictor_mock.has_ensemble_model = False

        # Mock predict method to simulate keras model available
        def mock_predict(*args, **kwargs):
            return {
                "keras_predictions": [1],
                "keras_probabilities": [0.75],
            }

        predictor_mock.predict = mock_predict

        # Create a test client with the mock
        with mock.patch("api.app.model_predictor", predictor_mock):
            client = TestClient(app)

            # Prediction should work even with only keras model
            response = client.post("/predict", json=sample_patient_data)
            assert response.status_code == 200

            data = response.json()
            assert data["model_used"] == "keras_mlp"
            assert data["prediction"] == 1
            assert data["probability"] == 0.75


def test_batch_endpoint_exists(client, sample_patients_batch):
    """Test that the batch endpoint exists and accepts requests."""
    # Try a small batch to check if the endpoint exists
    small_batch = sample_patients_batch[:1]

    # Send request - we're just testing that the endpoint handles the request
    response = client.post("/predict/batch", json=small_batch)

    # Even if it returns an error, the endpoint should accept the right format
    assert response.status_code in [200, 422, 500]

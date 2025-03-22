"""
Integration tests for the Heart Disease Prediction API.
These tests verify end-to-end functionality of the API.
"""
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api.app import app


def test_complete_api_flow(client, sample_patients_batch):
    """Test the complete API flow: health check, model info, and prediction."""
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

    # Store which models are available for later prediction tests
    sklearn_available = info_data["models_available"].get("sklearn_mlp", False)
    keras_available = info_data["models_available"].get("keras_mlp", False)
    ensemble_available = info_data.get("ensemble_available", False)

    # 3. Make a prediction for a single patient (first in batch)
    patient_data = sample_patients_batch[0]
    prediction_response = client.post("/predict", json=patient_data)

    # If any models are available, prediction should work
    if sklearn_available or keras_available:
        assert prediction_response.status_code == 200
        prediction_data = prediction_response.json()
        assert "prediction" in prediction_data
        assert "probability" in prediction_data
        assert "risk_level" in prediction_data
        assert isinstance(prediction_data["prediction"], int)
        assert 0 <= prediction_data["probability"] <= 1
        assert prediction_data["risk_level"] in ["LOW", "MODERATE", "HIGH"]

        # 4. Test batch prediction if single prediction works
        batch_response = client.post("/predict/batch", json=sample_patients_batch)
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert "predictions" in batch_data
        assert len(batch_data["predictions"]) == len(sample_patients_batch)

        # 5. Check that specific model endpoints work if available
        patient_data = sample_patients_batch[0]

        if sklearn_available:
            # Test sklearn model endpoint
            sklearn_response = client.post(
                "/predict", json=patient_data, params={"model": "sklearn"}
            )
            assert sklearn_response.status_code == 200
            sklearn_data = sklearn_response.json()
            assert "model_used" in sklearn_data
            assert sklearn_data["model_used"] == "sklearn_mlp"

        if keras_available:
            # Test keras model endpoint
            keras_response = client.post(
                "/predict", json=patient_data, params={"model": "keras"}
            )
            assert keras_response.status_code == 200
            keras_data = keras_response.json()
            assert "model_used" in keras_data
            assert keras_data["model_used"] == "keras_mlp"

        if ensemble_available:
            # Test ensemble model endpoint
            ensemble_response = client.post(
                "/predict", json=patient_data, params={"model": "ensemble"}
            )
            assert ensemble_response.status_code == 200
            ensemble_data = ensemble_response.json()
            assert "model_used" in ensemble_data
            assert ensemble_data["model_used"] == "ensemble"


def test_error_handling(client, invalid_patient_data):
    """Test API error handling with invalid input data."""
    response = client.post("/predict", json=invalid_patient_data)

    # Either validation will catch the error (422) or the API will handle it (400)
    assert response.status_code in [400, 422]

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
        predictor_mock.has_preprocessor = True

        # Mock prediction methods
        predictor_mock.predict_keras = mock.MagicMock(return_value=(1, 0.75))
        predictor_mock.predict_sklearn = mock.MagicMock(
            side_effect=ValueError("Model not available")
        )
        predictor_mock.predict_ensemble = mock.MagicMock(
            side_effect=ValueError("Cannot use ensemble without all models")
        )

        # Create a test client with the mock
        with mock.patch("api.app.predictor", predictor_mock):
            client = TestClient(app)

            # Prediction should work even with only keras model
            response = client.post("/predict", json=sample_patient_data)
            assert response.status_code == 200

            data = response.json()
            assert data["model_used"] == "keras_mlp"
            assert data["prediction"] == 1
            assert data["probability"] == 0.75


def test_stress_test_batch_endpoint(client, sample_patients_batch):
    """Test the batch endpoint with a large number of patients to ensure it scales."""
    # Create a larger batch of patients by repeating the sample (100 patients)
    large_batch = sample_patients_batch * 50

    # Check if API is running and models are available
    model_info = client.get("/models/info").json()
    model_sklearn = model_info["models_available"].get("sklearn_mlp", False)
    model_keras = model_info["models_available"].get("keras_mlp", False)

    if not (model_sklearn or model_keras):
        pytest.skip("No models available for prediction")

    # Test batch prediction
    response = client.post("/predict/batch", json=large_batch)

    # If the endpoint works, we should get 200 status
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(large_batch)

    # Check just the first few predictions to avoid too many assertions
    for pred in data["predictions"][:5]:
        assert "prediction" in pred
        assert "probability" in pred
        assert "risk_level" in pred

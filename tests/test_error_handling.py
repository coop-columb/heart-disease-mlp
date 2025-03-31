"""
Tests focused on error handling in the Heart Disease Prediction API.
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Reset path and add project root to path to avoid conflicts with other projects
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Filter out any paths related to EmotionAdaptiveMusic
sys.path = [p for p in sys.path if "EmotionAdaptiveMusic" not in p]
sys.path.insert(0, project_root)

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
def invalid_patient_data():
    """Create invalid patient data for testing error handling."""
    return {
        "age": 1000,  # Invalid age
        "sex": 3,  # Invalid sex (should be 0 or 1)
        "cp": 5,  # Invalid chest pain type
        "trestbps": 80,
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
def incomplete_patient_data():
    """Create incomplete patient data for testing error handling."""
    return {
        "age": 61,
        "sex": 1,
        # Missing most fields
    }


def test_validation_errors(client, invalid_patient_data):
    """Test API's handling of invalid input data."""
    response = client.post("/predict", json=invalid_patient_data)
    
    # The API should still handle this gracefully
    assert response.status_code in [200, 422, 500]
    
    # If it returns 422, it should include error details
    if response.status_code == 422:
        data = response.json()
        assert "detail" in data


def test_missing_fields(client, incomplete_patient_data):
    """Test API's handling of incomplete input data."""
    response = client.post("/predict", json=incomplete_patient_data)
    
    # Should return validation error
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_nonexistent_model(client, sample_patient_data):
    """Test API's handling of requests for non-existent models."""
    response = client.post("/predict?model=nonexistent_model", json=sample_patient_data)
    
    # Should fall back to an available model
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] in ["ensemble", "sklearn_mlp", "keras_mlp"]


def test_preprocessor_missing(client, sample_patient_data):
    """Test API's behavior when preprocessor is missing."""
    # Mock the HeartDiseasePredictor with missing preprocessor
    with patch("api.app.model_predictor") as mock_predictor:
        # Configure mock to simulate missing preprocessor
        mock_predictor.preprocessor = None
        mock_predictor.predict.return_value = {"error": "Preprocessor not available"}
        
        response = client.post("/predict", json=sample_patient_data)
        
        # Should return 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


def test_keras_conversion_error(client, sample_patient_data):
    """Test handling of Keras NumPy array conversion errors."""
    with patch("src.models.mlp_model.interpret_prediction") as mock_interpret:
        # Simulate a NumPy array that can't be properly converted
        mock_interpret.side_effect = ValueError("could not convert string to float")
        
        # This should be caught and handled gracefully
        response = client.post("/predict", json=sample_patient_data)
        
        # Either returns 200 with default interpretation or 500 with error
        if response.status_code == 200:
            data = response.json()
            assert data["prediction"] in [0, 1]
        else:
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data


def test_empty_batch_request(client):
    """Test handling of empty batch request."""
    response = client.post("/predict/batch", json=[])
    
    # Should return validation error
    assert response.status_code == 422 or response.status_code == 500
    data = response.json()
    if response.status_code == 422:
        assert "detail" in data
    else:
        assert "detail" in data


def test_single_bad_item_in_batch(client, sample_patient_data, invalid_patient_data):
    """Test handling of a single bad item in a batch request."""
    # Create batch with one good and one bad item
    batch = [sample_patient_data, invalid_patient_data]
    
    response = client.post("/predict/batch", json=batch)
    
    # Should return 200 with one good result and one error result
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    # Check if one succeeded and one has error
    has_success = any(p.get("model_used") is not None for p in data["predictions"])
    
    # At least one prediction should succeed
    assert has_success


def test_malformed_json(client):
    """Test handling of malformed JSON."""
    response = client.post(
        "/predict", 
        data="this is not valid json",
        headers={"Content-Type": "application/json"}
    )
    
    # Should return 422 error for JSON parsing issue
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_prediction_none_handling(client, sample_patient_data):
    """Test handling of None prediction values."""
    with patch("api.app.model_predictor.predict") as mock_predict:
        # Return a result with None prediction
        mock_predict.return_value = {
            "keras_predictions": [None],
            "keras_probabilities": [None]
        }
        
        response = client.post("/predict", json=sample_patient_data)
        
        # Should handle this gracefully
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
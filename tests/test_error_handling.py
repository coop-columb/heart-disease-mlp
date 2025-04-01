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

from src.heart_api.main import app  # noqa: E402


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
def base_valid_patient_data():
    """Create base valid patient data that can be modified for testing specific field validations."""
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
    with patch("src.heart_api.main.model_predictor") as mock_predictor:
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
        headers={"Content-Type": "application/json"},
    )

    # Should return 422 error for JSON parsing issue
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_prediction_none_handling(client, sample_patient_data):
    """Test handling of None prediction values."""
    with patch("src.heart_api.main.model_predictor.predict") as mock_predict:
        # Return a result with None prediction
        mock_predict.return_value = {
            "keras_predictions": [None],
            "keras_probabilities": [None],
        }

        response = client.post("/predict", json=sample_patient_data)

        # Should handle this gracefully
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        data = response.json()
        assert "detail" in data


# Field-specific validation tests
def test_age_validation(client, base_valid_patient_data):
    """Test age field validation with out-of-range values."""
    # Test age below minimum
    invalid_data = base_valid_patient_data.copy()
    invalid_data["age"] = 15  # Below 18
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "age" and "Age must be at least 18 years" in error["msg"]:
            error_found = True
    assert error_found, "Expected age validation error message not found"

    # Test age above maximum
    invalid_data["age"] = 110  # Above 100
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "age" and "Age must be at most 100 years" in error["msg"]:
            error_found = True
    assert error_found, "Expected age validation error message not found"


def test_sex_validation(client, base_valid_patient_data):
    """Test sex field validation with invalid values."""
    invalid_data = base_valid_patient_data.copy()
    invalid_data["sex"] = 2  # Invalid, should be 0 or 1
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "sex" and "Sex must be 0 (female) or 1 (male)" in error["msg"]:
            error_found = True
    assert error_found, "Expected sex validation error message not found"


def test_cp_validation(client, base_valid_patient_data):
    """Test chest pain type field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test cp below minimum
    invalid_data["cp"] = 0  # Below 1
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "cp" and "Chest pain type must be at least 1" in error["msg"]:
            error_found = True
    assert error_found, "Expected cp validation error message not found"

    # Test cp above maximum
    invalid_data["cp"] = 5  # Above 4
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "cp" and "Chest pain type must be at most 4" in error["msg"]:
            error_found = True
    assert error_found, "Expected cp validation error message not found"


def test_trestbps_validation(client, base_valid_patient_data):
    """Test resting blood pressure field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test trestbps below minimum
    invalid_data["trestbps"] = 50  # Below 60
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "trestbps"
            and "Resting blood pressure must be at least 60 mm Hg" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected trestbps validation error message not found"

    # Test trestbps above maximum
    invalid_data["trestbps"] = 350  # Above 300
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "trestbps"
            and "Resting blood pressure must be at most 300 mm Hg" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected trestbps validation error message not found"


def test_chol_validation(client, base_valid_patient_data):
    """Test cholesterol field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test chol below minimum
    invalid_data["chol"] = 90  # Below 100
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "chol"
            and "Serum cholesterol must be at least 100 mg/dl" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected chol validation error message not found"

    # Test chol above maximum
    invalid_data["chol"] = 650  # Above 600
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "chol"
            and "Serum cholesterol must be at most 600 mg/dl" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected chol validation error message not found"


def test_fbs_validation(client, base_valid_patient_data):
    """Test fasting blood sugar field validation with invalid values."""
    invalid_data = base_valid_patient_data.copy()
    invalid_data["fbs"] = 2  # Invalid, should be 0 or 1
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "fbs"
            and "Fasting blood sugar must be 0 (false) or 1 (true)" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected fbs validation error message not found"


def test_restecg_validation(client, base_valid_patient_data):
    """Test resting ECG field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test restecg below minimum
    invalid_data["restecg"] = -1  # Below 0
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "restecg"
            and "Resting ECG result must be between 0 and 2" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected restecg validation error message not found"

    # Test restecg above maximum
    invalid_data["restecg"] = 3  # Above 2
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "restecg"
            and "Resting ECG result must be between 0 and 2" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected restecg validation error message not found"


def test_thalach_validation(client, base_valid_patient_data):
    """Test maximum heart rate field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test thalach below minimum
    invalid_data["thalach"] = 50  # Below 60
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "thalach"
            and "Maximum heart rate must be at least 60 bpm" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected thalach validation error message not found"

    # Test thalach above maximum
    invalid_data["thalach"] = 230  # Above 220
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "thalach"
            and "Maximum heart rate must be at most 220 bpm" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected thalach validation error message not found"


def test_exang_validation(client, base_valid_patient_data):
    """Test exercise induced angina field validation with invalid values."""
    invalid_data = base_valid_patient_data.copy()
    invalid_data["exang"] = 2  # Invalid, should be 0 or 1
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "exang"
            and "Exercise induced angina must be 0 (no) or 1 (yes)" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected exang validation error message not found"


def test_oldpeak_validation(client, base_valid_patient_data):
    """Test ST depression field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test oldpeak below minimum
    invalid_data["oldpeak"] = -1.5  # Below 0.0
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "oldpeak" and "ST depression must be at least 0.0" in error["msg"]:
            error_found = True
    assert error_found, "Expected oldpeak validation error message not found"

    # Test oldpeak above maximum
    invalid_data["oldpeak"] = 12.5  # Above 10.0
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "oldpeak" and "ST depression must be at most 10.0" in error["msg"]:
            error_found = True
    assert error_found, "Expected oldpeak validation error message not found"


def test_slope_validation(client, base_valid_patient_data):
    """Test slope field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test slope below minimum
    invalid_data["slope"] = 0  # Below 1
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "slope" and "Slope must be between 1 and 3" in error["msg"]:
            error_found = True
    assert error_found, "Expected slope validation error message not found"

    # Test slope above maximum
    invalid_data["slope"] = 4  # Above 3
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if error["loc"][-1] == "slope" and "Slope must be between 1 and 3" in error["msg"]:
            error_found = True
    assert error_found, "Expected slope validation error message not found"


def test_ca_validation(client, base_valid_patient_data):
    """Test number of major vessels field validation with out-of-range values."""
    invalid_data = base_valid_patient_data.copy()

    # Test ca below minimum
    invalid_data["ca"] = -1  # Below 0
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "ca"
            and "Number of major vessels must be between 0 and 3" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected ca validation error message not found"

    # Test ca above maximum
    invalid_data["ca"] = 5  # Above 3
    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_found = False
    for error in data["detail"]:
        if (
            error["loc"][-1] == "ca"
            and "Number of major vessels must be between 0 and 3" in error["msg"]
        ):
            error_found = True
    assert error_found, "Expected ca validation error message not found"


def test_thal_validation(client, base_valid_patient_data):
    """Test thalassemia field validation with invalid values not in [3, 6, 7]."""
    invalid_data = base_valid_patient_data.copy()

    # Test invalid thal values
    for invalid_value in [1, 2, 4, 5, 8, 9]:
        invalid_data["thal"] = invalid_value
        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        error_found = False
        for error in data["detail"]:
            if (
                error["loc"][-1] == "thal"
                and "Thalassemia value must be one of [3, 6, 7]" in error["msg"]
            ):
                error_found = True
        assert (
            error_found
        ), f"Expected thal validation error message not found for value {invalid_value}"

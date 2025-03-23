#!/usr/bin/env python
"""
Script to test the Heart Disease Prediction API manually.
"""

import json
import sys

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def test_models_info():
    """Test the models info endpoint."""
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Models info: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def test_predict():
    """Test the prediction endpoint."""
    sample_patient = {
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

    response = requests.post(f"{BASE_URL}/predict", json=sample_patient)
    print(f"Prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def test_batch_predict():
    """Test the batch prediction endpoint."""
    sample_batch = [
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

    response = requests.post(f"{BASE_URL}/predict/batch", json=sample_batch)
    print(f"Batch prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def main():
    """Run the tests."""
    print("Testing Heart Disease Prediction API...")

    try:
        test_health()
        test_models_info()
        test_predict()
        test_batch_predict()
        print("All tests completed successfully.")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Make sure the API server is running on http://localhost:8000")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Script to test the Heart Disease Prediction API manually.
"""

import argparse
import json
import sys
import time

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


def test_cache_endpoints():
    """Test cache-related endpoints."""
    print("=== Testing Cache Endpoints ===")

    # Get cache stats
    print("Cache Statistics:")
    response = requests.get(f"{BASE_URL}/cache/stats")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Test repeated prediction with caching
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

    print("First prediction (cache miss expected):")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=sample_patient)
    first_prediction_time = time.time() - start_time
    print(f"Time: {first_prediction_time:.4f} seconds")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    print("Second prediction (cache hit expected):")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=sample_patient)
    second_prediction_time = time.time() - start_time
    print(f"Time: {second_prediction_time:.4f} seconds")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Calculate and display speedup
    if first_prediction_time > 0:
        speedup = (
            first_prediction_time / second_prediction_time
            if second_prediction_time > 0
            else float("inf")
        )
        print(f"Cache speedup: {speedup:.2f}x")
    print()

    # Get updated cache stats to verify hit count increased
    print("Updated Cache Statistics (hits should have increased):")
    response = requests.get(f"{BASE_URL}/cache/stats")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Update cache configuration
    print("Updating Cache Configuration:")
    new_config = {"enabled": True, "max_size": 2000, "ttl": 7200}
    response = requests.post(f"{BASE_URL}/cache/config", json=new_config)
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Clear cache
    print("Clearing Cache:")
    response = requests.post(f"{BASE_URL}/cache/clear")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Get cache stats after clearing
    print("Cache Statistics After Clearing:")
    response = requests.get(f"{BASE_URL}/cache/stats")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    print("=== Cache Endpoint Tests Completed ===")


def main():
    """Run the tests."""
    parser = argparse.ArgumentParser(description="Test the Heart Disease Prediction API manually")
    parser.add_argument(
        "--test-cache-endpoints", action="store_true", help="Test cache-related endpoints"
    )
    args = parser.parse_args()

    print("Testing Heart Disease Prediction API...")

    try:
        test_health()
        test_models_info()
        test_predict()
        test_batch_predict()

        if args.test_cache_endpoints:
            test_cache_endpoints()

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

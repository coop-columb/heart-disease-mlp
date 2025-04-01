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

# Authentication options
AUTH_ENABLED = True  # Set to False to disable authentication
API_KEY = "dev_api_key"  # Default development API key


def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def test_models_info():
    """Test the models info endpoint."""
    token = get_auth_token() if AUTH_ENABLED else None
    headers = get_auth_headers(token)
    response = requests.get(f"{BASE_URL}/models/info", headers=headers)
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

    token = get_auth_token() if AUTH_ENABLED else None
    headers = get_auth_headers(token)
    response = requests.post(
        f"{BASE_URL}/predict", json=sample_patient, headers=headers
    )
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

    token = get_auth_token() if AUTH_ENABLED else None
    headers = get_auth_headers(token)
    response = requests.post(
        f"{BASE_URL}/predict/batch", json=sample_batch, headers=headers
    )
    print(f"Batch prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()


def test_cache_endpoints():
    """Test cache-related endpoints."""
    print("=== Testing Cache Endpoints ===")

    token = get_auth_token() if AUTH_ENABLED else None
    headers = get_auth_headers(token)

    # Get cache stats
    print("Cache Statistics:")
    response = requests.get(f"{BASE_URL}/cache/stats", headers=headers)
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
    response = requests.post(
        f"{BASE_URL}/predict", json=sample_patient, headers=headers
    )
    first_prediction_time = time.time() - start_time
    print(f"Time: {first_prediction_time:.4f} seconds")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    print("Second prediction (cache hit expected):")
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/predict", json=sample_patient, headers=headers
    )
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
    response = requests.get(f"{BASE_URL}/cache/stats", headers=headers)
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Update cache configuration
    print("Updating Cache Configuration:")
    new_config = {"enabled": True, "max_size": 2000, "ttl": 7200}
    response = requests.post(
        f"{BASE_URL}/cache/config", json=new_config, headers=headers
    )
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Clear cache
    print("Clearing Cache:")
    response = requests.post(f"{BASE_URL}/cache/clear", headers=headers)
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    # Get cache stats after clearing
    print("Cache Statistics After Clearing:")
    response = requests.get(f"{BASE_URL}/cache/stats", headers=headers)
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print()

    print("=== Cache Endpoint Tests Completed ===")


def get_auth_token():
    """Get an authentication token from the API."""
    if not AUTH_ENABLED:
        return None

    try:
        response = requests.post(f"{BASE_URL}/auth/token")
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            print(
                f"WARNING: Failed to get authentication token. Status code: {response.status_code}"
            )
            return None
    except Exception as e:
        print(f"WARNING: Error getting authentication token: {str(e)}")
        return None


def get_auth_headers(token=None):
    """Get authentication headers based on current settings."""
    headers = {"Content-Type": "application/json"}

    if not AUTH_ENABLED:
        return headers

    # Try token authentication first
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        # Fall back to API key authentication
        headers["X-API-Key"] = API_KEY

    return headers


def test_auth():
    """Test authentication endpoints."""
    print("=== Testing Authentication ===")

    # Test getting a token
    response = requests.post(f"{BASE_URL}/auth/token")
    print(f"Get token: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(json.dumps(token_data, indent=4))

        # Test using the token
        token = token_data.get("access_token")
        auth_headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/models/info", headers=auth_headers)
        print(f"\nUsing token to access protected endpoint: {response.status_code}")

        if response.status_code == 200:
            print("✓ Token authentication successful")
        else:
            print("✗ Token authentication failed")
            print(response.text)
    else:
        print(f"Failed to get token: {response.text}")

    # Test API key authentication
    api_key_headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/models/info", headers=api_key_headers)
    print(f"\nUsing API key to access protected endpoint: {response.status_code}")

    if response.status_code == 200:
        print("✓ API key authentication successful")
    else:
        print("✗ API key authentication failed")
        print(response.text)

    print("=== Authentication Tests Completed ===\n")


def main():
    """Run the tests."""
    global AUTH_ENABLED, API_KEY

    parser = argparse.ArgumentParser(
        description="Test the Heart Disease Prediction API manually"
    )
    parser.add_argument(
        "--test-cache-endpoints",
        action="store_true",
        help="Test cache-related endpoints",
    )
    parser.add_argument(
        "--test-auth", action="store_true", help="Test authentication endpoints"
    )
    parser.add_argument(
        "--api-key", type=str, help="API key to use for authentication", default=API_KEY
    )
    parser.add_argument(
        "--no-auth", action="store_true", help="Disable authentication for testing"
    )
    args = parser.parse_args()

    # Update authentication settings based on arguments
    if args.no_auth:
        AUTH_ENABLED = False
    if args.api_key:
        API_KEY = args.api_key

    print("Testing Heart Disease Prediction API...")
    print(f"Authentication enabled: {AUTH_ENABLED}")

    # Get auth token for tests
    # Get token for tests but don't store it in a variable
    # since we get fresh tokens for each test
    get_auth_token() if AUTH_ENABLED else None

    try:
        # Test authentication if requested
        if args.test_auth:
            test_auth()

        # Run regular tests
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

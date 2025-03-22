#!/usr/bin/env python
"""
Manual test script for the Heart Disease Prediction API.
This script will only run when explicitly invoked and is skipped by pytest.
"""

import json
import sys


def main():
    """Main function to test API endpoints."""
    import requests

    # Define base URL
    base_url = "http://localhost:8000"

    print(f"Testing Heart Disease Prediction API at {base_url}")
    print("Make sure the API server is running before using this script.")
    print("=" * 70)

    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        print("Health endpoint response:", response.status_code)
        print(response.json())
        print()

        # Test model info endpoint
        response = requests.get(f"{base_url}/models/info")
        print("Model info endpoint response:", response.status_code)
        print(json.dumps(response.json(), indent=2))
        print()

        # Test prediction endpoint
        patient_data = {
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

        response = requests.post(f"{base_url}/predict", json=patient_data)
        print("Prediction endpoint response:", response.status_code)
        print(json.dumps(response.json(), indent=2))

        return 0
    except requests.exceptions.ConnectionError:
        print("Error: Failed to connect to API server. Make sure it's running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

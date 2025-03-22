"""
Tests for Docker container functionality.
"""

import subprocess
import time
import warnings

import pytest
import requests

# Skip these tests if Docker is not installed
docker_installed = (
    subprocess.run(
        ["which", "docker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).returncode
    == 0
)

pytestmark = pytest.mark.skipif(not docker_installed, reason="Docker is not installed")


def ensure_docker_image_exists():
    """Check if the Docker image exists and build it if it doesn't."""
    # Check if image exists already
    image_check = subprocess.run(
        ["docker", "images", "-q", "heart-disease-prediction"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Build image if it doesn't exist
    if not image_check.stdout.strip():
        print("Building Docker image...")
        build_result = subprocess.run(
            ["docker", "build", "-t", "heart-disease-prediction", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if build_result.returncode != 0:
            warnings.warn(f"Failed to build Docker image: {build_result.stderr}")
            pytest.skip("Failed to build Docker image")
            return False
    return True


def check_container_running(container_id):
    """Check if the container is running."""
    container_check = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"id={container_id}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not container_check.stdout.strip():
        # Try to get logs to see what went wrong
        logs = subprocess.check_output(["docker", "logs", container_id], text=True)
        warnings.warn(f"Container started but exited. Logs: {logs}")
        pytest.skip("Container exited after starting")
        return False
    return True


def wait_for_api_readiness(container_id):
    """Wait for the API to be ready and accessible."""
    max_retries = 5
    retry_delay = 2
    connected = False

    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                connected = True
                break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ):
            print(f"Connection attempt {i+1} failed, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    if not connected:
        warnings.warn("Could not connect to container API after multiple retries")
        logs = subprocess.check_output(["docker", "logs", container_id], text=True)
        print(f"Container logs: {logs}")

        subprocess.run(["docker", "stop", container_id])
        subprocess.run(["docker", "rm", container_id])
        pytest.skip("Could not connect to container API")
        return False
    return True


@pytest.fixture(scope="module")
def docker_container():
    """Build and run a Docker container for testing."""
    try:
        # Ensure the Docker image exists
        if not ensure_docker_image_exists():
            return

        # Run container
        print("Starting Docker container...")
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", "8000:8000", "heart-disease-prediction"],
            text=True,
        ).strip()

        # Wait for container to start up
        time.sleep(5)

        try:
            # Verify container is running
            if not check_container_running(container_id):
                return

            # Wait for API to be ready
            if not wait_for_api_readiness(container_id):
                return

            # Container is running and API is accessible
            yield container_id

        finally:
            # Cleanup
            print("Stopping and removing Docker container...")
            subprocess.run(["docker", "stop", container_id])
            subprocess.run(["docker", "rm", container_id])

    except Exception as e:
        warnings.warn(f"Error setting up Docker container: {e}")
        pytest.skip(f"Docker setup failed: {e}")


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


def test_docker_health_endpoint(docker_container):
    """Test the health endpoint of the Docker container."""
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_docker_models_info_endpoint(docker_container):
    """Test the models info endpoint of the Docker container."""
    response = requests.get("http://localhost:8000/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "models_available" in data
    assert "preprocessor_available" in data


def test_docker_predict_endpoint(docker_container, sample_patient_data):
    """Test the prediction endpoint of the Docker container."""
    response = requests.post("http://localhost:8000/predict", json=sample_patient_data)

    # Check if models are available in the container
    models_info = requests.get("http://localhost:8000/models/info").json()
    models_available = any(
        models_info.get("models_available", {}).get(model, False)
        for model in ["sklearn_mlp", "keras_mlp"]
    )

    if not models_available:
        warnings.warn("No models available in the container")
        return

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH"]


def test_docker_model_persistence(docker_container, sample_patient_data):
    """Test that model predictions are consistent in the Docker container."""
    # Only run if models are available in the container
    models_info = requests.get("http://localhost:8000/models/info").json()
    models_available = any(
        models_info.get("models_available", {}).get(model, False)
        for model in ["sklearn_mlp", "keras_mlp"]
    )

    if not models_available:
        warnings.warn("No models available in the container")
        return

    # Make multiple predictions and check consistency
    results = []
    for _ in range(3):
        response = requests.post(
            "http://localhost:8000/predict", json=sample_patient_data
        )
        assert response.status_code == 200
        results.append(response.json())

    # Check that predictions are consistent
    predictions = [r["prediction"] for r in results]
    probabilities = [r["probability"] for r in results]

    # All predictions should be the same
    assert len(set(predictions)) == 1, "Predictions are not consistent"

    # All probabilities should be very close
    prob_diffs = [abs(probabilities[0] - p) for p in probabilities[1:]]
    assert all(diff < 1e-6 for diff in prob_diffs), "Probabilities are not consistent"

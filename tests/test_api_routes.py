import logging

import pytest
from fastapi.testclient import TestClient

from heart_api.factory.test_app_factory import create_test_app

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def client():
    app = create_test_app()
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid(client: TestClient):
    # Debug the app state
    app = client.app
    logger.info("Checking app state configuration...")
    logger.info(f"App has state attribute: {hasattr(app, 'state')}")
    if hasattr(app, "state"):
        logger.info(f"State has model_predictor: {hasattr(app.state, 'model_predictor')}")
        if hasattr(app.state, "model_predictor"):
            logger.info(f"Model predictor is not None: {app.state.model_predictor is not None}")

    sample = {
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
    response = client.post("/predict", json=sample)
    if response.status_code != 200:
        logger.error(f"Response status code: {response.status_code}")
        logger.error(f"Response content: {response.content}")
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "probability" in payload
    assert "risk_level" in payload


def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_auth_token_with_api_key(client):
    response = client.post("/auth/token", headers={"X-API-Key": "dev_api_key"})
    assert response.status_code == 200
    token = response.json().get("access_token")
    assert token is not None


def test_auth_token_invalid_key(client):
    response = client.post("/auth/token", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401

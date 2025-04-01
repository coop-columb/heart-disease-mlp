import pytest
from fastapi.testclient import TestClient

from heart_api.core import model_predictor
from heart_api.factory.test_app_factory import create_test_app


@pytest.fixture(scope="session")
def client():
    app = create_test_app()
    app.state.model_predictor = model_predictor
    return TestClient(app)

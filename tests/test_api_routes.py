from fastapi.testclient import TestClient

from heart_api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid():
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
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "probability" in payload
    assert "risk_level" in payload

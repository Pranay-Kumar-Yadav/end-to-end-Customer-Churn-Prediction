from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_reports_model_state():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ready", "model_not_trained"}


def test_prediction_requires_features():
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422

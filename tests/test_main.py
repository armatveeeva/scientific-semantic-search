from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_recommend_endpoint():
    payload = {"text": "машинное обучение в материаловедении", "top_k": 1}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert "similarity" in response.json()[0]

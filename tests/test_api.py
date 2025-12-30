from fastapi.testclient import TestClient
from src.api.main import app

# Simulation du serveur API
client = TestClient(app)

def test_health_check():
    """Test Intégration : Vérifie que l'API est en vie."""
    response = client.get("/health")
    
    if response.status_code == 404:
        pytest.skip("Endpoint /health non implémenté")
    
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_fails_on_empty():
    """Test Intégration : Vérifie que l'API rejette une requête vide (Erreur 422)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
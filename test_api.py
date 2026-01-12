"""
Script de test rapide pour l'API Iris
Usage: python test_api.py
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test du endpoint /health"""
    print("🔍 Test du endpoint /health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print(f"✅ Health check OK: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_predict():
    """Test du endpoint /predict"""
    print("\n🔍 Test du endpoint /predict...")
    
    # Exemple de données pour setosa
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prédiction réussie:")
            print(f"   Données: {json.dumps(test_data, indent=2)}")
            print(f"   Résultat: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Prédiction échouée: {response.status_code}")
            print(f"   Réponse: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Tests de l'API Iris MLOps")
    print("=" * 50)
    
    health_ok = test_health()
    predict_ok = test_predict()
    
    print("\n" + "=" * 50)
    if health_ok and predict_ok:
        print("✅ Tous les tests sont passés !")
    else:
        print("❌ Certains tests ont échoué.")
    print("=" * 50)

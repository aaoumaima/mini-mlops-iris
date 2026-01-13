# 📊 Monitoring & Rollback Strategy

## 🔍 Monitoring

### Endpoint de Monitoring

L'API expose un endpoint `/metrics` pour le monitoring basique :

```bash
curl http://localhost:8000/metrics
```

**Réponse :**
```json
{
  "model": {
    "path": "models/best_model.joblib",
    "exists": true,
    "size_bytes": 12345,
    "last_modified": "2024-01-15T10:30:00",
    "type": "Pipeline"
  },
  "api": {
    "uptime_seconds": 3600.5,
    "total_predictions": 150,
    "start_time": "2024-01-15T09:30:00"
  },
  "monitoring": {
    "status": "operational",
    "note": "Basic monitoring endpoint. For advanced drift detection, use MLflow or dedicated monitoring tools."
  }
}
```

### Métriques Disponibles

1. **Modèle**
   - Existence et taille du fichier
   - Date de dernière modification
   - Type de modèle

2. **API**
   - Uptime (temps de fonctionnement)
   - Nombre total de prédictions
   - Heure de démarrage

### Monitoring Avancé avec MLflow

Pour un monitoring plus avancé (drift detection, performance tracking) :

1. **Utiliser MLflow Tracking** :
   ```python
   import mlflow
   
   # Log des prédictions en production
   mlflow.log_metric("prediction_count", _prediction_count)
   mlflow.log_metric("api_uptime", uptime_seconds)
   ```

2. **Drift Detection** (à implémenter) :
   - Comparer les distributions des features en production vs training
   - Utiliser des outils comme Evidently AI ou NannyML
   - Alertes automatiques en cas de drift détecté

3. **Performance Monitoring** :
   - Temps de réponse de l'API
   - Taux d'erreur
   - Distribution des prédictions

---

## 🔄 Stratégie de Rollback

### Rollback avec MLflow

MLflow permet de gérer plusieurs versions de modèles et de faire du rollback facilement.

#### 1. Enregistrer les Modèles dans MLflow Model Registry

Lors de l'entraînement, les modèles sont automatiquement enregistrés dans MLflow :

```python
# Dans train_baseline.py
mlflow.sklearn.log_model(pipe, name="model")
```

#### 2. Promouvoir un Modèle en Production

Via l'interface MLflow UI (http://localhost:5000) :
- Aller dans "Models"
- Sélectionner un modèle
- Changer le stage à "Production"

Ou via l'API Python :
```python
import mlflow

# Promouvoir un modèle en production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="iris-model",
    version=2,
    stage="Production"
)
```

#### 3. Rollback vers une Version Précédente

**Méthode 1 : Via MLflow UI**
1. Ouvrir MLflow UI : http://localhost:5000
2. Aller dans "Models" → "iris-model"
3. Sélectionner une version précédente (ex: version 1)
4. Changer le stage à "Production"
5. L'ancienne version devient "Archived"

**Méthode 2 : Via API Python**
```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. Archiver la version actuelle en production
current_prod = client.get_latest_versions("iris-model", stages=["Production"])[0]
client.transition_model_version_stage(
    name="iris-model",
    version=current_prod.version,
    stage="Archived"
)

# 2. Promouvoir une version précédente
client.transition_model_version_stage(
    name="iris-model",
    version=1,  # Version à restaurer
    stage="Production"
)
```

**Méthode 3 : Script de Rollback Automatique**

Créer `scripts/rollback_model.py` :
```python
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import shutil

def rollback_to_version(model_name: str, target_version: int):
    """Rollback vers une version spécifique du modèle"""
    client = MlflowClient()
    
    # 1. Récupérer le modèle de la version cible
    model_uri = f"models:/{model_name}/{target_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # 2. Sauvegarder l'ancien modèle
    import os
    if os.path.exists("models/best_model.joblib"):
        shutil.copy("models/best_model.joblib", "models/best_model.joblib.backup")
    
    # 3. Charger le nouveau modèle
    joblib.dump(model, "models/best_model.joblib")
    
    # 4. Mettre à jour le stage dans MLflow
    # Archiver la version actuelle
    current = client.get_latest_versions(model_name, stages=["Production"])
    if current:
        client.transition_model_version_stage(
            name=model_name,
            version=current[0].version,
            stage="Archived"
        )
    
    # Promouvoir la version cible
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage="Production"
    )
    
    print(f"✅ Rollback vers version {target_version} effectué")

if __name__ == "__main__":
    import sys
    version = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rollback_to_version("iris-model", version)
```

**Usage :**
```bash
python scripts/rollback_model.py 1  # Rollback vers version 1
```

#### 4. Rollback Automatique dans l'API

Pour un rollback automatique basé sur des métriques :

```python
# Dans api/main.py (exemple conceptuel)
def check_model_performance():
    """Vérifie les performances et déclenche rollback si nécessaire"""
    # Calculer métriques récentes
    recent_accuracy = calculate_recent_accuracy()
    
    if recent_accuracy < THRESHOLD:
        # Rollback automatique
        rollback_to_previous_version()
        return True
    return False
```

### Rollback avec Docker

Si le modèle est déployé via Docker :

1. **Versionner les images Docker** :
   ```bash
   docker build -t iris-api:v1.0 .
   docker build -t iris-api:v1.1 .  # Nouvelle version
   ```

2. **Rollback** :
   ```bash
   # Arrêter la version actuelle
   docker compose down
   
   # Modifier docker-compose.yml pour utiliser l'ancienne version
   # Puis redémarrer
   docker compose up -d
   ```

### Checklist de Rollback

Avant de faire un rollback :

- [ ] Identifier la version cible du modèle
- [ ] Vérifier les métriques de la version cible dans MLflow
- [ ] Sauvegarder la version actuelle
- [ ] Tester le modèle de rollback sur un échantillon
- [ ] Notifier l'équipe
- [ ] Effectuer le rollback
- [ ] Vérifier que l'API fonctionne correctement
- [ ] Monitorer les métriques post-rollback

---

## 📈 Recommandations pour Production

### Monitoring Complet

1. **Intégrer un outil de monitoring dédié** :
   - Prometheus + Grafana
   - Datadog
   - New Relic

2. **Drift Detection** :
   - Evidently AI
   - NannyML
   - Custom scripts avec scikit-learn

3. **Alertes** :
   - Email/Slack en cas de drift
   - Alertes sur baisse de performance
   - Alertes sur erreurs API

### Rollback Automatique

Pour un environnement de production, considérer :

1. **Canary Deployments** : Déployer progressivement
2. **A/B Testing** : Tester deux versions en parallèle
3. **Feature Flags** : Activer/désactiver des versions
4. **Circuit Breakers** : Arrêter automatiquement en cas d'erreurs

---

## 🔗 Ressources

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Evidently AI - Drift Detection](https://www.evidentlyai.com/)
- [NannyML - Performance Monitoring](https://www.nannyml.com/)

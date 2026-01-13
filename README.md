#  Mini-Projet MLOps - Classification Iris

Pipeline MLOps complet pour la classification de fleurs Iris (setosa, versicolor, virginica) en utilisant les bonnes pratiques industrielles.

##  Objectif du projet

Mettre en place un pipeline MLOps complet pour un modèle de Machine Learning, depuis les données jusqu'au déploiement, en utilisant les bonnes pratiques industrielles.

Le modèle prédit la classe de la fleur Iris à partir de ses caractéristiques (sepal_length, sepal_width, petal_length, petal_width).

##  Technologies utilisées

- **Python 3.11** - Langage de programmation
- **Scikit-learn** - Modèles ML (Logistic Regression, SVM)
- **MLflow** - Tracking des expériences et versioning des modèles
- **Optuna** - Optimisation automatique des hyperparamètres
- **FastAPI** - API REST pour les prédictions
- **Docker & Docker Compose** - Conteneurisation et orchestration
- **Git** - Versioning du code
- **DVC + MinIO** - Versioning des données

##  Structure du projet

```
mini-mlops-iris/
├── api/
│   └── main.py              # API FastAPI
├── src/
│   ├── training/
│   │   ├── train_baseline.py    # Entraînement baseline
│   │   └── train_optuna.py      # Optimisation avec Optuna
│   └── utils/
│       └── load_data.py         # Chargement des données
├── data/
│   ├── raw/                     # Données brutes (versionnées avec DVC)
│   └── processed/               # Données traitées
├── models/
│   └── best_model.joblib        # Meilleur modèle sauvegardé
├── mlruns/                       # Runs MLflow
├── mlflow.db                     # Base de données MLflow
├── Dockerfile                    # Image Docker pour l'API
├── Dockerfile.mlflow             # Image Docker pour MLflow
├── docker-compose.yml            # Orchestration API + MLflow
├── requirements.txt              # Dépendances Python (développement)
├── requirements-docker.txt       # Dépendances Python (Docker - minimal)
├── render.yaml                   # Configuration Render.com
├── railway.json                  # Configuration Railway.app
├── Procfile                      # Configuration Heroku
├── runtime.txt                   # Version Python pour Heroku
├── README.md                     # Documentation principale
├── VERIFICATION.md               # Guide de vérification
└── QUICK_START.md                # Guide de démarrage rapide
```

##  Architecture du pipeline MLOps

### Vue d'ensemble du pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS COMPLET                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  1. DONNÉES  │  Iris Dataset (4 features: sepal, petal)
│   (DVC)      │  └─> Versioning avec DVC + MinIO
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. PRÉPARATION│  Train/Test Split (80/20)
│   DONNÉES    │  └─> StandardScaler
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. ENTRAÎNEMENT│ Baseline: Logistic Regression / SVM
│   BASELINE   │  └─> Sauvegarde: models/best_model.joblib
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. MLFLOW    │  Tracking: paramètres, métriques, modèles
│   TRACKING   │  └─> UI: http://localhost:5000
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. OPTUNA    │  Optimisation hyperparamètres (C, kernel)
│ OPTIMIZATION │  └─> Sélection meilleur modèle
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 6. FASTAPI   │  API REST de prédiction
│    SERVICE   │  └─> Endpoints: /health, /predict
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 7. DOCKER    │  Conteneurisation
│   COMPOSE    │  └─> API + MLflow orchestrés
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 8. CLOUD     │  Déploiement (Render, Railway, Azure)
│  DEPLOYMENT  │  └─> Production ready
└──────────────┘
```

### Architecture technique détaillée

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE STACK                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   iris-api       │         │   mlflow         │          │
│  │   (FastAPI)      │         │   (Tracking UI)  │          │
│  │                  │         │                  │          │
│  │  Port: 8000      │         │  Port: 5000      │          │
│  │  /health         │         │  /mlflow         │          │
│  │  /predict        │         │  /experiments    │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         └──────────┬───────────────────┘                     │
│                    │                                           │
│         ┌──────────▼──────────┐                               │
│         │  mlops-network      │                               │
│         │  (bridge network)   │                               │
│         └────────────────────┘                               │
│                                                               │
│  Volumes partagés:                                            │
│  - ./models → /app/models (read-only)                        │
│  - ./mlruns → /mlflow/mlruns                                 │
│  - ./mlflow.db → /mlflow/mlflow.db                           │
└─────────────────────────────────────────────────────────────┘
```

##  Installation et utilisation

### Prérequis

- Python 3.11+
- Docker & Docker Compose
- Git
- DVC (optionnel, pour le versioning des données)

### 1. Cloner le projet

```bash
git clone <repository-url>
cd mini-mlops-iris
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Pour l'entraînement complet (avec MLflow et Optuna) :
```bash
pip install mlflow optuna dvc
```

### 4. Charger les données

```bash
python src/utils/load_data.py
```

### 5. Entraîner le modèle baseline

```bash
# Logistic Regression
python src/training/train_baseline.py --model logreg --C 1.0

# SVM
python src/training/train_baseline.py --model svm --C 1.0 --kernel rbf
```

### 6. Optimiser avec Optuna

```bash
python src/training/train_optuna.py
```

### 7. Lancer MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Accéder à l'interface : http://localhost:5000

### 8. Lancer l'API localement

```bash
uvicorn api.main:app --reload
```

Accéder à l'API : http://localhost:8000
- Documentation Swagger : http://localhost:8000/docs
- Health check : http://localhost:8000/health

### 9. Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

##  Déploiement avec Docker

### Build et lancement avec Docker Compose

```bash
# Construire et lancer le service
docker compose up --build

# Lancer en arrière-plan
docker compose up -d

# Voir les logs
docker compose logs -f

# Arrêter les services
docker compose down
```

### Build manuel de l'image Docker

```bash
# Build l'image
docker build -t iris-api:1.0 .

# Lancer le conteneur
docker run --rm -p 8000:8000 iris-api:1.0
```

### Services disponibles avec Docker Compose

Lorsque vous lancez `docker compose up`, deux services sont disponibles :

- **API FastAPI** : http://localhost:8000
  - Documentation Swagger : http://localhost:8000/docs
  - Health check : http://localhost:8000/health
  - Endpoint de prédiction : http://localhost:8000/predict

- **MLflow UI** : http://localhost:5000
  - Interface de tracking des expériences
  - Visualisation des métriques et paramètres
  - Comparaison des modèles

##  Endpoints de l'API

### `GET /health`

Vérifie l'état de l'API.

**Réponse :**
```json
{
  "status": "ok"
}
```

### `POST /predict`

Effectue une prédiction de classe Iris.

**Body (JSON) :**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Réponse :**
```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

**Classes :**
- `0` : setosa
- `1` : versicolor
- `2` : virginica

##  Commandes utiles

### MLflow

```bash
# Lancer MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Lister les expériences
mlflow experiments list
```

### Docker

```bash
# Voir les conteneurs actifs
docker ps

# Build l'image
docker build -t iris-api:1.0 .

# Lancer le conteneur
docker run --rm -p 8000:8000 iris-api:1.0

# Voir les logs
docker compose logs -f
```

### Git & DVC

```bash
# Versioning des données avec DVC
dvc add data/raw/iris.csv
dvc push  # Push vers MinIO (si configuré)
```

##  Présentation du projet

> "J'ai réalisé un mini-projet MLOps complet incluant la gestion des données avec DVC, le suivi des expériences avec MLflow, l'optimisation avec Optuna et le déploiement d'un modèle via une API FastAPI, le tout versionné avec Git et conteneurisé avec Docker."

##  Fonctionnalités implémentées

### Core MLOps
- Structure MLOps complète et organisée
- Versioning des données (DVC + MinIO)
- Entraînement de modèles baseline (Logistic Regression, SVM)
-  Tracking des expériences (MLflow)
-  Optimisation des hyperparamètres (Optuna)

### Déploiement
-  API REST de prédiction (FastAPI)
-  Conteneurisation (Docker)
- Orchestration (Docker Compose avec API + MLflow)
-  Configuration cloud (Render, Railway, Heroku)

### Documentation
-  README complet avec architecture
-  Guide de vérification (VERIFICATION.md)
-  Guide de démarrage rapide (QUICK_START.md)
-  Schémas du pipeline MLOps

##  Troubleshooting

### Le modèle n'est pas trouvé

Assurez-vous d'avoir entraîné le modèle avant de lancer l'API :
```bash
python src/training/train_baseline.py --model logreg
```

### Port déjà utilisé

Si le port 8000 ou 5000 est déjà utilisé, modifiez les ports dans `docker-compose.yml`.

### Erreur Docker

Vérifiez que Docker est bien lancé :
```bash
docker ps
```

##  Déploiement Cloud (Bonus)

Le projet inclut des configurations pour le déploiement sur différentes plateformes cloud.

### Render.com

1. Créer un compte sur [Render](https://render.com)
2. Connecter votre repository GitHub
3. Render détectera automatiquement `render.yaml`
4. Le service sera déployé automatiquement

**Fichier de configuration :** `render.yaml`

### Railway.app

1. Créer un compte sur [Railway](https://railway.app)
2. Créer un nouveau projet depuis GitHub
3. Railway utilisera `railway.json` pour la configuration
4. Le déploiement se fait automatiquement

**Fichier de configuration :** `railway.json`

### Heroku

1. Installer [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Se connecter : `heroku login`
3. Créer une app : `heroku create iris-api`
4. Déployer : `git push heroku main`

**Fichiers de configuration :** `Procfile`, `runtime.txt`

### Azure Container Instances

```bash
# Build et push vers Azure Container Registry
az acr build --registry <registry-name> --image iris-api:latest .

# Déployer
az container create \
  --resource-group <resource-group> \
  --name iris-api \
  --image <registry-name>.azurecr.io/iris-api:latest \
  --dns-name-label iris-api \
  --ports 8000
```

### Variables d'environnement Cloud

Pour tous les déploiements cloud, assurez-vous de :
- Configurer le port via la variable `PORT` (généralement fournie automatiquement)
- Inclure le modèle `best_model.joblib` dans l'image Docker
- Configurer les volumes persistants si nécessaire (pour MLflow)

##  Ressources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Optuna Documentation](https://optuna.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Render Documentation](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app/)

##  Licence

Ce projet est un projet éducatif.

---

**Auteur** : Mini-Projet MLOps  
**Date** : 2024

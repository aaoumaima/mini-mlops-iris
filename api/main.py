from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Iris MLOps API")

MODEL_PATH = "models/best_model.joblib"

# Vérification modèle
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model not found. Train the model first.")

model = joblib.load(MODEL_PATH)

# Tracking simple pour monitoring
_prediction_count = 0
_start_time = datetime.now()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Endpoint de monitoring - retourne des métriques sur le modèle et l'API"""
    model_path = Path(MODEL_PATH)
    model_size = model_path.stat().st_size if model_path.exists() else 0
    model_mtime = (
        datetime.fromtimestamp(model_path.stat().st_mtime)
        if model_path.exists()
        else None
    )

    return {
        "model": {
            "path": MODEL_PATH,
            "exists": os.path.exists(MODEL_PATH),
            "size_bytes": model_size,
            "last_modified": model_mtime.isoformat() if model_mtime else None,
            "type": str(type(model).__name__),
        },
        "api": {
            "uptime_seconds": (datetime.now() - _start_time).total_seconds(),
            "total_predictions": _prediction_count,
            "start_time": _start_time.isoformat(),
        },
        "monitoring": {
            "status": "operational",
            "note": "Basic monitoring endpoint. For advanced drift detection, use MLflow or dedicated monitoring tools.",
        },
    }


@app.post("/predict")
def predict(data: IrisInput):
    global _prediction_count
    try:
        X = np.array(
            [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        )

        prediction = model.predict(X)[0]
        _prediction_count += 1

        classes = ["setosa", "versicolor", "virginica"]

        return {"prediction": int(prediction), "class_name": classes[int(prediction)]}

    except Exception as e:
        return {"error": str(e)}

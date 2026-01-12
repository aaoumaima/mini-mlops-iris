from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Iris MLOps API")

MODEL_PATH = "models/best_model.joblib"

# Vérification modèle
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model not found. Train the model first.")

model = joblib.load(MODEL_PATH)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: IrisInput):
    try:
        X = np.array([[  
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        prediction = model.predict(X)[0]

        classes = ["setosa", "versicolor", "virginica"]

        return {
            "prediction": int(prediction),
            "class_name": classes[int(prediction)]
        }

    except Exception as e:
        return {
            "error": str(e)
        }

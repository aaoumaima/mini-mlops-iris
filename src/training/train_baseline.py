# src/training/train_baseline.py
import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


RAW_CSV_PATH = "data/raw/iris.csv"
MODEL_OUT_PATH = "models/best_model.joblib"


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training for Iris with MLflow tracking")

    parser.add_argument("--data", type=str, default=RAW_CSV_PATH, help="Path to raw iris csv")
    parser.add_argument("--experiment", type=str, default="iris-mlflow-runs", help="MLflow experiment name")

    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm"], help="Model type")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter C")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel (used only if model=svm)")

    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    # On suppose que la dernière colonne est la target (comme iris.csv classique)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def build_pipeline(model_name: str, C: float, kernel: str):
    if model_name == "logreg":
        estimator = LogisticRegression(C=C, max_iter=1000)
    else:
        estimator = SVC(C=C, kernel=kernel, probability=True)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    return pipe


def main():
    args = parse_args()

    # Vérifs fichiers/dossiers
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset introuvable: {args.data}. Lance d'abord: python src/utils/load_data.py")

    Path("models").mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # MLflow
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.model}-C{args.C}-kernel{args.kernel}"):
        # Log paramètres
        mlflow.log_param("model", args.model)
        mlflow.log_param("C", args.C)
        mlflow.log_param("kernel", args.kernel if args.model == "svm" else "N/A")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("data_path", args.data)

        # Build + fit
        pipe = build_pipeline(args.model, args.C, args.kernel)
        pipe.fit(X_train, y_train)

        # Predict + metrics
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # Log model to MLflow (nom de dossier "model")
        mlflow.sklearn.log_model(sk_model=pipe, name="model")

        # Sauvegarde locale (IMPORTANT pour l'API FastAPI)
        joblib.dump(pipe, MODEL_OUT_PATH)

        print(f"✅ {args.model} | C={args.C} | kernel={args.kernel} | acc={acc:.4f} | f1={f1:.4f}")
        print(f"✅ Modèle sauvegardé: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()

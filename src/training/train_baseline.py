import os
import argparse
import joblib
import mlflow
import mlflow.sklearn

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def build_model(model_name: str, C: float, kernel: str):
    if model_name == "logreg":
        clf = LogisticRegression(C=C, max_iter=1000)
    elif model_name == "svm":
        clf = SVC(C=C, kernel=kernel, probability=True)
    else:
        raise ValueError("model must be 'logreg' or 'svm'")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="logreg", choices=["logreg", "svm"]
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
    )
    args = parser.parse_args()

    iris = load_iris()
    X = iris.data  # ✅ 4 features
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("iris-mlflow-runs")

    with mlflow.start_run(run_name=f"{args.model}_C{args.C}_kernel{args.kernel}"):
        pipe = build_model(args.model, args.C, args.kernel)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model", args.model)
        mlflow.log_param("C", args.C)
        mlflow.log_param("kernel", args.kernel)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # log model
        mlflow.sklearn.log_model(pipe, name="model")

        # ✅ save local best model for API
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, "models/best_model.joblib")

        print(
            f"✅ {args.model} | C={args.C} | kernel={args.kernel} | acc={acc:.4f} | f1={f1:.4f}"
        )


if __name__ == "__main__":
    main()

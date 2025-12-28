import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train(model_type: str, C: float, kernel: str):
    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop(columns=["target", "target_name"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("iris-mlflow-runs")

    with mlflow.start_run():
        if model_type == "logreg":
            model = LogisticRegression(max_iter=300, C=C)
        else:
            model = SVC(C=C, kernel=kernel)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("C", C)
        if model_type == "svm":
            mlflow.log_param("kernel", kernel)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        mlflow.sklearn.log_model(model, "model")
        print(f"✅ {model_type} | C={C} | kernel={kernel} | acc={acc:.4f} | f1={f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="rbf")
    args = parser.parse_args()

    train(args.model, args.C, args.kernel)

import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

def load_and_save_iris():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target_name"] = df["target"].map(
        dict(enumerate(iris.target_names))
    )

    df.to_csv("data/raw/iris.csv", index=False)
    print("✅ Dataset Iris sauvegardé dans data/raw/iris.csv")

if __name__ == "__main__":
    load_and_save_iris()

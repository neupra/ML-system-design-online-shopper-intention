import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


TARGET_COL = "Revenue"


def main():
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python src/evaluate.py <model_path> <test_csv> <metrics_path>"
        )

    model_path = Path(sys.argv[1])
    test_csv = Path(sys.argv[2])
    metrics_path = Path(sys.argv[3])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    model = joblib.load(model_path)

    test_df = pd.read_csv(test_csv)
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    if TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in test CSV.")

    X = test_df.drop(columns=[TARGET_COL])
    y_true = test_df[TARGET_COL].astype(int)

    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_score_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_binary": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_binary": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score_binary": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    roc_auc = None

    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None

    if roc_auc is None and hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X)
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None

    metrics["roc_auc"] = roc_auc

    cm = confusion_matrix(y_true, y_pred)

    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
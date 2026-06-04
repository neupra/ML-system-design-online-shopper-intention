import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

model_path = sys.argv[1]
test_csv = sys.argv[2]
metrics_path = Path(sys.argv[3])
metrics_path.parent.mkdir(parents=True, exist_ok=True)

model = joblib.load(model_path)
test_df = pd.read_csv(test_csv)

X = test_df.drop(columns=["Revenue"])
y_true = test_df["Revenue"]
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]  # ROC-AUC এর জন্য probability

metrics = {
    "accuracy":  float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, average="weighted")),
    "recall":    float(recall_score(y_true, y_pred, average="weighted")),
    "f1_score":  float(f1_score(y_true, y_pred, average="weighted")),
    "roc_auc":   float(roc_auc_score(y_true, y_prob)),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
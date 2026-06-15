from pathlib import Path
from typing import List

import joblib

from serving.app.preprocessing import clean_and_engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "dvc-pipeline" / "models" / "inference_model.joblib"

model = None


def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Inference model not found: {MODEL_PATH}. "
            "First create dvc-pipeline/models/inference_model.joblib."
        )

    model = joblib.load(MODEL_PATH)

    print("=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model type: {type(model)}")
    print("=" * 60)


def is_model_loaded() -> bool:
    return model is not None


def predict_records(records: List[dict]) -> List[dict]:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    df = clean_and_engineer_features(records)

    predictions = model.predict(df)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)

    results = []

    for i, pred in enumerate(predictions):
        pred_int = int(pred)

        result = {
            "prediction": pred_int,
            "prediction_label": "Revenue" if pred_int == 1 else "No Revenue",
        }

        if probabilities is not None:
            proba = probabilities[i]

            if len(proba) == 2:
                result["probability_no_revenue"] = float(proba[0])
                result["probability_revenue"] = float(proba[1])
            else:
                result["probabilities"] = [float(x) for x in proba]

        results.append(result)

    return results
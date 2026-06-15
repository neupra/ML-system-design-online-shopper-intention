import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def main():
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python src/create_inference_model.py "
            "<trained_model.joblib> <inference_model.joblib>"
        )

    trained_model_path = Path(sys.argv[1])
    inference_model_path = Path(sys.argv[2])
    inference_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not trained_model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {trained_model_path}")

    trained_pipeline = joblib.load(trained_model_path)

    if "preprocessor" not in trained_pipeline.named_steps:
        raise ValueError("Trained pipeline must contain a 'preprocessor' step.")

    if "model" not in trained_pipeline.named_steps:
        raise ValueError("Trained pipeline must contain a 'model' step.")

    inference_pipeline = Pipeline(
        steps=[
            ("preprocessor", trained_pipeline.named_steps["preprocessor"]),
            ("model", trained_pipeline.named_steps["model"]),
        ]
    )

    joblib.dump(inference_pipeline, inference_model_path)

    print("=" * 60)
    print("INFERENCE MODEL CREATED")
    print("=" * 60)
    print(f"Loaded trained model from: {trained_model_path}")
    print(f"Saved inference model to: {inference_model_path}")
    print("Removed training-only sampler such as SMOTE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
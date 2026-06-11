import sys
from pathlib import Path
import traceback

print("SKLEARN ONNX EXPORT SCRIPT STARTED", flush=True)

import joblib
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    try:
        print("=" * 60, flush=True)
        print("EXPORTING SKLEARN MODEL TO ONNX", flush=True)
        print("=" * 60, flush=True)

        if len(sys.argv) != 4:
            print("Usage:", flush=True)
            print(
                "python src/export_onnx.py <model.joblib> <train.csv> <model.onnx>",
                flush=True,
            )
            raise ValueError("Invalid number of arguments.")

        model_path = Path(sys.argv[1])
        train_csv_path = Path(sys.argv[2])
        onnx_output_path = Path(sys.argv[3])

        print(f"Model path: {model_path}", flush=True)
        print(f"Train CSV path: {train_csv_path}", flush=True)
        print(f"ONNX output path: {onnx_output_path}", flush=True)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not train_csv_path.exists():
            raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

        onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

        print("Loading joblib model...", flush=True)
        model = joblib.load(model_path)

        print(f"Loaded model type: {type(model)}", flush=True)

        print("Reading train CSV...", flush=True)
        train_df = pd.read_csv(train_csv_path)
        train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

        if "Revenue" not in train_df.columns:
            raise ValueError("Revenue column not found in train CSV.")

        X_train = train_df.drop(columns=["Revenue"])

        n_features = X_train.shape[1]

        print(f"Train shape: {train_df.shape}", flush=True)
        print(f"Input feature count: {n_features}", flush=True)

        initial_types = [
            ("float_input", FloatTensorType([None, n_features]))
        ]

        options = {}

        # For classifiers, zipmap=False gives probability as array if available.
        try:
            options = {
                id(model): {
                    "zipmap": False
                }
            }
        except Exception:
            options = {}

        print("Converting sklearn model to ONNX...", flush=True)

        onnx_model = convert_sklearn(
            model,
            initial_types=initial_types,
            target_opset=12,
            options=options,
        )

        print("Saving ONNX model...", flush=True)

        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved successfully: {onnx_output_path}", flush=True)
        print("=" * 60, flush=True)
        print("EXPORT COMPLETE", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        print("ERROR OCCURRED:", str(e), flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

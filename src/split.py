import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


TARGET_COL = "Revenue"


def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: python src/split.py <input_csv> <output_dir>")

    params = yaml.safe_load(open("params.yaml"))["split"]

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET_COL],
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"Input shape: {df.shape}")
    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Test:  {len(test_df)} rows -> {test_path}")

    print("\nTrain Revenue distribution:")
    print(train_df[TARGET_COL].value_counts())

    print("\nTest Revenue distribution:")
    print(test_df[TARGET_COL].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
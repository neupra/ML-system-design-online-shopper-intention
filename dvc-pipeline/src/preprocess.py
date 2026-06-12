import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


TARGET_COL = "Revenue"


def main():
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python src/preprocess.py <raw_csv_path> <output_dir>"
        )

    raw_data_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    print("=" * 60)
    print("PREPROCESSING STARTED")
    print("Cleaning + feature engineering only")
    print("=" * 60)

    # ==================================================================
    # LOAD RAW DATA
    # ==================================================================
    df_raw = pd.read_csv(raw_data_path)

    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # ==================================================================
    # VALIDATE COLUMNS
    # ==================================================================
    expected_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        TARGET_COL,
    ]

    missing_columns = sorted(set(expected_columns) - set(df_raw.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df_raw[expected_columns].copy()

    print(f"All {len(expected_columns)} required columns are present.")

    # ==================================================================
    # BOOLEAN CLEANING
    # ==================================================================
    bool_map = {
        "TRUE": 1,
        "FALSE": 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
        "1": 1,
        "0": 0,
    }

    for col in ["Weekend", TARGET_COL]:
        df[col] = df[col].map(bool_map)

    if df[TARGET_COL].isna().any():
        raise ValueError("Revenue has invalid values after boolean conversion.")

    if df["Weekend"].isna().any():
        raise ValueError("Weekend has invalid values after boolean conversion.")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)

    # ==================================================================
    # MONTH CLEANING + FEATURE ENGINEERING
    # ==================================================================
    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    month_to_num = {month: index + 1 for index, month in enumerate(month_order)}

    df["Month_num"] = df["Month"].map(month_to_num)

    if df["Month_num"].isna().any():
        mode_month = df["Month_num"].mode().iloc[0]
        df["Month_num"] = df["Month_num"].fillna(mode_month)

    df["TotalSessionDuration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )

    df["ProductInfoRatio"] = df["ProductRelated"] / (df["Informational"] + 1)

    df["EngagementScore"] = df["PageValues"] / (
        df["TotalSessionDuration"] + 1
    )

    df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12.0)

    df = df.drop(columns=["Month", "Month_num"])

    # ==================================================================
    # SAVE CLEANED DATA
    # ==================================================================
    output_csv = output_dir / "online_shoppers_intention_cleaned.csv"
    metadata_path = output_dir / "metadata.json"

    df.to_csv(output_csv, index=False)

    metadata = {
        "stage": "preprocess",
        "description": "Cleaning and feature engineering only. No encoding, scaling, or balancing.",
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "target_name": TARGET_COL,
        "positive_rate": round(float(df[TARGET_COL].mean() * 100), 2),
        "class_distribution": {
            str(k): int(v) for k, v in Counter(df[TARGET_COL]).items()
        },
        "engineered_features": [
            "TotalSessionDuration",
            "ProductInfoRatio",
            "EngagementScore",
            "Month_sin",
            "Month_cos",
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Cleaned data shape: {df.shape}")
    print(f"Saved cleaned CSV to: {output_csv}")
    print(f"Saved metadata to: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

from typing import List

import numpy as np
import pandas as pd


RAW_FEATURES = [
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
]

CLEAN_FEATURES = [
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
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "TotalSessionDuration",
    "ProductInfoRatio",
    "EngagementScore",
    "Month_sin",
    "Month_cos",
]


def clean_and_engineer_features(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    missing_columns = sorted(set(RAW_FEATURES) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[RAW_FEATURES].copy()

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

    df["Weekend"] = df["Weekend"].map(bool_map)

    if df["Weekend"].isna().any():
        raise ValueError("Weekend has invalid values.")

    df["Weekend"] = df["Weekend"].astype(int)

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
        invalid_months = df.loc[df["Month_num"].isna(), "Month"].unique().tolist()
        raise ValueError(f"Invalid Month values: {invalid_months}")

    df["TotalSessionDuration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )

    df["ProductInfoRatio"] = df["ProductRelated"] / (
        df["Informational"] + 1
    )

    df["EngagementScore"] = df["PageValues"] / (
        df["TotalSessionDuration"] + 1
    )

    df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12.0)

    df = df.drop(columns=["Month", "Month_num"])

    df = df[CLEAN_FEATURES].copy()

    return df
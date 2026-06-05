import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

# ======================================================================
# PATHS & PARAMS
# ======================================================================
RAW_DATA_PATH = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
params = yaml.safe_load(open("params.yaml"))["preprocess"]
BALANCING = params["balancing"]  # smote / adasyn / smotetomek / smoteenn /
                                  # borderline_smote / random_oversample /
                                  # random_undersample / tomeklinks / nearmiss / none

TARGET_COL = "Revenue"

print("=" * 60)
print(f"Balancing technique: {BALANCING.upper()}")
print("=" * 60)

# ======================================================================
# STEP 1: LOAD RAW DATA
# ======================================================================
df_raw = pd.read_csv(RAW_DATA_PATH)
print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# ======================================================================
# STEP 2: VALIDATE COLUMNS
# ======================================================================
expected_columns = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
    "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend", TARGET_COL
]
missing_columns = sorted(set(expected_columns) - set(df_raw.columns))
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df_raw[expected_columns].copy()
print(f"All {len(expected_columns)} required columns present")

# ======================================================================
# STEP 3: CLEAN & ENGINEER FEATURES
# ======================================================================

# Convert Weekend & Revenue to 0/1
bool_map = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}
for col in ["Weekend", TARGET_COL]:
    df[col] = df[col].map(bool_map)

# Convert month names to numbers
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_to_num = {m: i + 1 for i, m in enumerate(month_order)}
df["Month_num"] = df["Month"].map(month_to_num)
if df["Month_num"].isna().any():
    df["Month_num"] = df["Month_num"].fillna(df["Month_num"].mode().iloc[0])

# Feature engineering
df["TotalSessionDuration"] = (
    df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
)
df["ProductInfoRatio"] = df["ProductRelated"] / (df["Informational"] + 1)
df["EngagementScore"] = df["PageValues"] / (df["TotalSessionDuration"] + 1)

# Cyclic month features
df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12.0)
df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12.0)
df = df.drop(columns=["Month_num", "Month"])

# Final validation
if df[TARGET_COL].isna().any():
    raise ValueError("Revenue has invalid values after cleaning!")
df[TARGET_COL] = df[TARGET_COL].astype(int)
df["Weekend"] = df["Weekend"].astype(int)

print(f"Cleaned data shape: {df.shape}")

# ======================================================================
# STEP 4: BUILD PREPROCESSING PIPELINE
# ======================================================================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

categorical_features = ["VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
numerical_features = [col for col in X.columns if col not in categorical_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# ======================================================================
# STEP 5: APPLY PIPELINE
# ======================================================================
X_processed = preprocessor.fit_transform(X)

cat_transformer = preprocessor.named_transformers_.get("cat")
encoder = cat_transformer.named_steps.get("encoder") if hasattr(cat_transformer, "named_steps") else cat_transformer
try:
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
except Exception:
    cat_feature_names = np.array([])

processed_feature_names = list(numerical_features) + list(cat_feature_names)
X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X.index)

print(f"Processed shape: {X_processed_df.shape[0]} rows x {X_processed_df.shape[1]} features")
print(f"Original class distribution: {Counter(y)}")

# ======================================================================
# STEP 6: APPLY BALANCING TECHNIQUE
# ======================================================================

# Sampler selection
samplers = {
    "smote":              SMOTE(random_state=42),
    "adasyn":             ADASYN(random_state=42),
    "borderline_smote":   BorderlineSMOTE(random_state=42),
    "random_oversample":  RandomOverSampler(random_state=42),
    "smotetomek":         SMOTETomek(random_state=42),
    "smoteenn":           SMOTEENN(random_state=42),
    "random_undersample": RandomUnderSampler(random_state=42),
    "tomeklinks":         TomekLinks(),
    "nearmiss":           NearMiss(),
    "none":               None,
}

if BALANCING not in samplers:
    raise ValueError(
        f"Unknown balancing technique: '{BALANCING}'\n"
        f"Available: {list(samplers.keys())}"
    )

sampler = samplers[BALANCING]

if sampler is not None:
    X_resampled, y_resampled = sampler.fit_resample(X_processed_df, y)
    print(f"After {BALANCING}: {Counter(y_resampled)}")
else:
    X_resampled, y_resampled = X_processed_df.values, y.values
    print("No balancing applied.")

X_resampled_df = pd.DataFrame(X_resampled, columns=X_processed_df.columns)
y_resampled_df = pd.Series(y_resampled, name=TARGET_COL)

# ======================================================================
# STEP 7: SAVE ARTIFACTS
# ======================================================================

# Save preprocessor
joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

# Save final combined dataset
df_final = pd.concat([X_resampled_df, y_resampled_df], axis=1)
df_final.to_csv(OUTPUT_DIR / "online_shoppers_intention_prepared.csv", index=False)

# Save metadata
metadata = {
    "balancing_technique": BALANCING,
    "total_rows_original": len(df),
    "total_rows_after_balancing": len(df_final),
    "total_features": X_resampled_df.shape[1],
    "target_name": TARGET_COL,
    "positive_rate_original": round(float(y.mean() * 100), 2),
    "class_distribution_original": {str(k): int(v) for k, v in Counter(y).items()},
    "class_distribution_after": {str(k): int(v) for k, v in Counter(y_resampled).items()},
    "engineered_features": [
        "TotalSessionDuration", "ProductInfoRatio",
        "EngagementScore", "Month_sin", "Month_cos"
    ],
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("=" * 60)
print("PREPROCESSING COMPLETE!")
print(f"Balancing: {BALANCING}")
print(f"Final dataset: {df_final.shape[0]} rows x {df_final.shape[1]} columns")
print(f"Saved to: {OUTPUT_DIR}")
print("=" * 60)

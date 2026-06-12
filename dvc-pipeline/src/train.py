import sys
from pathlib import Path
from collections import Counter

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    NearMiss,
)


TARGET_COL = "Revenue"


def build_sampler(balancing: str, random_state: int):
    samplers = {
        "smote": SMOTE(random_state=random_state),
        "adasyn": ADASYN(random_state=random_state),
        "borderline_smote": BorderlineSMOTE(random_state=random_state),
        "random_oversample": RandomOverSampler(random_state=random_state),
        "smotetomek": SMOTETomek(random_state=random_state),
        "smoteenn": SMOTEENN(random_state=random_state),
        "random_undersample": RandomUnderSampler(random_state=random_state),
        "tomeklinks": TomekLinks(),
        "nearmiss": NearMiss(),
        "none": None,
    }

    if balancing not in samplers:
        raise ValueError(
            f"Unknown balancing technique: '{balancing}'. "
            f"Available: {list(samplers.keys())}"
        )

    return samplers[balancing]


def build_model(train_config: dict):
    algorithm = train_config["algorithm"]
    random_state = train_config["random_state"]

    if algorithm == "LogisticRegression":
        model_params = dict(train_config.get("LogisticRegression", {}))
        model_params.setdefault("random_state", random_state)
        model = LogisticRegression(**model_params)

    elif algorithm == "SVM":
        model_params = dict(train_config.get("SVM", {}))
        model_params.setdefault("random_state", random_state)

        # Needed for predict_proba and ROC-AUC
        model_params.setdefault("probability", True)

        model = SVC(**model_params)

    elif algorithm == "RandomForest":
        model_params = dict(train_config.get("RandomForest", {}))
        model_params.setdefault("random_state", random_state)
        model_params.setdefault("n_jobs", -1)
        model = RandomForestClassifier(**model_params)

    elif algorithm == "XGBoost":
        model_params = dict(train_config.get("XGBoost", {}))
        model_params.setdefault("random_state", random_state)
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("eval_metric", "logloss")
        model = XGBClassifier(**model_params)

    else:
        raise ValueError(
            f"Unsupported algorithm: '{algorithm}'. "
            "Please check params.yaml."
        )

    return algorithm, model


def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: python src/train.py <train_csv> <model_output_path>")

    config = yaml.safe_load(open("params.yaml"))

    train_config = config["train"]
    preprocess_config = config["preprocess"]

    random_state = train_config["random_state"]
    balancing = preprocess_config.get("balancing", "none")

    train_csv = Path(sys.argv[1])
    model_out = Path(sys.argv[2])
    model_out.parent.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    train_df = pd.read_csv(train_csv)
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train CSV.")

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL].astype(int)

    categorical_features = [
        "VisitorType",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
    ]

    missing_cat_cols = sorted(set(categorical_features) - set(X.columns))

    if missing_cat_cols:
        raise ValueError(f"Missing categorical columns: {missing_cat_cols}")

    numerical_features = [
        col for col in X.columns if col not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    sampler = build_sampler(balancing, random_state)

    algorithm, model = build_model(train_config)

    pipeline_steps = [
        ("preprocessor", preprocessor),
    ]

    if sampler is not None:
        pipeline_steps.append(("sampler", sampler))

    pipeline_steps.append(("model", model))

    pipeline = Pipeline(steps=pipeline_steps)

    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Balancing: {balancing}")
    print(f"Train shape: {train_df.shape}")
    print(f"Feature count before encoding: {X.shape[1]}")
    print(f"Class distribution before balancing: {Counter(y)}")

    print("\nPipeline steps:")
    for step_name, step_obj in pipeline.steps:
        print(f"  {step_name}: {step_obj.__class__.__name__}")

    print("\nFitting pipeline...")
    pipeline.fit(X, y)

    joblib.dump(pipeline, model_out)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Saved full pipeline to: {model_out}")
    print(f"Saved object type: {type(pipeline)}")

    final_model = pipeline.named_steps["model"]

    if hasattr(final_model, "classes_"):
        print(f"Classes: {final_model.classes_}")

    print("=" * 60)


if __name__ == "__main__":
    main()
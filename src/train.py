import sys
from pathlib import Path
import joblib
import pandas as pd
import yaml

# সব সম্ভাব্য ক্লাসিফায়ার মডেল ইমপোর্ট করা

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


config = yaml.safe_load(open("params.yaml"))["train"]
algorithm = config["algorithm"]
rand_state = config["random_state"]

train_csv = sys.argv[1]
model_out = Path(sys.argv[2])
model_out.parent.mkdir(parents=True, exist_ok=True)


train_df = pd.read_csv(train_csv)
X = train_df.drop(columns=["Revenue"])
y = train_df["Revenue"]

print(f"--- Training Info ---")
print(f"Selected Algorithm: {algorithm}")
print(f"Feature count: {X.shape[1]}")


if algorithm == "LogisticRegression":
    model_params = config["LogisticRegression"]
    model = LogisticRegression(**model_params, random_state=rand_state)

elif algorithm == "SVM":
    model_params = config["SVM"]
    model = SVC(**model_params, random_state=rand_state)

elif algorithm == "RandomForest":
    model_params = config["RandomForest"]
    model = RandomForestClassifier(
        **model_params, random_state=rand_state, n_jobs=-1
    )

elif algorithm == "XGBoost":
    model_params = config["XGBoost"]
    model = XGBClassifier(**model_params, random_state=rand_state, n_jobs=-1)


else:
    raise ValueError(
        f"Unsupported algorithm: '{algorithm}'. Please check your params.yaml file."
    )


print(f"Fitting {algorithm} model...")
model.fit(X, y)


joblib.dump(model, model_out)
print(f"Saved model successfully -> {model_out}")


if hasattr(model, "classes_"):
    print(f"Classes: {model.classes_}")
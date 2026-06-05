import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

input_path = sys.argv[1]
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)


df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["Revenue"],
)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
print(f"Train Revenue distribution:\n{train_df['Revenue'].value_counts()}")
print(f"Test Revenue distribution:\n{test_df['Revenue'].value_counts()}")
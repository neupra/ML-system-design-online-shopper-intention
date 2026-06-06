# 🛒 Online Shoppers Intention — ML System Design

An end-to-end **ML System Design** project for predicting online shoppers' purchase intention. This repository is structured to support the full ML lifecycle — from data versioning to model serving.

---

## 🗺️ Project Roadmap

| Part | Topic | Status |
|---|---|---|
| **Part 1** | Data Versioning & ML Pipeline (DVC) | ✅ Done |
| **Part 2** | Experiment Tracking (MLflow) | 🔜 Coming |
| **Part 3** | Model Serving & API | 🔜 Coming |
| **Part 4** | Monitoring & Drift Detection | 🔜 Coming |

---

## 📁 Repository Structure

```
ml-system-design/
├── dvc-pipeline/          # Part 1: Data versioning & ML pipeline
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── data/
│   ├── models/
│   ├── metrics/
│   ├── dvc.yaml
│   └── params.yaml
│
├── serving/               # Part 3: Model serving (coming soon)
├── monitoring/            # Part 4: Monitoring (coming soon)
└── README.md
```

---

## Part 1: DVC Pipeline

### 📊 Dataset

**Online Shoppers Purchasing Intention Dataset**

| Property | Value |
|---|---|
| Total rows | 12,330 |
| Features | 18 |
| Target | `Revenue` (Binary: 0/1) |
| Class distribution | 0 → 10,422 \| 1 → 1,908 (Imbalanced) |

---

### ⚙️ Setup

```bash
pip install "pathspec==0.11.2"
pip install "dvc[all]"
pip install pandas scikit-learn imbalanced-learn pyyaml joblib numpy
```

```bash
git init
dvc init
dvc add data/online_shoppers_intention.csv
git add .
git commit -m "Initialize DVC pipeline"
```

---

### 🔄 Pipeline

```
preprocess → split → train → evaluate
```

| Stage | Input | Output |
|---|---|---|
| **preprocess** | Raw CSV | Cleaned + Balanced CSV |
| **split** | Processed CSV | train.csv, test.csv |
| **train** | train.csv | model.joblib |
| **evaluate** | model.joblib, test.csv | metrics.json |

---

### 🧪 Experiments

**Run Pipeline:**
```bash
dvc repro
```

**Change Balancing Technique** — edit `params.yaml`:

```yaml
preprocess:
  balancing: smote
```

| Value | Technique | Type |
|---|---|---|
| `smote` | SMOTE | Oversampling |
| `adasyn` | ADASYN | Oversampling |
| `borderline_smote` | BorderlineSMOTE | Oversampling |
| `random_oversample` | RandomOverSampler | Oversampling |
| `smotetomek` | SMOTETomek | Combination |
| `smoteenn` | SMOTEENN | Combination |
| `random_undersample` | RandomUnderSampler | Undersampling |
| `tomeklinks` | TomekLinks | Undersampling |
| `nearmiss` | NearMiss | Undersampling |
| `none` | No balancing | — |

**Change Model Parameters** — edit `params.yaml`:

```yaml
train:
  model: random_forest
  n_estimators: 100
  max_depth: 12
  random_state: 42
```

---

### 📈 Metrics

```bash
dvc metrics show          # current metrics
dvc metrics diff HEAD~1 HEAD   # compare versions
```

| Metric | Description |
|---|---|
| `accuracy` | Overall accuracy |
| `precision` | Weighted precision |
| `recall` | Weighted recall |
| `f1_score` | Weighted F1 score |
| `roc_auc` | ROC-AUC score |

---

### 🗂️ Version Control Workflow

```bash
dvc repro
git add .
git commit -m "Experiment: SMOTE, RF n_estimators=100"
git push
```

**Go to the Previous version :**
```bash
git log --oneline
git checkout <commit-id>
dvc checkout
```

**Return to the Latest:**
```bash
git checkout main
dvc checkout
```

---

### 📋 Feature Engineering

| Feature | Formula |
|---|---|
| `TotalSessionDuration` | Admin + Info + ProductRelated Duration |
| `ProductInfoRatio` | ProductRelated / (Informational + 1) |
| `EngagementScore` | PageValues / (TotalSessionDuration + 1) |
| `Month_sin` | Cyclic encoding of month |
| `Month_cos` | Cyclic encoding of month |

---

## Part 2: Experiment Tracking — Coming Soon

MLflow integration for tracking experiments, parameters, and model registry.

---

## Part 3: Model Serving — Coming Soon

REST API for model inference using FastAPI/Flask.

---

## Part 4: Monitoring — Coming Soon

Data drift detection and model performance monitoring.

# 🛒 Online Shoppers Intention — ML System Design

An end-to-end **ML System Design** project for predicting online shoppers' purchase intention. The project is designed around a reproducible ML lifecycle, including data versioning, pipeline automation, model training, inference model creation, and model serving.

---

## 🗺️ Project Roadmap

| Part       | Topic                               | Status         |
| ---------- | ----------------------------------- | -------------- |
| **Part 1** | Data Versioning & ML Pipeline (DVC) | ✅ Done         |
| **Part 2** | Experiment Tracking (MLflow)        | 🔜 Coming      |
| **Part 3** | Model Serving & API                 | 🚧 In Progress |
| **Part 4** | Monitoring & Drift Detection        | 🔜 Coming      |

---

## 📁 Repository Structure

```text
ml-system-design/
│
├── dvc-pipeline/                  # Part 1: Data versioning & ML pipeline
│   ├── src/
│   │   ├── preprocess.py          # Data preprocessing
│   │   ├── split.py               # Train-test splitting
│   │   ├── train.py               # Model training
│   │   ├── evaluate.py            # Model evaluation
│   │   └── create_inference_model.py
│   │
│   ├── data/
│   │   ├── online_shoppers_intention.csv
│   │   ├── processed/
│   │   └── prepared/
│   │
│   ├── models/
│   │   ├── model.joblib
│   │   └── inference_model.joblib
│   │
│   ├── metrics/
│   │   └── metrics.json
│   │
│   ├── dvc.yaml
│   └── params.yaml
│
├── serving/                       # Part 3: Model serving & API
│   └── app/
│       ├── main.py                # FastAPI application
│       ├── model_loader.py        # Model loading
│       ├── preprocessing.py       # Inference preprocessing
│       └── schemas.py              # Input validation schemas
│
├── monitoring/                    # Part 4: Monitoring & drift detection
│
└── README.md
```

---

# Part 1: Data Versioning & ML Pipeline

## 📊 Dataset

**Online Shoppers Purchasing Intention Dataset**

The dataset contains session-level information about online shoppers and whether a purchase was completed during the session.

| Property           | Value              |
| ------------------ | ------------------ |
| Total rows         | 12,330             |
| Features           | 18                 |
| Target             | `Revenue` (Binary) |
| Class 0            | 10,422             |
| Class 1            | 1,908              |
| Class distribution | Imbalanced         |

---

## ⚙️ Setup

Install the required dependencies:

```bash
pip install "pathspec==0.11.2"
pip install "dvc[all]"
pip install pandas scikit-learn imbalanced-learn pyyaml joblib numpy
```

Initialize Git and DVC:

```bash
git init
dvc init
```

Track the dataset with DVC:

```bash
dvc add data/online_shoppers_intention.csv
```

Commit the DVC configuration:

```bash
git add .
git commit -m "Initialize DVC pipeline"
```

---

## 🔄 ML Pipeline

The current DVC pipeline is:

```text
preprocess → split → train → create_inference_model → evaluate
```

### Pipeline Stages

| Stage                      | Input            | Output                   |
| -------------------------- | ---------------- | ------------------------ |
| **preprocess**             | Raw CSV          | Cleaned CSV + metadata   |
| **split**                  | Cleaned CSV      | `train.csv` + `test.csv` |
| **train**                  | Training CSV     | `model.joblib`           |
| **create_inference_model** | Trained model    | `inference_model.joblib` |
| **evaluate**               | Model + test CSV | `metrics.json`           |

Run the complete pipeline using:

```bash
dvc repro
```

DVC automatically determines which stages need to be rerun when dependencies or parameters change.

---

## ⚖️ Class Imbalance Handling

The target variable `Revenue` is imbalanced:

```text
Class 0 → 10,422 sessions
Class 1 →  1,908 sessions
```

The training pipeline supports multiple balancing techniques through `params.yaml`.

Current configuration:

```yaml
preprocess:
  balancing: smote
```

Supported techniques:

| Value                | Technique          | Type          |
| -------------------- | ------------------ | ------------- |
| `smote`              | SMOTE              | Oversampling  |
| `adasyn`             | ADASYN             | Oversampling  |
| `borderline_smote`   | BorderlineSMOTE    | Oversampling  |
| `random_oversample`  | RandomOverSampler  | Oversampling  |
| `smotetomek`         | SMOTETomek         | Combination   |
| `smoteenn`           | SMOTEENN           | Combination   |
| `random_undersample` | RandomUnderSampler | Undersampling |
| `tomeklinks`         | TomekLinks         | Undersampling |
| `nearmiss`           | NearMiss           | Undersampling |
| `none`               | No balancing       | —             |

> Balancing is applied during model training and is not applied to the test data.

---

## 🤖 Models

The training configuration supports:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

Example configuration:

```yaml
train:
  algorithm: SVM

  LogisticRegression:
    max_iter: 1000
    class_weight: balanced

  SVM:
    C: 1.0
    kernel: rbf
    probability: true
    class_weight: balanced

  RandomForest:
    n_estimators: 200
    max_depth: 12
    class_weight: balanced

  XGBoost:
    n_estimators: 200
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
```

Model configuration is separated from the source code using `params.yaml`, allowing experiments without modifying the training code.

---

## 📈 Model Evaluation

The evaluation pipeline generates:

```text
metrics/metrics.json
```

Current evaluation metrics include:

| Metric      | Purpose                                 |
| ----------- | --------------------------------------- |
| `accuracy`  | Overall classification accuracy         |
| `precision` | Correctness of positive predictions     |
| `recall`    | Ability to identify purchasing sessions |
| `f1_score`  | Balance between precision and recall    |
| `roc_auc`   | Overall ranking/discrimination ability  |

Because the target class is imbalanced, **F1-score, Recall, Precision, and ROC-AUC** are considered alongside accuracy.

---

## 🧪 DVC Experiments

Run the pipeline:

```bash
dvc repro
```

Change parameters in `params.yaml` and reproduce the pipeline.

For example:

```yaml
preprocess:
  balancing: smote
```

or:

```yaml
preprocess:
  balancing: adasyn
```

Then run:

```bash
dvc repro
```

Compare metrics:

```bash
dvc metrics show
```

```bash
dvc metrics diff HEAD~1 HEAD
```

---

## 🗂️ Data & Model Versioning

DVC is used to track datasets and ML pipeline outputs while Git tracks source code and configuration.

Example workflow:

```bash
dvc repro

git add .
git commit -m "Experiment: SMOTE with SVM"
git push
```

### Restore a Previous Version

```bash
git log --oneline
git checkout <commit-id>
dvc checkout
```

### Return to Latest Version

```bash
git checkout main
dvc checkout
```

This allows previous dataset/model states and experiment configurations to be reproduced.

---

## 🛠️ Feature Engineering

The pipeline creates additional features to represent user engagement and session behavior.

| Feature                | Formula                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| `TotalSessionDuration` | Administrative Duration + Informational Duration + Product Related Duration |
| `ProductInfoRatio`     | Product Related / (Informational + 1)                                       |
| `EngagementScore`      | Page Value / (Total Session Duration + 1)                                   |
| `Month_sin`            | Cyclic encoding of month                                                    |
| `Month_cos`            | Cyclic encoding of month                                                    |

---

# Part 2: Experiment Tracking — Coming Soon

**MLflow** will be integrated to track:

* Model parameters
* Training metrics
* Experiments
* Model versions
* Experiment comparisons

Planned workflow:

```text
DVC Pipeline
     ↓
Model Training
     ↓
MLflow Experiment Tracking
     ↓
Model Comparison
     ↓
Best Model
```

---

# Part 3: Model Serving & API — 🚧 In Progress

A model serving layer is being developed using **FastAPI**.

The serving system provides:

### Single Prediction

```text
POST /predict
```

### Batch Prediction

```text
POST /predict_batch
```

The serving pipeline performs:

```text
User Session Data
        ↓
Input Validation
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Inference Model
        ↓
Prediction + Probability
        ↓
JSON Response
```

The inference model is generated separately from the training model:

```text
model.joblib
      ↓
create_inference_model.py
      ↓
inference_model.joblib
```

The inference model contains the preprocessing and trained model required for prediction while excluding training-only operations such as SMOTE.

This separation ensures that synthetic oversampling is performed only during training and not during inference.

---

# Part 4: Monitoring & Drift Detection — 🔜 Coming Soon

The monitoring layer will focus on maintaining model reliability after deployment.

Planned components include:

* Data drift detection
* Feature distribution monitoring
* Prediction distribution monitoring
* Model performance monitoring
* Drift alerts
* Retraining triggers

Planned workflow:

```text
Production Data
      ↓
Monitoring
      ↓
Drift Detection
      ↓
Performance Check
      ↓
Alert / Retrain
```

---

# 🏗️ Overall ML System Architecture

```text
                    ┌──────────────────┐
                    │   Raw Dataset    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Preprocess  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Train/Test Split │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Model Training   │
                    │ + SMOTE          │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Model Evaluation │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Inference Model  │
                    └────────┬─────────┘
                             ↓
══════════════════════════════════════════════════
                    ONLINE SERVING
══════════════════════════════════════════════════
                             ↓
                    ┌──────────────────┐
                    │   FastAPI API    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Input Validation │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Feature Pipeline │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Inference Model  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Prediction +     │
                    │ Probability      │
                    └────────┬─────────┘
                             ↓
══════════════════════════════════════════════════
                       MONITORING
══════════════════════════════════════════════════
                             ↓
                    Drift & Performance
                       Monitoring
```

---

# 🎯 Project Goals

The system is designed around three business objectives:

### 1. Maximize Sales

Identify sessions with a high probability of purchase and enable targeted actions.

### 2. Reduce Advertising Cost

Avoid unnecessarily targeting sessions with a low probability of conversion.

### 3. Maintain a Balanced System

Use appropriate classification metrics and probability-based predictions to balance sales opportunities and marketing costs.

---

# 🚀 Future Improvements

* MLflow experiment tracking
* Model registry
* Production deployment
* Data drift detection
* Model performance monitoring
* Automated retraining
* Online learning
* Cloud deployment
* CI/CD for ML pipelines

---

## 👨‍💻 Technologies

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn
DVC
Git
GitHub
Joblib
PyYAML
FastAPI
```

---

## 📌 Project Status

The project is being developed incrementally as an **end-to-end ML Systems Design project**, with emphasis on reproducibility, modularity, scalable inference, and the complete ML lifecycle.


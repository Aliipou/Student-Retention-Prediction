<div align="center">

# Student Retention Prediction

[![CI](https://github.com/Aliipou/Student-Retention-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/Student-Retention-Prediction/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](requirements.txt)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Predicts which students will drop out before it is too late to intervene.**

</div>

---

## 1. Problem

Student dropout costs higher-education institutions **$26 billion annually** in the United States alone. Every dropout also represents a personal and economic setback for the individual.

The standard institutional response is reactive: interventions trigger only after a student has already failed a course, missed a payment, or disappeared from attendance records. By week 8-10 of a semester, the window for meaningful intervention has often closed.

This system enables **early detection**: using engagement signals available in week 4 (LMS logins, attendance rate, assignment submission patterns), it identifies students at high dropout risk while there is still time to assign advisor support, reduce course load, or provide financial aid. At a cohort of 500 students, the model flags approximately 65 at-risk students per semester, a caseload feasible for an advising team.

---

## 2. Architecture

The pipeline has four stages: data ingestion and feature engineering, model training, evaluation, and serving.

```
Raw Student Data (CSV / Student Information System)
      |
      v
[Feature Engineering]   Attendance rate, GPA variance, assignment
  src/preprocessing.py  completion, LMS activity, social integration
      |
      v
[Training Pipeline]     SMOTE oversampling -> GradientBoosting fit
  src/train_pipeline.py  -> calibration -> model serialization
      |
      v
[Evaluation]            AUC-ROC, F1, SHAP feature importance
  src/evaluation.py
      |
      v
  Serving Layer
  +-------------------------------------------+
  | FastAPI REST API      src/api.py         |
  |   POST /predict -> risk score + factors  |
  |   GET  /health  -> uptime + model status |
  +-------------------------------------------+
  | Streamlit Dashboard   src/dashboard.py   |
  |   Interactive cohort risk explorer       |
  +------------------------------------------+
```

---

## 3. Key Design Decisions

**Why GradientBoosting over neural networks.**
Tabular student data is typically low-dimensional (10-20 features) and sparse. Neural networks require large labeled datasets and provide poor calibration out of the box. Gradient-boosted trees train reliably on datasets of 2,000-20,000 records, produce naturally calibrated probabilities when combined with isotonic regression, and are interpretable at the feature-importance level. Advisors can understand why a student is flagged without a data science background.

**Why SMOTE for class imbalance.**
Dropout rates are typically 10-20% of enrolled students. Training a classifier directly on this imbalance results in a model that maximizes accuracy by predicting retained for everyone. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic at-risk examples in feature space, forcing the model to learn the minority-class decision boundary. In testing, SMOTE reduced the false-negative rate by 22% compared to no resampling.

**Why SHAP for explainability.**
Advisors need to know why a student is flagged, not just that they are. SHAP (SHApley Additive eyPlanations) provides additive feature attributions that sum to the model prediction, are theoretically grounded in cooperative game theory, and are consistent across model types.

**Why Streamlit for the dashboard.**
The primary consumers of this system are academic advisors, not engineers. Streamlit allows deploying an interactive, filterable cohort view without a frontend team. For a production deployment serving thousands of concurrent advisor sessions, a dedicated React frontend against the FastAPI backend would be appropriate.

---

## 4. Tech Stack

| Component | Technology | Justification |
|---|---|---|
| ML model | scikit-learn GradientBoosting | Robust on tabular data; calibrated probabilities; interpretable |
| Class imbalance | imbalanced-learn SMOTE | Prevents majority-class bias without discarding data |
| Explainability | SHAP | Model-agnostic, theoretically grounded feature attribution |
| Additional models | XGBoost, LightGBM | Ensemble comparison; LightGBM wins on speed for large cohorts |
| API  | FastAPI + Uvicorn | Async, OpenAPI-documented, sub-millisecond overhead |
| Dashboard | Streamlit | Rapid deployment of advisor-facing UI |
| Serialization | joblib | sklearn-compatible model persistence |
| Testing | pytest + pytest-cov | Unit and integration coverage on preprocessing and model code |
| Security | pip-audit (CI) | Dependency vulnerability scanning on every push |
| Container | Docker multi-stage + non-root | Minimal image size; no root process in production |

---

## 5. Running Locally

**Prerequisites:** Python 3.12, pip

```bash
git clone https://github.com/Aliipou/Student-Retention-Prediction.git
cd Student-Retention-Prediction

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data and train models
python -m src.train_pipeline

# Run the Streamlit dashboard
streamlit run src/dashboard.py
# Dashboard: http://localhost
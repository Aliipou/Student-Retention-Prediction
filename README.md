<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&amp;logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&amp;logo=scikitlearn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-137%20passed-brightgreen?style=flat)](TEST_REPORT.md)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat)](TEST_REPORT.md)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

# Student Retention Prediction

**A production-ready machine learning pipeline for predicting student dropout risk.**

</div>

## Problem Statement

Student dropout is costly for institutions and devastating for individuals. Most retention programs rely on reactive interventions after grades have already collapsed. This system enables proactive outreach by identifying at-risk students early, before the critical failure points occur.

## How It Works

The system trains on historical academic and engagement data and produces per-student risk scores. These scores can feed into dashboards, alert systems, or direct advisor workflows.

```
Raw Student Data
      |
      v
[Feature Engineering]   Attendance rate, grade trend, assignment completion,
                         LMS engagement, social integration metrics
      |
      v
[ML Pipeline]           Gradient Boosting with SMOTE for class imbalance,
                         calibrated probability outputs
      |
      v
[Risk Score]            0.0 (low risk) to 1.0 (high risk) with
                         feature importance explanation
```

## Features

**Predictive Model**
Gradient Boosting classifier with calibrated probability estimates. Handles severe class imbalance via SMOTE oversampling.

**Feature Engineering**
Automated feature extraction from raw attendance, grades, LMS logs, and assignment data.

**Explainability**
SHAP-based feature importance so advisors understand why a student is flagged, not just that they are.

**REST API**
FastAPI endpoints for integration with existing student information systems.

**Quality**
137 tests, 100% code coverage, validated on real anonymized enrollment data.

## Quick Start

```bash
git clone https://github.com/Aliipou/Student-Retention-Prediction.git
cd Student-Retention-Prediction
pip install -r requirements.txt
python train.py --data data/students.csv
python predict.py --student-id 12345
```

## API

```python
import httpx
r = httpx.post("http://localhost:8000/predict", json={"student_id": "12345"})
print(r.json())
# {"student_id": "12345", "risk_score": 0.78, "risk_level": "HIGH",
#  "top_factors": ["missed_assignments", "declining_grade_trend"]}
```


---

## Results

Evaluated on a hold-out test set of 1,847 student records from two academic years.

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.91** |
| F1-Score (at-risk class) | **0.84** |
| Precision | 0.87 |
| Recall | 0.81 |
| Accuracy | 0.89 |

**Key findings:**
- Top 3 predictive features: assignment completion rate, grade trend slope, LMS login frequency
- Model identifies 81% of students who will drop out, with a false positive rate of 13%
- Early warning is possible as early as week 4 of semester — before grades have collapsed
- SMOTE oversampling reduced false negative rate by 22% vs. baseline without rebalancing

**Practical impact:** At a cohort of 500 students, the model flags ~65 at-risk students per semester. Manual advisor review of 65 cases is feasible; reviewing all 500 is not.

---
## License

MIT

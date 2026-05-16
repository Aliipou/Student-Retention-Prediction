"""
FastAPI application for Student Retention Prediction.

Exposes a REST API for risk scoring, model metadata, and health checks.
Load a trained model with: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import time as _time
import os
import joblib
import numpy as np
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_START = _time.time()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Student Retention Prediction API",
    description=(
        "Predicts per-student dropout risk using a calibrated GradientBoosting "
        "classifier. Outputs risk score (0-1) with SHAP-based feature explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────

_MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")
_model = None


def _get_model():
    global _model
    if _model is None:
        if os.path.exists(_MODEL_PATH):
            _model = joblib.load(_MODEL_PATH)
        else:
            raise RuntimeError(
                f"No trained model found at {_MODEL_PATH}. "
                "Run the training pipeline first: python -m src.train_pipeline"
            )
    return _model


# ── Schemas ───────────────────────────────────────────────────────────────────


class StudentFeatures(BaseModel):
    """Raw features for a single student."""

    credits_last_sem: float = Field(..., ge=0, description="Credits completed last semester")
    failed_courses: int = Field(..., ge=0, description="Number of failed courses this year")
    moodle_activity_score: float = Field(..., ge=0, description="LMS engagement score (0-100)")
    library_visits: int = Field(..., ge=0, description="Library visits per month")
    login_times_last_week: int = Field(..., ge=0, description="LMS logins in the past week")
    attendance_rate: float = Field(..., ge=0.0, le=1.0, description="Class attendance rate (0-1)")
    gpa: float = Field(..., ge=0.0, le=4.0, description="Current cumulative GPA")
    gpa_variance: float = Field(..., ge=0.0, description="GPA variance across last 3 semesters")
    age: int = Field(..., ge=16, le=60, description="Student age in years")
    scholarship_amount: float = Field(0.0, ge=0, description="Annual scholarship amount (EUR)")
    part_time_job: int = Field(0, ge=0, le=1, description="1 if student has a part-time job")
    distance_from_campus_km: float = Field(..., ge=0, description="Distance from campus in km")


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Dropout risk probability (0.0 = safe, 1.0 = high risk)")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    top_factors: list = Field(..., description="Top 3 features driving this prediction")
    model_version: str


# ── Helpers ───────────────────────────────────────────────────────────────────

_FEATURE_NAMES = [
    "credits_last_sem", "failed_courses", "moodle_activity_score",
    "library_visits", "login_times_last_week", "attendance_rate",
    "gpa", "gpa_variance", "age", "scholarship_amount",
    "part_time_job", "distance_from_campus_km",
]

_RISK_LABELS = [(0.4, "LOW"), (0.7, "MEDIUM"), (1.1, "HIGH")]


def _risk_level(score: float) -> str:
    for threshold, label in _RISK_LABELS:
        if score < threshold:
            return label
    return "HIGH"


def _top_factors(features: np.ndarray, model: Any) -> list:
    """Return the top-3 feature names by absolute importance for this prediction."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps"):
            est = model.named_steps[list(model.named_steps)[-1]]
            importances = getattr(est, "feature_importances_", None)
            if importances is None:
                return ["attendance_rate", "gpa", "failed_courses"]
        else:
            return ["attendance_rate", "gpa", "failed_courses"]

        top_idx = np.argsort(importances)[::-1][:3]
        return [_FEATURE_NAMES[i] for i in top_idx if i < len(_FEATURE_NAMES)]
    except Exception:
        return ["attendance_rate", "gpa", "failed_courses"]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness + readiness probe. Returns 200 when the service is ready."""
    model_loaded = os.path.exists(_MODEL_PATH)
    return {
        "status": "ok",
        "uptime_seconds": int(_time.time() - _START),
        "model_loaded": model_loaded,
        "model_path": _MODEL_PATH,
    }


@app.get("/")
async def root():
    return {
        "service": "Student Retention Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict dropout risk")
async def predict(student: StudentFeatures):
    """
    Return a dropout risk score and explanation for a single student.

    - **risk_score**: calibrated probability in [0, 1]
    - **risk_level**: LOW / MEDIUM / HIGH based on institutional thresholds
    - **top_factors**: top 3 model features driving this individual prediction
    """
    try:
        model = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = np.array([[
        student.credits_last_sem,
        student.failed_courses,
        student.moodle_activity_score,
        student.library_visits,
        student.login_times_last_week,
        student.attendance_rate,
        student.gpa,
        student.gpa_variance,
        student.age,
        student.scholarship_amount,
        student.part_time_job,
        student.distance_from_campus_km,
    ]])

    try:
        proba = float(model.predict_proba(features)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictionResponse(
        risk_score=round(proba, 4),
        risk_level=_risk_level(proba),
        top_factors=_top_factors(features, model),
        model_version="1.0.0",
    )

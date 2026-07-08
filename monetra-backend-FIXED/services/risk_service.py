"""
services/risk_service.py — Model health & failure risk scoring
==============================================================
Combines drift metrics, prediction confidence, and error-rate
signals into a single composite health score and risk label.

Risk tiers (matching frontend colour coding):
  health_score ≥ 75  → LOW    (green)
  health_score 50-75 → MEDIUM (amber)
  health_score < 50  → HIGH   (red)
"""

from typing import List, Dict


# ── Weights for the composite health score ────────────────────────────────────
W_ACCURACY  = 0.35   # model accuracy vs threshold
W_DRIFT     = 0.30   # average PSI across features
W_CONFIDENCE= 0.20   # average prediction confidence
W_ERROR     = 0.15   # error / anomaly rate

ACCURACY_THRESHOLD = 0.85   # below this → degraded
DRIFT_PSI_HIGH     = 0.25   # PSI threshold for "severe drift"


def compute_risk_level(health_score: float) -> str:
    """Convert numeric health score (0–100) to a risk label."""
    if health_score >= 75:
        return "LOW"
    elif health_score >= 50:
        return "MEDIUM"
    return "HIGH"


def compute_failure_probability(health_score: float, avg_psi: float) -> float:
    """
    Heuristic estimate of the probability that the model will fail
    (accuracy drop below threshold) within the next 7 days.
    """
    drift_factor   = min(avg_psi / DRIFT_PSI_HIGH, 1.0)
    health_factor  = max(0.0, (100 - health_score) / 100)
    prob           = 0.6 * health_factor + 0.4 * drift_factor
    return round(min(max(prob, 0.01), 0.99), 4)


def compute_health_score(
    accuracy:        float,
    avg_confidence:  float,
    avg_psi:         float,
    error_rate:      float,
) -> float:
    """
    Composite health score 0–100:
      • accuracy_score   : how far above threshold we are
      • drift_score      : inverse of normalised PSI
      • confidence_score : avg prediction confidence
      • error_score      : inverse of error rate
    """
    accuracy_score    = min(accuracy / ACCURACY_THRESHOLD, 1.0) * 100
    drift_score       = max(0.0, 1.0 - avg_psi / DRIFT_PSI_HIGH) * 100
    confidence_score  = avg_confidence * 100
    error_score       = max(0.0, 1.0 - error_rate * 10) * 100   # error_rate in [0,1]

    health = (
        W_ACCURACY   * accuracy_score
      + W_DRIFT      * drift_score
      + W_CONFIDENCE * confidence_score
      + W_ERROR      * error_score
    )
    return round(min(max(health, 0.0), 100.0), 2)


def infer_feature_contributions(application: dict) -> dict:
    """
    Lightweight local feature-weight attribution (not SHAP).
    Returns a dict of feature → contribution_score (0–1).
    Useful for the frontend "feature importance" bar in the prediction panel.
    """
    credit_score   = application.get("cibil_score") or application.get("credit_score") or 700
    annual_income  = application.get("income_annum") or application.get("annual_income") or 500000
    loan_amount    = application.get("loan_amount") or 300000
    applicant_age  = application.get("applicant_age") or 35
    loan_tenure    = application.get("loan_term") or application.get("loan_tenure") or 36

    income_loan_ratio = min(annual_income / max(loan_amount, 1), 5) / 5

    raw = {
        "credit_score":    (credit_score - 300) / 600,
        "annual_income":   income_loan_ratio,
        "loan_amount":     1 - income_loan_ratio,
        "applicant_age":   min((applicant_age - 18) / 40, 1.0),
        "loan_tenure":     1 - loan_tenure / 360,
        "employment_type": 0.6,   # salaried is safest; approximated
    }

    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}


def build_risk_summary(
    predictions:  List[dict],
    drift_summary: Dict,
) -> dict:
    """
    Aggregate a batch of recent predictions + drift data into a risk summary.
    """
    if not predictions:
        return {
            "overall_risk": "LOW", "health_score": 90.0,
            "failure_probability": 0.05,
            "drift_contribution": 0.0, "latency_contribution": 0.1,
            "bias_contribution": 0.05,
        }

    avg_confidence = sum(p.get("confidence", 0.85) for p in predictions) / len(predictions)
    avg_psi        = drift_summary.get("avg_psi", 0.1)

    from utils.model_loader import get_active_model_key, get_all_meta
    meta = get_all_meta()
    active_key = get_active_model_key()
    active = meta.get(active_key, {})
    accuracy = active.get("accuracy", 0.924)
    error_rate     = 0.008       # ~0.8% as shown in the frontend

    health_score        = compute_health_score(accuracy, avg_confidence, avg_psi, error_rate)
    failure_probability = compute_failure_probability(health_score, avg_psi)
    overall_risk        = compute_risk_level(health_score)

    drift_contrib   = min(avg_psi / DRIFT_PSI_HIGH, 1.0)
    latency_contrib = 0.28   # static placeholder; hook into actual latency metrics
    bias_contrib    = 0.09

    return {
        "overall_risk":         overall_risk,
        "health_score":         health_score,
        "failure_probability":  failure_probability,
        "drift_contribution":   round(drift_contrib, 4),
        "latency_contribution": latency_contrib,
        "bias_contribution":    bias_contrib,
    }

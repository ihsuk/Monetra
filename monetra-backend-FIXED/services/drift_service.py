"""
services/drift_service.py — Data drift detection
=================================================
Implements:
  • PSI  (Population Stability Index)  — bucket-based distribution comparison
  • KS   (Kolmogorov-Smirnov) test     — non-parametric distribution test
  • Concept / label drift detection    — approval-rate shift

PSI thresholds (industry standard):
  < 0.10  → no drift (LOW)
  0.10–0.25 → moderate drift
  > 0.25  → high drift
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Feature names must match the order used during model training ─────────────
FEATURE_NAMES = [
    "loan_amount", "annual_income", "credit_score",
    "applicant_age", "loan_tenure", "employment_type"
]

# ── Reference distribution (baseline) — kept in memory after seed ─────────────
# Loaded from DB at startup via seed_service; overwritten by compute functions.
REFERENCE_STATS: Dict[str, dict] = {}


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between a reference and a current distribution.
    Adds a small epsilon to avoid log(0).
    """
    eps = 1e-4
    # Add tiny random jitter to smooth out discrete features
    ref_jitter = reference + np.random.normal(0, 1e-3, len(reference))
    cur_jitter = current + np.random.normal(0, 1e-3, len(current))

    # Use reference-defined quantile edges so bucket widths are stable
    breakpoints = np.percentile(ref_jitter, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)   # collapse duplicate edges

    ref_counts, _ = np.histogram(ref_jitter, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_jitter, bins=breakpoints)

    ref_pct = (ref_counts + eps) / len(reference)
    cur_pct = (cur_counts + eps) / len(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(psi, 4))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Returns (ks_stat, p_value) from the two-sample KS test."""
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(round(ks_stat, 4)), float(round(p_value, 4))


def classify_drift(psi: float, p_value: float) -> Tuple[str, str]:
    """
    Map PSI + p-value to a drift status label.
    Returns (drift_type, status).
    """
    if psi > 0.25:
        status     = "HIGH"
        drift_type = "Data Drift"
    elif psi > 0.10 or p_value < 0.05:
        status     = "MODERATE"
        drift_type = "Pred. Drift"
    else:
        status     = "LOW"
        drift_type = "Stable"
    return drift_type, status


def analyse_feature(
    feature: str,
    reference_samples: np.ndarray,
    current_samples:   np.ndarray,
) -> dict:
    """Full drift analysis for a single feature."""
    psi               = compute_psi(reference_samples, current_samples)
    ks_stat, p_value  = compute_ks(reference_samples, current_samples)
    drift_type, status = classify_drift(psi, p_value)

    ref_mean = reference_samples.mean()
    cur_mean = current_samples.mean()
    ref_var  = reference_samples.var()
    cur_var  = current_samples.var()

    mean_delta_pct = ((cur_mean - ref_mean) / (ref_mean + 1e-9)) * 100
    var_delta_pct  = ((cur_var  - ref_var)  / (ref_var  + 1e-9)) * 100

    return {
        "feature":        feature,
        "psi_score":      psi,
        "ks_stat":        ks_stat,
        "p_value":        p_value,
        "mean_delta_pct": round(mean_delta_pct, 2),
        "var_delta_pct":  round(var_delta_pct, 2),
        "drift_type":     drift_type,
        "status":         status,
    }


def run_full_drift_analysis(
    recent_predictions: List[dict],
    reference_stats:    Dict[str, dict],
) -> dict:
    """
    Given a list of recent prediction-log dicts and a reference-stats dict,
    compute full drift metrics for all numeric features.

    Returns a summary dict matching DriftSummary schema.
    """
    if not recent_predictions:
        logger.warning("No recent predictions to analyse drift.")
        return _empty_drift_summary()

    feature_results = []
    ks_failures     = 0

    for feature in FEATURE_NAMES[:-1]:   # skip categorical employment_type
        ref_stats = reference_stats.get(feature)
        if ref_stats is None:
            continue

        if feature == "loan_tenure":
            ref_samples = np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 1000)
        elif feature == "credit_score":
            ref_samples = np.random.uniform(
                ref_stats["min_val"], ref_stats["max_val"], 1000
            )
        else:
            # Reconstruct reference distribution from stats via normal approximation
            ref_samples = np.random.normal(
                ref_stats["mean"], ref_stats["std"] + 1e-6, 1000
            ).clip(ref_stats["min_val"], ref_stats["max_val"])

        cur_samples = np.array([
            float(p.get(feature, ref_stats["mean"]))
            for p in recent_predictions
            if p.get(feature) is not None
        ])

        if len(cur_samples) < 10:
            continue

        result = analyse_feature(feature, ref_samples, cur_samples)
        feature_results.append(result)

        if result["p_value"] < 0.05:
            ks_failures += 1

    high_drift    = [f for f in feature_results if f["status"] == "HIGH"]
    moderate_drift= [f for f in feature_results if f["status"] == "MODERATE"]
    avg_psi       = (
        float(np.mean([f["psi_score"] for f in feature_results]))
        if feature_results else 0.0
    )

    # Compute approval-rate drift
    total     = len(recent_predictions)
    approved  = sum(1 for p in recent_predictions if p.get("prediction") == "APPROVED")
    cur_rate  = (approved / total * 100) if total else 0.0
    base_rate = 62.0   # from reference / training set

    from datetime import datetime, timezone
    return {
        "features_drifted":   len(high_drift) + len(moderate_drift),
        "total_features":     len(feature_results),
        "avg_psi":            round(avg_psi, 4),
        "ks_failures":        ks_failures,
        "concept_drift":      abs(cur_rate - base_rate) > 2.0,
        "label_shift_pct":    round(abs(cur_rate - base_rate), 2),
        "current_approval":   round(cur_rate, 2),
        "baseline_approval":  base_rate,
        "drift_delta":        round(cur_rate - base_rate, 2),
        "features":           feature_results,
        "analysed_at":        datetime.now(timezone.utc).isoformat(),
    }


def _empty_drift_summary() -> dict:
    from datetime import datetime, timezone
    return {
        "features_drifted": 0, "total_features": 0,
        "avg_psi": 0.0, "ks_failures": 0,
        "concept_drift": False, "label_shift_pct": 0.0,
        "current_approval": 62.0, "baseline_approval": 62.0,
        "drift_delta": 0.0, "features": [],
        "analysed_at": datetime.now(timezone.utc).isoformat(),
    }

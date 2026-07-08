"""
routes/drift.py — GET /drift
"""

from fastapi import APIRouter, Depends, Query
from database import get_db
from models.schemas import DriftSummary
from services.drift_service import run_full_drift_analysis
from services.seed_service import load_reference_stats

router = APIRouter()


@router.get("/drift", response_model=DriftSummary)
def get_drift(
    window: int  = Query(default=500, ge=10, le=5000),
    save:   bool = Query(default=True),
    conn         = Depends(get_db),
):
    from utils.model_loader import get_active_model_key
    active_key = get_active_model_key()

    cur = conn.execute(
        """SELECT loan_amount,annual_income,credit_score,applicant_age,
                  loan_tenure,prediction,confidence
           FROM prediction_logs WHERE model_key=? ORDER BY id DESC LIMIT ?""", (active_key, window)
    )
    recent_preds = [dict(r) for r in cur.fetchall()]

    reference_stats = load_reference_stats()
    drift_result    = run_full_drift_analysis(recent_preds, reference_stats)

    if save and drift_result["features"]:
        for feat in drift_result["features"]:
            conn.execute(
                """INSERT INTO drift_logs
                   (feature,psi_score,ks_stat,p_value,mean_delta,variance_delta,drift_status)
                   VALUES (?,?,?,?,?,?,?)""",
                (feat["feature"], feat["psi_score"], feat["ks_stat"], feat["p_value"],
                 feat["mean_delta_pct"], feat["var_delta_pct"], feat["status"]),
            )

    return DriftSummary(
        features_drifted  = drift_result["features_drifted"],
        total_features    = drift_result["total_features"],
        avg_psi           = drift_result["avg_psi"],
        ks_failures       = drift_result["ks_failures"],
        concept_drift     = drift_result["concept_drift"],
        label_shift_pct   = drift_result["label_shift_pct"],
        current_approval  = drift_result["current_approval"],
        baseline_approval = drift_result["baseline_approval"],
        drift_delta       = drift_result["drift_delta"],
        features          = drift_result["features"],
        analysed_at       = drift_result["analysed_at"],
    )

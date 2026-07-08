"""
routes/overview.py — GET /overview
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timezone
from database import get_db
from models.schemas import OverviewMetrics
from services.seed_service import load_reference_stats
from services.drift_service import run_full_drift_analysis
from services.risk_service import build_risk_summary
from utils.model_loader import get_active_model_key

router = APIRouter()


@router.get("/overview", response_model=OverviewMetrics)
def overview(conn=Depends(get_db)):
    active_key = get_active_model_key()

    total     = conn.execute("SELECT COUNT(*) FROM prediction_logs WHERE model_key=?", (active_key,)).fetchone()[0] or 0
    approved  = conn.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction='APPROVED' AND model_key=?", (active_key,)).fetchone()[0] or 0
    anomalies = conn.execute("SELECT COUNT(*) FROM prediction_logs WHERE confidence < 0.55 AND model_key=?", (active_key,)).fetchone()[0] or 0
    avg_conf  = conn.execute("SELECT AVG(confidence) FROM prediction_logs WHERE model_key=?", (active_key,)).fetchone()[0] or 0.847

    cur = conn.execute(
        """SELECT loan_amount,annual_income,credit_score,applicant_age,
                  loan_tenure,prediction,confidence
           FROM prediction_logs WHERE model_key=? ORDER BY id DESC LIMIT 500""",
        (active_key,)
    )
    recent_preds = [dict(r) for r in cur.fetchall()]

    reference_stats = load_reference_stats()
    drift_summary   = run_full_drift_analysis(recent_preds, reference_stats)
    risk_summary    = build_risk_summary(recent_preds, drift_summary)

    return OverviewMetrics(
        total_predictions   = total,
        approved            = approved,
        rejected            = total - approved,
        anomalies           = anomalies,
        avg_confidence      = round(float(avg_conf), 4),
        approval_rate       = round(approved / total * 100 if total else 62.0, 2),
        health_score        = risk_summary.get("health_score", 90.0),
        avg_psi             = drift_summary.get("avg_psi", 0.0),
        high_drift_features = sum(1 for f in drift_summary.get("features", []) if f["status"] == "HIGH"),
        risk_level          = risk_summary.get("overall_risk", "LOW"),
        last_updated        = datetime.now(timezone.utc).isoformat(),
    )

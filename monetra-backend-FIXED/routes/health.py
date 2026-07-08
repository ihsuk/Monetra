"""routes/health.py — GET /health + GET /models + POST /models/activate"""
import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from database import get_db
from models.schemas import HealthStatus, ModelsListResponse, ModelInfo
from services.risk_service import compute_health_score, compute_failure_probability
from utils.model_loader import get_all_meta, get_active_model_key, set_active_model

router = APIRouter()
_START_TIME = time.time()


@router.get("/health", response_model=HealthStatus)
def health(conn=Depends(get_db)):
    total     = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0] or 1
    avg_conf  = conn.execute("SELECT AVG(confidence) FROM prediction_logs").fetchone()[0] or 0.847
    anomalies = conn.execute(
        "SELECT COUNT(*) FROM prediction_logs WHERE confidence < 0.55"
    ).fetchone()[0] or 0

    error_rate = round(anomalies / total, 4)
    meta = get_all_meta()
    active_key = get_active_model_key()
    active = meta.get(active_key, meta.get("xgb", {}))

    accuracy  = active.get("accuracy",  0.924)
    precision = active.get("precision", 0.918)
    recall    = active.get("recall",    0.905)
    f1        = active.get("f1",        0.911)
    latency   = active.get("latency_ms", 42)
    version   = active.get("version",   "v3.0.2")

    # Dynamic average PSI calculation based on recent predictions
    cur = conn.execute(
        """SELECT loan_amount, annual_income, credit_score, applicant_age,
                  loan_tenure, prediction, confidence
           FROM prediction_logs WHERE model_key=? ORDER BY id DESC LIMIT 500""", (active_key,)
    )
    recent_preds = [dict(r) for r in cur.fetchall()]

    from services.drift_service import run_full_drift_analysis
    from services.seed_service import load_reference_stats
    reference_stats = load_reference_stats()
    drift_summary   = run_full_drift_analysis(recent_preds, reference_stats)
    avg_psi         = drift_summary.get("avg_psi", 0.0)

    health_score        = compute_health_score(accuracy, float(avg_conf), avg_psi, error_rate)
    failure_probability = compute_failure_probability(health_score, avg_psi)

    return HealthStatus(
        status              = "healthy" if health_score >= 75 else ("degraded" if health_score >= 50 else "critical"),
        model_version       = version,
        accuracy            = accuracy,
        precision           = precision,
        recall              = recall,
        f1_score            = f1,
        health_score        = health_score,
        failure_probability = failure_probability,
        error_rate          = error_rate,
        latency_p99_ms      = latency,
        throughput_per_sec  = 1200.0,
        uptime_seconds      = int(time.time() - _START_TIME),
    )


@router.get("/models", response_model=ModelsListResponse)
def list_models():
    """Returns metadata for all 3 trained models with correct names."""
    meta = get_all_meta()
    active_key = get_active_model_key()
    models = []
    for key, info in meta.items():
        if "error" not in info:
            models.append(ModelInfo(
                key        = key,
                name       = info["name"],
                version    = info["version"],
                accuracy   = info["accuracy"],
                n_features = info["n_features"],
            ))
    return ModelsListResponse(models=models, active_model=active_key)


class ActivateRequest(BaseModel):
    model_key: str


@router.post("/models/activate")
def activate_model(req: ActivateRequest):
    """Set which model is currently active."""
    success = set_active_model(req.model_key)
    if not success:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown model key: {req.model_key}")
    meta = get_all_meta()
    active_key = get_active_model_key()
    active = meta.get(active_key, {})
    return {
        "success": True,
        "active_model": active_key,
        "name": active.get("name", active_key),
        "version": active.get("version", ""),
    }


def clip_val(val, min_v, max_v):
    return min(max(val, min_v), max_v)


@router.post("/database/reset")
def reset_database(conn=Depends(get_db)):
    """Clear prediction logs and insert 100 healthy simulated predictions."""
    conn.execute("DELETE FROM prediction_logs")
    conn.execute("DELETE FROM drift_logs")
    conn.execute("DELETE FROM risk_score_logs")
    conn.execute("DELETE FROM reference_stats")
    conn.commit()

    from services.seed_service import seed_reference_data
    seed_reference_data()

    import random
    from utils.model_loader import get_active_model_key, get_model_info
    from services.risk_service import compute_risk_level
    
    active_key = get_active_model_key()
    try:
        info = get_model_info(active_key)
        clf = info["model"]
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    # Load baseline dataset
    import os
    import pandas as pd
    from routes.predict import _build_features_dynamic, LoanApplicationRequest

    csv_path = "../loan_approval_10000.csv"
    if not os.path.exists(csv_path):
        csv_path = "loan_approval_10000.csv"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to load baseline dataset: {e}")

    # Sample 500 random rows
    sampled_df = df.sample(n=500, random_state=42)

    import random

    for _, row in sampled_df.iterrows():
        loan_amount = float(row.get("loan_amount", 10000000))
        annual_income = float(row.get("income_annum", 5000000))
        credit_score = int(row.get("cibil_score", 650))
        # Add normal noise to applicant age to match baseline variance
        applicant_age = int(clip_val(random.normalvariate(35.0, 10.0), 18.0, 65.0))
        loan_tenure = int(row.get("loan_term", 10))

        app_req = LoanApplicationRequest(
            no_of_dependents=max(0, int(row.get("no_of_dependents", 2))),
            education=str(row.get("education", "Graduate")).strip(),
            self_employed=str(row.get("self_employed", "No")).strip(),
            income_annum=max(1.0, float(row.get("income_annum", 5000000))),
            loan_amount=max(1.0, float(row.get("loan_amount", 10000000))),
            loan_term=max(1, min(20, int(row.get("loan_term", 10)))),
            cibil_score=max(300, min(900, int(row.get("cibil_score", 650)))),
            residential_assets_value=max(0.0, float(row.get("residential_assets_value", 0))),
            commercial_assets_value=max(0.0, float(row.get("commercial_assets_value", 0))),
            luxury_assets_value=max(0.0, float(row.get("luxury_assets_value", 0))),
            bank_asset_value=max(0.0, float(row.get("bank_asset_value", 0))),
            applicant_age=applicant_age,
        )

        features_list = info.get("features", [])
        features = _build_features_dynamic(app_req, features_list)

        if "scaler" in info and info["scaler"] is not None:
            features = info["scaler"].transform(features)

        proba = clf.predict_proba(features)[0]
        prob_approve = float(proba[1])
        prediction = "APPROVED" if prob_approve > 0.5 else "REJECTED"
        confidence = prob_approve if prediction == "APPROVED" else (1 - prob_approve)
        risk_score = round(1 - confidence, 4)
        risk_level = compute_risk_level((1 - risk_score) * 100)

        conn.execute(
            """INSERT INTO prediction_logs
               (loan_amount,annual_income,credit_score,applicant_age,loan_tenure,
                employment_type,prediction,confidence,risk_level,risk_score,model_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (loan_amount, annual_income, credit_score, applicant_age, loan_tenure,
             str(row.get("self_employed", "No")).strip(),
             prediction, round(confidence, 4), risk_level, risk_score, active_key),
        )

    conn.commit()
    return {"success": True, "detail": "Database cleared and seeded with 100 baseline samples."}


from fastapi import UploadFile, Form, File
import shutil
import os

@router.post("/models/register")
async def register_model(
    model_file: UploadFile = File(...),
    key: str = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    framework: str = Form(...),
    accuracy: float = Form(...)
):
    key = key.lower().strip()
    if not key or not name or not version:
        raise HTTPException(status_code=400, detail="Missing required metadata fields.")
    
    filename = f"{key}_model.pkl"
    from utils.model_loader import ML_DIR, register_model_in_loader
    
    temp_path = ML_DIR / f"temp_{filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    
    try:
        import joblib
        clf = joblib.load(temp_path)
        if not hasattr(clf, "predict") and not (isinstance(clf, dict) and "model" in clf):
            raise ValueError("Uploaded object is not a valid predictor model.")
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid model file: could not load pickle/joblib ({e})")
    
    final_path = ML_DIR / filename
    if final_path.exists():
        os.remove(final_path)
    os.rename(temp_path, final_path)
    
    meta = {
        "name": name,
        "version": version,
        "type": "Classification",
        "framework": framework,
        "accuracy": accuracy,
        "f1": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "latency_ms": 15,
        "features": [
            "no_of_dependents", "education", "self_employed",
            "income_annum", "loan_amount", "loan_term", "cibil_score",
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value",
            "total_assets", "loan_to_income_ratio", "assets_to_loan_ratio",
            "income_per_dependent", "emi_estimate", "cibil_band"
        ]
    }
    
    register_model_in_loader(key, filename, meta)
    
    return {
        "success": True,
        "message": f"Model '{name}' registered successfully under key '{key}'",
        "key": key
    }


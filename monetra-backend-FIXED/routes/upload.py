"""
routes/upload.py — POST /upload-csv
====================================
Accepts a CSV file, runs predictions on each row using the active model,
stores all results in prediction_logs, and returns a summary.
This is the "live data feed" endpoint for the demo panel.
"""

import io
import csv
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from database import get_db
from utils.model_loader import get_model_info, get_active_model_key
from services.risk_service import compute_risk_level

router = APIRouter()


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _build_features_for_row(row: dict, model_key: str) -> np.ndarray:
    """Build feature array from a CSV row dict, matching the training feature order."""
    # Normalize column names (handle both cases)
    r = {k.strip().lower(): v for k, v in row.items()}

    no_dep = _safe_int(r.get("no_of_dependents", 0))
    edu_raw = str(r.get("education", "Graduate")).strip()
    edu = 0 if edu_raw.lower() in ("graduate", "0") else 1
    emp_raw = str(r.get("self_employed", "No")).strip()
    self_emp = 1 if emp_raw.lower() in ("yes", "1") else 0
    income = _safe_float(r.get("income_annum", 5000000))
    loan_amt = _safe_float(r.get("loan_amount", 10000000))
    loan_term = _safe_int(r.get("loan_term", 10))
    cibil = _safe_int(r.get("cibil_score", 650))
    res_asset = _safe_float(r.get("residential_assets_value", 0))
    com_asset = _safe_float(r.get("commercial_assets_value", 0))
    lux_asset = _safe_float(r.get("luxury_assets_value", 0))
    bank_asset = _safe_float(r.get("bank_asset_value", 0))

    total_assets = res_asset + com_asset + lux_asset + bank_asset
    lti = loan_amt / (income + 1)
    atl = total_assets / (loan_amt + 1)
    cibil_band = 0 if cibil <= 549 else 1 if cibil <= 649 else 2 if cibil <= 749 else 3
    ipd = income / (no_dep + 1)
    emi = loan_amt / (loan_term * 12 + 1)

    if model_key == "lasso":
        return np.array([[no_dep, edu, self_emp, income, loan_amt, loan_term, cibil,
                          res_asset, com_asset, lux_asset, bank_asset,
                          lti, total_assets, atl, cibil_band]])
    elif model_key == "rf":
        return np.array([[no_dep, edu, self_emp, income, loan_amt, loan_term, cibil,
                          res_asset, com_asset, lux_asset, bank_asset,
                          lti, total_assets, atl, cibil_band, ipd, emi]])
    else:  # xgb
        return np.array([[no_dep, edu, self_emp, income, loan_amt, loan_term, cibil,
                          res_asset, com_asset, lux_asset, bank_asset,
                          total_assets, lti, atl, ipd, emi, cibil_band]])


@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    model: str = Query(default="", description="Model to use: lasso | rf | xgb. Empty = active model."),
    conn=Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data rows.")

    model_key = model.lower() if model.lower() in ("lasso", "rf", "xgb") else get_active_model_key()

    try:
        info = get_model_info(model_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model '{model_key}': {e}")

    clf = info["model"]
    results = []
    n_approved = 0
    n_rejected = 0
    n_anomalies = 0
    total_conf = 0.0
    errors = 0

    for i, row in enumerate(rows):
        try:
            features = _build_features_for_row(row, model_key)

            if model_key == "xgb" and "scaler" in info:
                features = info["scaler"].transform(features)
                proba = clf.predict_proba(features)[0]
            else:
                proba = clf.predict_proba(features)[0]

            prob_approve = float(proba[1])
            prediction = "APPROVED" if prob_approve > 0.5 else "REJECTED"
            confidence = prob_approve if prediction == "APPROVED" else (1 - prob_approve)
            risk_score = round(1 - confidence, 4)
            risk_level = compute_risk_level((1 - risk_score) * 100)

            r = {k.strip().lower(): v for k, v in row.items()}
            age = _safe_int(r.get("applicant_age") or r.get("age"))
            if age == 0:
                age = 35  # default to baseline mean to prevent false drift alarms

            conn.execute(
                """INSERT INTO prediction_logs
                   (loan_amount,annual_income,credit_score,applicant_age,loan_tenure,
                    employment_type,prediction,confidence,risk_level,risk_score,model_key)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (_safe_float(r.get("loan_amount")),
                 _safe_float(r.get("income_annum")),
                 _safe_int(r.get("cibil_score")),
                 age,
                 _safe_int(r.get("loan_term")),
                 str(r.get("self_employed", "No")),
                 prediction, round(confidence, 4), risk_level, risk_score, model_key),
            )

            if prediction == "APPROVED":
                n_approved += 1
            else:
                n_rejected += 1
            if confidence < 0.55:
                n_anomalies += 1
            total_conf += confidence

            results.append({
                "row": i + 1,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
            })
        except Exception as e:
            errors += 1
            results.append({"row": i + 1, "error": str(e)})

    conn.commit()

    total_processed = len(results) - errors
    return {
        "status": "success",
        "model_used": model_key,
        "total_rows": len(rows),
        "processed": total_processed,
        "errors": errors,
        "approved": n_approved,
        "rejected": n_rejected,
        "anomalies": n_anomalies,
        "avg_confidence": round(total_conf / max(total_processed, 1), 4),
        "approval_rate": round(n_approved / max(total_processed, 1) * 100, 2),
        "sample_results": results[:10],
    }

"""
routes/predict.py — POST /predict
Supports all 3 models: lasso, rf, xgb (default).
Pass ?model=lasso | rf | xgb in the query string.
"""
import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from database import get_db
from models.schemas import LoanApplicationRequest, PredictionResponse
from utils.model_loader import get_model_info
from services.risk_service import compute_risk_level, infer_feature_contributions

router = APIRouter()

def _get_real_contributions(model_key: str, info: dict, features_array: np.ndarray, app_dict: dict) -> dict:
    try:
        # Get scaler and weights dynamically
        if model_key == "xgb":
            scaler = info.get("scaler")
            clf = info["model"]
            weights = clf.feature_importances_
        else:
            pipeline = info["model"]
            scaler = pipeline.named_steps.get("scaler")
            # Find the classifier step dynamically (the step that is not "scaler")
            clf_step_name = [name for name in pipeline.named_steps.keys() if name != "scaler"][0]
            clf = pipeline.named_steps[clf_step_name]
            
            if hasattr(clf, "coef_"):
                weights = clf.coef_[0]
            elif hasattr(clf, "feature_importances_"):
                weights = clf.feature_importances_
            else:
                weights = np.zeros(len(features_array[0]))
        
        # Scale the features
        if scaler is not None:
            features_scaled = scaler.transform(features_array)[0]
        else:
            features_scaled = features_array[0]
            
        # Determine feature names list dynamically based on number of weights / features
        num_features = len(weights)
        if num_features == 15:
            feature_names = [
                "no_of_dependents", "education", "self_employed",
                "income_annum", "loan_amount", "loan_term", "cibil_score",
                "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value",
                "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio", "cibil_band"
            ]
        elif num_features == 17:
            if "rf" in model_key or (hasattr(clf, "feature_importances_") and model_key != "xgb"):
                feature_names = [
                    "no_of_dependents", "education", "self_employed",
                    "income_annum", "loan_amount", "loan_term", "cibil_score",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value",
                    "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio",
                    "cibil_band", "income_per_dependent", "emi_estimate"
                ]
            else:
                feature_names = [
                    "no_of_dependents", "education", "self_employed",
                    "income_annum", "loan_amount", "loan_term", "cibil_score",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value",
                    "total_assets", "loan_to_income_ratio", "assets_to_loan_ratio",
                    "income_per_dependent", "emi_estimate", "cibil_band"
                ]
        elif num_features == 6:
            feature_names = [
                "loan_amount", "income_annum", "cibil_score",
                "applicant_age", "loan_term", "self_employed"
            ]
        else:
            feature_names = info.get("features", [])

        # Correlation directions for features (1: positive correlation with approval, -1: negative correlation)
        feature_directions = {
            "credit_score": 1, "cibil_score": 1, "cibil_band": 1,
            "annual_income": 1, "income_annum": 1, "income_per_dependent": 1,
            "residential_assets_value": 1, "commercial_assets_value": 1,
            "luxury_assets_value": 1, "bank_asset_value": 1,
            "total_assets": 1, "assets_to_loan_ratio": 1, "education": 1,
            "loan_amount": -1, "loan_to_income_ratio": -1,
            "no_of_dependents": -1, "self_employed": -1,
            "loan_tenure": -1, "loan_term": -1, "emi_estimate": -1
        }
 
        mapped_contrib = {
            "credit_score": 0.0,
            "annual_income": 0.0,
            "loan_amount": 0.0,
            "employment_type": 0.0,
            "loan_tenure": 0.0,
            "applicant_age": 0.0
        }

        is_tree = model_key in ("xgb", "rf") or hasattr(clf, "feature_importances_")

        for idx, name in enumerate(feature_names):
            if idx >= len(features_scaled) or idx >= len(weights):
                continue
            
            direction = 1
            if is_tree:
                direction = feature_directions.get(name, 1)

            val = float(direction * weights[idx] * features_scaled[idx])
            
            if name in ("credit_score", "cibil_score", "cibil_band"):
                mapped_contrib["credit_score"] += val
            elif name in ("annual_income", "income_annum", "income_per_dependent"):
                mapped_contrib["annual_income"] += val
            elif name in ("loan_amount", "loan_to_income_ratio", "assets_to_loan_ratio"):
                mapped_contrib["loan_amount"] += val
            elif name in ("employment_type", "self_employed", "education"):
                mapped_contrib["employment_type"] += val
            elif name in ("loan_tenure", "loan_term", "emi_estimate"):
                mapped_contrib["loan_tenure"] += val
            elif name in ("applicant_age", "no_of_dependents"):
                mapped_contrib["applicant_age"] += val

        # Normalise so sum of absolute values is 1.0
        total_abs = sum(abs(v) for v in mapped_contrib.values()) or 1.0
        return {k: round(v / total_abs, 4) for k, v in mapped_contrib.items()}
    except Exception as e:
        return infer_feature_contributions(app_dict)

def _build_features_lasso(app: LoanApplicationRequest) -> np.ndarray:
    """15 features — Lasso notebook exact order."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti   = app.loan_amount / (app.income_annum + 1)
    atl   = total_assets / (app.loan_amount + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        lti, total_assets, atl, cibil_band
    ]])

def _build_features_rf(app: LoanApplicationRequest) -> np.ndarray:
    """17 features — Random Forest notebook (adds income_per_dependent, emi_estimate)."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti  = app.loan_amount / (app.income_annum + 1)
    atl  = total_assets / (app.loan_amount + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    ipd  = app.income_annum / (app.no_of_dependents + 1)
    emi  = app.loan_amount / (app.loan_term * 12 + 1)
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        lti, total_assets, atl, cibil_band, ipd, emi
    ]])

def _build_features_xgb(app: LoanApplicationRequest) -> np.ndarray:
    """XGBoost features — all base + engineered, matching train_models.py order."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti  = app.loan_amount / (app.income_annum + 1)
    atl  = total_assets / (app.loan_amount + 1)
    ipd  = app.income_annum / (app.no_of_dependents + 1)
    emi  = app.loan_amount / (app.loan_term * 12 + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        total_assets, lti, atl, ipd, emi, cibil_band
    ]])

def _build_features_dynamic(app: LoanApplicationRequest, features_list: list) -> np.ndarray:
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti  = app.loan_amount / (app.income_annum + 1)
    atl  = total_assets / (app.loan_amount + 1)
    ipd  = app.income_annum / (app.no_of_dependents + 1)
    emi  = app.loan_amount / (app.loan_term * 12 + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    
    feat_dict = {
        "no_of_dependents": app.no_of_dependents,
        "education": edu,
        "self_employed": self_emp,
        "income_annum": app.income_annum,
        "loan_amount": app.loan_amount,
        "loan_term": app.loan_term,
        "cibil_score": app.cibil_score,
        "residential_assets_value": app.residential_assets_value,
        "commercial_assets_value": app.commercial_assets_value,
        "luxury_assets_value": app.luxury_assets_value,
        "bank_asset_value": app.bank_asset_value,
        "total_assets": total_assets,
        "loan_to_income_ratio": lti,
        "assets_to_loan_ratio": atl,
        "income_per_dependent": ipd,
        "emi_estimate": emi,
        "cibil_band": cibil_band
    }
    
    if not features_list:
        features_list = [
            "no_of_dependents", "education", "self_employed",
            "income_annum", "loan_amount", "loan_term", "cibil_score",
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value",
            "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio",
            "cibil_band", "income_per_dependent", "emi_estimate"
        ]
        
    return np.array([[feat_dict[name] for name in features_list if name in feat_dict]])

@router.post("/predict", response_model=PredictionResponse)
def predict(
    application: LoanApplicationRequest,
    model: str = Query(default=None, description="Model to use: lasso | rf | xgb. If omitted, uses active model."),
    conn = Depends(get_db)
):
    from utils.model_loader import get_active_model_key, MODEL_MAP
    
    # Use specified query parameter if it exists in MODEL_MAP, otherwise use active model
    model_key = model.lower() if (model and model.lower() in MODEL_MAP) else get_active_model_key()

    try:
        info = get_model_info(model_key)
        
        # Build features dynamically based on model's feature list
        features_list = info.get("features", [])
        features_raw = _build_features_dynamic(application, features_list)
        features = features_raw.copy()

        clf = info["model"]

        # Run inference
        if "scaler" in info and info["scaler"] is not None:
            features = info["scaler"].transform(features)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(features)[0]
        elif hasattr(clf, "predict"):
            pred_class = clf.predict(features)[0]
            prob_approve = 0.85 if pred_class == 1 else 0.15
            proba = [1 - prob_approve, prob_approve]
        else:
            raise ValueError(f"Model {model_key} doesn't support predict/predict_proba")

        prob_approve = float(proba[1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    prediction  = "APPROVED" if prob_approve > 0.5 else "REJECTED"
    confidence  = prob_approve if prediction == "APPROVED" else (1 - prob_approve)
    risk_score  = round(1 - confidence, 4)
    risk_level  = compute_risk_level((1 - risk_score) * 100)
    
    # Compute real mathematical contributions using raw features
    feature_weights = _get_real_contributions(model_key, info, features_raw, application.model_dump())

    cur = conn.execute(
        """INSERT INTO prediction_logs
           (loan_amount,annual_income,credit_score,applicant_age,loan_tenure,
            employment_type,prediction,confidence,risk_level,risk_score,model_key)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (application.loan_amount, application.income_annum, application.cibil_score,
         application.applicant_age or 35, application.loan_term, application.self_employed,
         prediction, round(confidence, 4), risk_level, risk_score, model_key),
    )
    prediction_id = cur.lastrowid

    return PredictionResponse(
        prediction      = prediction,
        confidence      = round(confidence, 4),
        confidence_pct  = f"{confidence * 100:.1f}%",
        risk_level      = risk_level,
        risk_score      = risk_score,
        feature_weights = feature_weights,
        model_version   = info.get("version", "v1.0"),
        prediction_id   = prediction_id,
    )

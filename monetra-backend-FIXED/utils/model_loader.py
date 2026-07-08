"""
utils/model_loader.py
=====================
Loads all 3 trained models: Lasso, Random Forest, XGBoost.
Tracks the currently active model (can be changed at runtime).
Also loads model_metadata.pkl for feature lists & real metrics.
"""
import joblib
import pickle
from pathlib import Path

ML_DIR = Path(__file__).parent.parent / "ml_models"
_cache = {}

MODEL_MAP = {
    "lasso": "lasso_model.pkl",
    "rf":    "rf_model.pkl",
    "xgb":  "xgb_model.pkl",
}

# Correct metadata matching actual pkl bundles
MODEL_META_DEFAULTS = {
    "lasso": {
        "name": "Lasso Logistic Regression", "version": "v1.0.0",
        "type": "Classification", "framework": "Scikit-learn",
        "accuracy": 0.9835, "f1": 0.9866, "precision": 0.984, "recall": 0.982,
        "latency_ms": 8,
    },
    "rf": {
        "name": "Random Forest Classifier", "version": "v2.0.0",
        "type": "Classification", "framework": "Scikit-learn",
        "accuracy": 0.992, "f1": 0.9936, "precision": 0.993, "recall": 0.991,
        "latency_ms": 24,
    },
    "xgb": {
        "name": "XGBoost Classifier", "version": "v3.0.2",
        "type": "Classification", "framework": "GradientBoosting",
        "accuracy": 0.992, "f1": 0.9936, "precision": 0.993, "recall": 0.991,
        "latency_ms": 42,
    },
}

# Try loading real metadata from train_models.py output
_metadata_file = ML_DIR / "model_metadata.pkl"
_real_metadata = {}
if _metadata_file.exists():
    try:
        with open(_metadata_file, "rb") as f:
            _real_metadata = pickle.load(f)
        # Update defaults with real trained metrics
        for key in _real_metadata:
            if key in MODEL_META_DEFAULTS and "metrics" in _real_metadata[key]:
                m = _real_metadata[key]["metrics"]
                MODEL_META_DEFAULTS[key]["accuracy"]  = m.get("accuracy",  MODEL_META_DEFAULTS[key]["accuracy"])
                MODEL_META_DEFAULTS[key]["f1"]        = m.get("f1",        MODEL_META_DEFAULTS[key]["f1"])
                MODEL_META_DEFAULTS[key]["precision"]  = m.get("precision", MODEL_META_DEFAULTS[key]["precision"])
                MODEL_META_DEFAULTS[key]["recall"]     = m.get("recall",    MODEL_META_DEFAULTS[key]["recall"])
    except Exception:
        pass

# ── Dynamic Registry Persistence ──────────────────────────────────────────────
import json
DYNAMIC_REGISTRY_FILE = ML_DIR / "dynamic_registry.json"

def load_dynamic_registry():
    if DYNAMIC_REGISTRY_FILE.exists():
        try:
            with open(DYNAMIC_REGISTRY_FILE, "r") as f:
                data = json.load(f)
                for key, val in data.get("map", {}).items():
                    MODEL_MAP[key] = val
                for key, val in data.get("defaults", {}).items():
                    MODEL_META_DEFAULTS[key] = val
        except Exception:
            pass

def register_model_in_loader(key: str, filename: str, meta: dict):
    key = key.lower()
    MODEL_MAP[key] = filename
    MODEL_META_DEFAULTS[key] = meta
    
    # Save to JSON
    try:
        data = {"map": {}, "defaults": {}}
        if DYNAMIC_REGISTRY_FILE.exists():
            with open(DYNAMIC_REGISTRY_FILE, "r") as f:
                data = json.load(f)
        data["map"][key] = filename
        data["defaults"][key] = meta
        with open(DYNAMIC_REGISTRY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# Load any dynamically registered models on boot
load_dynamic_registry()

_active_model_key = "xgb"


def set_active_model(key: str) -> bool:
    global _active_model_key
    key = key.lower()
    if key not in MODEL_MAP:
        return False
    _active_model_key = key
    return True


def get_active_model_key() -> str:
    return _active_model_key


def get_model_info(model_key: str = "xgb") -> dict:
    key = model_key.lower()
    if key not in MODEL_MAP:
        key = "xgb"
    if key not in _cache:
        path = ML_DIR / MODEL_MAP[key]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        raw = joblib.load(path)
        defaults = MODEL_META_DEFAULTS[key]

        # Get real feature list from metadata
        real_feats = _real_metadata.get(key, {}).get("features", [])

        if not real_feats:
            n_feats = None
            if isinstance(raw, dict) and "scaler" in raw and raw["scaler"] is not None:
                n_feats = raw["scaler"].n_features_in_
            elif hasattr(raw, "named_steps"):
                if "scaler" in raw.named_steps:
                    n_feats = raw.named_steps["scaler"].n_features_in_
                else:
                    clf_step_name = [name for name in raw.named_steps.keys() if name != "scaler"][0]
                    n_feats = raw.named_steps[clf_step_name].n_features_in_
            
            if n_feats == 15:
                real_feats = [
                    "no_of_dependents", "education", "self_employed",
                    "income_annum", "loan_amount", "loan_term", "cibil_score",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value",
                    "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio", "cibil_band"
                ]
            elif n_feats == 6:
                real_feats = [
                    "loan_amount", "income_annum", "cibil_score",
                    "applicant_age", "loan_term", "self_employed"
                ]
            else:
                is_rf = False
                if hasattr(raw, "named_steps"):
                    clf_step_name = [name for name in raw.named_steps.keys() if name != "scaler"][0]
                    clf_class_name = raw.named_steps[clf_step_name].__class__.__name__.lower()
                    if "forest" in clf_class_name:
                        is_rf = True
                if is_rf:
                    real_feats = [
                        "no_of_dependents", "education", "self_employed",
                        "income_annum", "loan_amount", "loan_term", "cibil_score",
                        "residential_assets_value", "commercial_assets_value",
                        "luxury_assets_value", "bank_asset_value",
                        "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio",
                        "cibil_band", "income_per_dependent", "emi_estimate"
                    ]
                else:
                    real_feats = [
                        "no_of_dependents", "education", "self_employed",
                        "income_annum", "loan_amount", "loan_term", "cibil_score",
                        "residential_assets_value", "commercial_assets_value",
                        "luxury_assets_value", "bank_asset_value",
                        "total_assets", "loan_to_income_ratio", "assets_to_loan_ratio",
                        "income_per_dependent", "emi_estimate", "cibil_band"
                    ]

        if isinstance(raw, dict) and "model" in raw:
            # XGBoost bundle: {model, scaler}
            raw.update(defaults)
            raw["features"] = real_feats
        else:
            # Pipeline models (Lasso, RF) — wrap them
            wrapped = {
                "model": raw,  # the sklearn Pipeline itself
            }
            wrapped.update(defaults)
            wrapped["features"] = real_feats
            raw = wrapped

        _cache[key] = raw
    return _cache[key]


def get_all_meta() -> dict:
    result = {}
    for key in MODEL_MAP:
        try:
            info = get_model_info(key)
            d = MODEL_META_DEFAULTS[key]
            result[key] = {
                "name":       info.get("name",       d["name"]),
                "version":    info.get("version",    d["version"]),
                "type":       info.get("type",       d["type"]),
                "framework":  info.get("framework",  d["framework"]),
                "accuracy":   info.get("accuracy",   d["accuracy"]),
                "f1":         info.get("f1",         d["f1"]),
                "precision":  info.get("precision",  d["precision"]),
                "recall":     info.get("recall",     d["recall"]),
                "latency_ms": info.get("latency_ms", d["latency_ms"]),
                "n_features": len(info.get("features", [])),
                "is_active":  key == _active_model_key,
            }
        except Exception as e:
            result[key] = {"error": str(e), "is_active": key == _active_model_key}
    return result

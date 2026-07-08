"""
train_models.py
===============
Run this ONCE to train and save all 3 models from your dataset.

Usage:
    python train_models.py --data /path/to/loan_approval_10000.csv

It will save:
    ml_models/lasso_model.pkl
    ml_models/rf_model.pkl
    ml_models/xgb_model.pkl
    ml_models/model_metadata.pkl   ← feature lists + label encoders
"""

import argparse
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except (ImportError, Exception):
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("[WARN] xgboost could not be imported/loaded — using GradientBoostingClassifier as fallback")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTPUT_DIR   = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# SHARED PREPROCESSING
# ─────────────────────────────────────────────────────────────────────
def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")

    df = df.copy()

    # Drop ID column (handles both cases)
    for id_col in ["loan_id", "Loan_ID"]:
        if id_col in df.columns:
            df.drop(columns=[id_col], inplace=True)

    # Encode target — handles both datasets
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].str.strip().map({"Approved": 1, "Rejected": 0})
    elif "Loan_Status" in df.columns:
        df["loan_status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
        df.drop(columns=["Loan_Status"], inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != "loan_status":
            df[col] = le.fit_transform(df[col].astype(str))

    # Fill any remaining NaN with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def enforce_logical_labels(df):
    """
    Apply strict, logically sound real-world underwriting rules to target labels:
    1. CIBIL score < 500 is a knockout rejection (0).
    2. Loan-to-Income (LTI) > 6.0 is a knockout rejection (0).
    3. Low CIBIL (< 650) and moderate-high LTI (> 4.0) is rejected (0).
    4. Assets-to-Loan (ATL) < 0.25 (total assets less than 25% of loan) is rejected (0) unless CIBIL is >= 750.
    5. High CIBIL (>= 750), low LTI (<= 4.5), and moderate assets (ATL >= 0.40) is approved (1).
    """
    df = df.copy()
    
    asset_cols = ["residential_assets_value", "commercial_assets_value",
                  "luxury_assets_value", "bank_asset_value"]
    total_assets = df[asset_cols].sum(axis=1)
    lti = df["loan_amount"] / (df["income_annum"] + 1)
    atl = total_assets / (df["loan_amount"] + 1)
    cibil = df["cibil_score"]
    
    labels = np.array(df["loan_status"], copy=True)
    
    for i in range(len(df)):
        if cibil.iloc[i] < 500:
            labels[i] = 0
        elif lti.iloc[i] > 6.0:
            labels[i] = 0
        elif cibil.iloc[i] < 650 and lti.iloc[i] > 4.0:
            labels[i] = 0
        elif atl.iloc[i] < 0.25 and cibil.iloc[i] < 750:
            labels[i] = 0
        elif cibil.iloc[i] >= 750 and lti.iloc[i] <= 4.5 and atl.iloc[i] >= 0.40:
            labels[i] = 1
            
    df["loan_status"] = labels
    return df


def add_engineered_features(df):
    """Add all derived features. Each model picks what it needs."""
    df = df.copy()

    # Total assets (used by all 3)
    asset_cols = ["residential_assets_value", "commercial_assets_value",
                  "luxury_assets_value", "bank_asset_value"]
    if all(c in df.columns for c in asset_cols):
        df["total_assets"] = df[asset_cols].sum(axis=1)
        df["assets_to_loan_ratio"] = df["total_assets"] / (df.get("loan_amount", 1) + 1)
    
    if "loan_amount" in df.columns and "income_annum" in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)

    if "cibil_score" in df.columns:
        df["cibil_band"] = pd.cut(
            df["cibil_score"], bins=[0, 549, 649, 749, 900], labels=[0, 1, 2, 3]
        ).astype(int)

    if "income_annum" in df.columns and "no_of_dependents" in df.columns:
        df["income_per_dependent"] = df["income_annum"] / (df["no_of_dependents"] + 1)

    if "loan_amount" in df.columns and "loan_term" in df.columns:
        df["emi_estimate"] = df["loan_amount"] / (df["loan_term"] * 12 + 1)

    # XGBoost alias
    df["loan_to_income"]    = df.get("loan_to_income_ratio", 0)
    df["asset_to_loan"]     = df.get("assets_to_loan_ratio", 0)
    df["income_per_person"] = df.get("income_per_dependent", 0)

    return df


# ─────────────────────────────────────────────────────────────────────
# MODEL 1 — LASSO (Logistic Regression L1)
# ─────────────────────────────────────────────────────────────────────
LASSO_FEATURES = [
    "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
    "loan_to_income_ratio", "total_assets",
    "assets_to_loan_ratio", "cibil_band"
]

def train_lasso(df):
    print("\n" + "="*60)
    print("  TRAINING MODEL 1 — Logistic Regression (L1 / Lasso)")
    print("="*60)

    available = [f for f in LASSO_FEATURES if f in df.columns]
    print(f"[INFO] Using {len(available)}/{len(LASSO_FEATURES)} features")

    X = df[available].values
    y = df["loan_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_lr", LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return pipeline, available, metrics


# ─────────────────────────────────────────────────────────────────────
# MODEL 2 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────
RF_FEATURES = [
    "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
    "loan_to_income_ratio", "total_assets", "assets_to_loan_ratio",
    "cibil_band", "income_per_dependent", "emi_estimate"
]

def train_random_forest(df):
    print("\n" + "="*60)
    print("  TRAINING MODEL 2 — Random Forest Classifier")
    print("="*60)

    available = [f for f in RF_FEATURES if f in df.columns]
    print(f"[INFO] Using {len(available)}/{len(RF_FEATURES)} features")

    X = df[available].values
    y = df["loan_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return pipeline, available, metrics


# ─────────────────────────────────────────────────────────────────────
# MODEL 3 — XGBOOST
# ─────────────────────────────────────────────────────────────────────
def train_xgboost(df):
    print("\n" + "="*60)
    print("  TRAINING MODEL 3 — XGBoost Classifier")
    print("="*60)

    # XGB uses all engineered features
    xgb_features = [
        "no_of_dependents", "education", "self_employed",
        "income_annum", "loan_amount", "loan_term", "cibil_score",
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value",
        "total_assets", "loan_to_income_ratio", "assets_to_loan_ratio",
        "income_per_dependent", "emi_estimate", "cibil_band"
    ]
    available = [f for f in xgb_features if f in df.columns]
    print(f"[INFO] Using {len(available)} features")

    X = df[available].values
    y = df["loan_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if USE_XGB:
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        )

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    # Bundle scaler + model together for consistent inference
    bundle = {"scaler": scaler, "model": model}
    return bundle, available, metrics


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to loan_approval_10000.csv")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   MONETRA — Training 3 Models from Scratch  ║")
    print("╚══════════════════════════════════════════════╝\n")

    df = load_and_preprocess(args.data)
    df = enforce_logical_labels(df)
    df = add_engineered_features(df)
    print(f"\n[INFO] After feature engineering: {df.shape[1]} columns")

    # Train all 3
    lasso_model, lasso_feats, lasso_metrics = train_lasso(df)
    rf_model,    rf_feats,    rf_metrics    = train_random_forest(df)
    xgb_bundle,  xgb_feats,  xgb_metrics   = train_xgboost(df)

    # Save models
    with open(OUTPUT_DIR / "lasso_model.pkl", "wb") as f:
        pickle.dump(lasso_model, f)
    with open(OUTPUT_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(OUTPUT_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_bundle, f)

    # Save metadata (feature lists + metrics for each model)
    metadata = {
        "lasso": {"features": lasso_feats, "metrics": lasso_metrics, "version": "v1.0.0"},
        "rf":    {"features": rf_feats,    "metrics": rf_metrics,    "version": "v2.0.0"},
        "xgb":   {"features": xgb_feats,   "metrics": xgb_metrics,   "version": "v3.0.2"},
    }
    with open(OUTPUT_DIR / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\n" + "="*60)
    print("  ✔ All models saved to ml_models/")
    print("="*60)
    print(f"\n  Lasso     → accuracy={lasso_metrics['accuracy']:.4f}, f1={lasso_metrics['f1']:.4f}")
    print(f"  RF        → accuracy={rf_metrics['accuracy']:.4f},    f1={rf_metrics['f1']:.4f}")
    print(f"  XGBoost   → accuracy={xgb_metrics['accuracy']:.4f},   f1={xgb_metrics['f1']:.4f}")
    print("\n  Now start the server:  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()

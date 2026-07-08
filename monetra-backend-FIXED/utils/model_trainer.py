"""
utils/model_trainer.py — Train & persist a baseline loan model
==============================================================
Run this script once to generate `ml_models/loan_model.pkl`.
Uses a GradientBoostingClassifier trained on synthetic data that
mimics the statistical profile visible in the Monetra frontend.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 10_000   # synthetic samples


def generate_synthetic_data(n: int = N):
    """
    Produce synthetic loan application data with realistic correlations.
    Approval logic mirrors what the frontend charts imply:
      - credit_score  → strongest signal
      - income / loan → second strongest
      - age, tenure   → minor effects
    """
    credit_score    = np.random.normal(680, 80, n).clip(300, 900)
    annual_income   = np.random.lognormal(13.5, 0.6, n).clip(100_000, 5_000_000)
    loan_amount     = annual_income * np.random.uniform(0.3, 1.8, n)
    applicant_age   = np.random.normal(35, 10, n).clip(18, 75).astype(int)
    loan_tenure     = np.random.choice([12, 24, 36, 48, 60], n)
    # encode employment: 0=salaried, 1=self_employed, 2=business
    employment_type = np.random.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15])

    # Approval probability formula (sigmoid-based)
    score = (
        0.40 * (credit_score   / 900)
      + 0.30 * np.minimum(annual_income / loan_amount, 3) / 3
      + 0.15 * (applicant_age  / 65)
      + 0.10 * (1 - employment_type * 0.15)
      + 0.05 * (1 - loan_tenure / 360)
    )
    noise     = np.random.normal(0, 0.05, n)
    prob      = 1 / (1 + np.exp(-8 * (score - 0.55) + noise))
    approved  = (prob > 0.5).astype(int)

    # Apply strict real-world logical knockout assumptions
    for i in range(n):
        lti = loan_amount[i] / (annual_income[i] + 1)
        if credit_score[i] < 500:
            approved[i] = 0
        elif lti > 6.0:
            approved[i] = 0
        elif credit_score[i] < 650 and lti > 4.0:
            approved[i] = 0
        elif credit_score[i] >= 750 and lti <= 4.5:
            approved[i] = 1

    X = np.column_stack([
        loan_amount, annual_income, credit_score,
        applicant_age, loan_tenure, employment_type
    ])
    return X, approved


def train_and_save(output_path: str = "ml_models/loan_model.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["REJECTED", "APPROVED"]))

    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")
    return pipeline


if __name__ == "__main__":
    train_and_save()

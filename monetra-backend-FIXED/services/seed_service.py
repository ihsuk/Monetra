"""
services/seed_service.py — First-run initialisation
====================================================
Called once at app startup.
• Trains and saves the ML model if it doesn't exist.
• Populates the reference_stats table from training data statistics.
"""

import os
import logging
from database import get_conn

logger = logging.getLogger(__name__)

MODEL_PATH = "ml_models/loan_model.pkl"

REFERENCE_BASELINE = {
    "loan_amount":   {"mean": 15135880.0, "std": 9055387.5, "min_val": 300000.0, "max_val": 39500000.0, "p25": 7700000.0, "p50": 14500000.0, "p75": 21400000.0},
    "annual_income": {"mean": 5069230.0,  "std": 2811789.0, "min_val": 200000.0, "max_val": 9900000.0,  "p25": 2600000.0, "p50": 5100000.0,  "p75": 7500000.0},
    "credit_score":  {"mean": 605.85,     "std": 170.69,    "min_val": 300.0,    "max_val": 900.0,      "p25": 461.0,     "p50": 609.0,      "p75": 753.25},
    "applicant_age": {"mean": 35.0,       "std": 10.0,      "min_val": 18.0,     "max_val": 75.0,       "p25": 27.0,      "p50": 34.0,       "p75": 43.0},
    "loan_tenure":   {"mean": 10.93,      "std": 5.62,      "min_val": 2.0,      "max_val": 20.0,       "p25": 6.0,       "p50": 10.0,       "p75": 16.0},
}


def seed_reference_data():
    conn = get_conn()
    cur  = conn.execute("SELECT COUNT(*) FROM reference_stats")
    if cur.fetchone()[0] > 0:
        logger.info("Reference stats already seeded. Skipping.")
        _ensure_model_exists()
        return

    logger.info("Seeding reference statistics …")
    for feature, stats in REFERENCE_BASELINE.items():
        conn.execute(
            "INSERT OR IGNORE INTO reference_stats (feature,mean,std,min_val,max_val,p25,p50,p75) VALUES (?,?,?,?,?,?,?,?)",
            (feature, stats["mean"], stats["std"], stats["min_val"], stats["max_val"], stats["p25"], stats["p50"], stats["p75"]),
        )
    conn.commit()
    logger.info("Reference stats seeded successfully.")
    _ensure_model_exists()


def _ensure_model_exists():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}.")
        return
    logger.info("No pre-trained model found. Training now …")
    try:
        from utils.model_trainer import train_and_save
        train_and_save(MODEL_PATH)
        logger.info("Model trained and saved.")
    except Exception as exc:
        logger.error(f"Model training failed: {exc}")


def load_reference_stats() -> dict:
    conn = get_conn()
    cur  = conn.execute("SELECT feature,mean,std,min_val,max_val,p25,p50,p75 FROM reference_stats")
    return {
        row["feature"]: {
            "mean": row["mean"], "std": row["std"],
            "min_val": row["min_val"], "max_val": row["max_val"],
            "p25": row["p25"], "p50": row["p50"], "p75": row["p75"],
        }
        for row in cur.fetchall()
    }

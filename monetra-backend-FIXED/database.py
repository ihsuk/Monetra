"""
database.py — Lightweight SQLite layer (stdlib only)
=====================================================
Uses Python's built-in sqlite3 module so the system runs with zero
additional installs beyond scikit-learn / scipy / numpy / joblib.

When FastAPI is available the get_db() generator is used as a
dependency-injection helper; the DB connection is safe to use
directly from service code via get_conn() as well.

To use PostgreSQL in production, swap sqlite3 for psycopg2 and
update the connection string — the SQL is ANSI-compatible.
"""

import sqlite3
import os
import threading

DATABASE_PATH = os.getenv("DATABASE_PATH", "monetra.db")

# One connection per thread (avoids cross-thread sharing issues)
_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row   # rows behave like dicts
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def get_db():
    """FastAPI dependency that yields a connection and commits on exit."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Create all tables idempotently."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reference_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            feature     TEXT UNIQUE NOT NULL,
            mean        REAL,
            std         REAL,
            min_val     REAL,
            max_val     REAL,
            p25         REAL,
            p50         REAL,
            p75         REAL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS prediction_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_amount     REAL,
            annual_income   REAL,
            credit_score    REAL,
            applicant_age   INTEGER,
            loan_tenure     INTEGER,
            employment_type TEXT,
            prediction      TEXT,
            confidence      REAL,
            risk_level      TEXT,
            risk_score      REAL,
            model_key       TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS drift_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            feature         TEXT,
            psi_score       REAL,
            ks_stat         REAL,
            p_value         REAL,
            mean_delta      REAL,
            variance_delta  REAL,
            drift_status    TEXT,
            analysed_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS risk_score_logs (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            overall_risk         TEXT,
            health_score         REAL,
            failure_probability  REAL,
            drift_contribution   REAL,
            latency_contribution REAL,
            bias_contribution    REAL,
            computed_at          TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

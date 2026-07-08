# Monetra — Backend API

Autonomous ML Model Monitoring & Failure Prediction for Loan Risk Models.  
Built with **FastAPI + SQLite + scikit-learn + RAG/LLM explainability**.

---

## Project Structure

```
monetra-backend/
│
├── main.py                     # FastAPI app entry point, CORS, router registration
├── database.py                 # SQLAlchemy engine, ORM models, session factory
├── requirements.txt
│
├── models/
│   └── schemas.py              # Pydantic request/response schemas
│
├── routes/
│   ├── predict.py              # POST /predict
│   ├── drift.py                # GET  /drift
│   ├── explain.py              # GET  /explain
│   ├── health.py               # GET  /health
│   └── overview.py             # GET  /overview
│
├── services/
│   ├── drift_service.py        # PSI + KS-test drift detection logic
│   ├── risk_service.py         # Health score + failure probability engine
│   ├── explain_service.py      # RAG retriever + LLM prompt builder + generator
│   └── seed_service.py         # First-run DB seeding + model training trigger
│
├── utils/
│   ├── model_loader.py         # Singleton model loader with fallback heuristic
│   ├── model_trainer.py        # Synthetic data generation + GBM training script
│   └── frontend_patch.py       # One-time HTML patcher to wire frontend → backend
│
├── ml_models/
│   └── loan_model.pkl          # Auto-generated on first run (GradientBoostingClassifier)
│
└── monetra.db                  # SQLite database (auto-created on first run)
```

---

## Quick Start

### 1 — Install dependencies

```bash
cd monetra-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — (Optional) Set OpenAI key for real LLM explanations

```bash
export OPENAI_API_KEY=sk-...     # skip to use the built-in mock generator
```

### 3 — Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first launch the server will:
1. Create `monetra.db` with all tables.
2. Seed the reference baseline statistics.
3. Train and save `ml_models/loan_model.pkl` (~10 seconds).

### 4 — Connect the frontend

```bash
python utils/frontend_patch.py path/to/monetra.html
# Opens monetra_live.html — open it in any browser
```

### 5 — Explore the API

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `POST /predict`

Run the loan risk model on a single application.

**Request**
```json
{
  "loan_amount":     500000,
  "annual_income":   800000,
  "credit_score":    720,
  "applicant_age":   32,
  "loan_tenure":     36,
  "employment_type": "salaried"
}
```

**Response**
```json
{
  "prediction":      "APPROVED",
  "confidence":      0.847,
  "confidence_pct":  "84.7%",
  "risk_level":      "LOW",
  "risk_score":      0.153,
  "feature_weights": {
    "credit_score":  0.38,
    "annual_income": 0.26,
    "loan_amount":   0.17,
    "applicant_age": 0.11,
    "loan_tenure":   0.05,
    "employment_type": 0.03
  },
  "model_version":   "v3.0.2",
  "prediction_id":   42
}
```

---

### `GET /drift?window=500`

Returns full data-drift analysis.

**Response**
```json
{
  "features_drifted":  4,
  "total_features":    5,
  "avg_psi":           0.41,
  "ks_failures":       2,
  "concept_drift":     true,
  "label_shift_pct":   2.8,
  "current_approval":  59.2,
  "baseline_approval": 62.0,
  "drift_delta":       -2.8,
  "features": [
    {
      "feature":        "loan_amount",
      "psi_score":      0.78,
      "ks_stat":        0.31,
      "p_value":        0.001,
      "mean_delta_pct": 18.4,
      "var_delta_pct":  22.1,
      "drift_type":     "Data Drift",
      "status":         "HIGH"
    }
  ],
  "analysed_at": "2024-11-15T10:30:00Z"
}
```

---

### `GET /explain?window=200`

RAG + LLM explanation of current model health.

**Response**
```json
{
  "summary": "The loan risk model is showing medium risk with a health score of 68/100. Significant data drift has been detected in loan_amount and income_band...",
  "root_causes": [
    {
      "cause": "Data drift — loan_amount",
      "contribution_pct": 45,
      "severity": "HIGH",
      "detail": "loan_amount PSI=0.78; incoming population shows a materially different distribution..."
    }
  ],
  "recommendations": [
    "Investigate distribution shift in loan_amount — check upstream data pipeline.",
    "Schedule model retraining on refreshed dataset.",
    "Set automated PSI alerts at 0.20."
  ],
  "risk_level":    "MEDIUM",
  "generated_at":  "2024-11-15T10:30:00Z",
  "context_used":  ["DRIFT-HIGH | feature=loan_amount psi=0.780 ...", "..."]
}
```

---

### `GET /health`

System-level health and model performance metrics.

**Response**
```json
{
  "status":               "healthy",
  "model_version":        "v3.0.2",
  "accuracy":             0.924,
  "precision":            0.918,
  "recall":               0.905,
  "f1_score":             0.911,
  "health_score":         87.0,
  "failure_probability":  0.23,
  "error_rate":           0.008,
  "latency_p99_ms":       42,
  "throughput_per_sec":   1200.0,
  "uptime_seconds":       3600
}
```

---

### `GET /overview`

Aggregated KPIs for the main dashboard.

**Response**
```json
{
  "total_predictions":   12540,
  "approved":            7421,
  "rejected":            5119,
  "anomalies":           38,
  "avg_confidence":      0.847,
  "approval_rate":       59.2,
  "health_score":        87.0,
  "avg_psi":             0.27,
  "high_drift_features": 2,
  "risk_level":          "MEDIUM",
  "last_updated":        "2024-11-15T10:30:00Z"
}
```

---

## Architecture

```
Browser (monetra.html)
       │  HTTP
       ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI (main.py)                  │
│  ┌────────────┐  ┌──────────┐  ┌───────────────────┐│
│  │  /predict  │  │  /drift  │  │ /explain /health  ││
│  └─────┬──────┘  └────┬─────┘  └────────┬──────────┘│
│        │              │                  │           │
│  ┌─────▼──────────────▼──────────────────▼─────────┐│
│  │                  Services                        ││
│  │  drift_service  risk_service  explain_service    ││
│  └─────┬──────────────┬──────────────────┬─────────┘│
│        │              │                  │           │
│  ┌─────▼──────┐ ┌─────▼──────┐  ┌───────▼─────────┐│
│  │ loan_model │ │  SQLite DB  │  │  LLM / Mock Gen ││
│  │   (.pkl)   │ │ (monetra.db)│  │  (RAG pipeline) ││
│  └────────────┘ └────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Drift Detection (PSI + KS)

| PSI Score | Status   | Action           |
|-----------|----------|------------------|
| < 0.10    | LOW      | Monitor          |
| 0.10–0.25 | MODERATE | Alert team       |
| > 0.25    | HIGH     | Retrain ASAP     |

### Health Score Formula

```
health = 0.35 × accuracy_score
       + 0.30 × (1 − PSI/0.25)
       + 0.20 × avg_confidence
       + 0.15 × (1 − error_rate×10)
```

### Risk Tiers

| Health Score | Risk Level | Frontend Colour |
|-------------|------------|-----------------|
| ≥ 75        | LOW        | 🟢 Green         |
| 50–74       | MEDIUM     | 🟡 Amber         |
| < 50        | HIGH       | 🔴 Red           |

---

## Configuration

| Variable        | Default               | Description                    |
|-----------------|-----------------------|--------------------------------|
| `DATABASE_URL`  | `sqlite:///monetra.db`| SQLAlchemy DB URL              |
| `OPENAI_API_KEY`| *(empty)*             | Set for real LLM explanations  |

---

## Production Notes

- Replace SQLite with PostgreSQL by setting `DATABASE_URL=postgresql://...`
- Hook `latency_p99_ms` and `throughput_per_sec` into a real APM (Datadog, Prometheus)
- Replace the heuristic `feature_weights` with SHAP values for production explainability
- Schedule `GET /drift` via a cron job or Celery beat every N minutes
- Add JWT authentication middleware before exposing endpoints publicly

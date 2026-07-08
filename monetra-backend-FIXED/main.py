"""
Monetra — ML Model Monitoring Backend
=====================================
Entry point for the FastAPI application.
Registers all routers and initialises the database on startup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db
from routes import predict, drift, explain, health, overview, upload
from services.seed_service import seed_reference_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    init_db()           # create all SQLite tables
    seed_reference_data()  # populate baseline stats if empty
    yield               # app runs here
    # (cleanup hooks go here if needed)


app = FastAPI(
    title="Monetra API",
    description="Autonomous ML Model Monitoring & Failure Prediction for Loan Risk Models",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the HTML frontend (served from any origin / file://) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router,    tags=["System"])
app.include_router(predict.router,   tags=["Inference"])
app.include_router(drift.router,     tags=["Drift"])
app.include_router(explain.router,   tags=["Explainability"])
app.include_router(overview.router,  tags=["Overview"])
app.include_router(upload.router,    tags=["Upload"])


@app.get("/", tags=["System"])
def root():
    return {"service": "Monetra API", "status": "online", "version": "1.0.0"}

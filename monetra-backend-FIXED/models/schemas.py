"""models/schemas.py — Pydantic schemas for all endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class LoanApplicationRequest(BaseModel):
    no_of_dependents:         int   = Field(default=2,        ge=0, le=10)
    education:                str   = Field(default="Graduate")
    self_employed:            str   = Field(default="No")
    income_annum:             float = Field(default=5000000,  gt=0)
    loan_amount:              float = Field(default=10000000, gt=0)
    loan_term:                int   = Field(default=10,       ge=1, le=20)
    cibil_score:              int   = Field(default=650,      ge=300, le=900)
    residential_assets_value: float = Field(default=2000000, ge=0)
    commercial_assets_value:  float = Field(default=0,       ge=0)
    luxury_assets_value:      float = Field(default=0,       ge=0)
    bank_asset_value:         float = Field(default=500000,  ge=0)
    # legacy compat
    annual_income:   Optional[float] = None
    credit_score:    Optional[int]   = None
    applicant_age:   Optional[int]   = None
    loan_tenure:     Optional[int]   = None
    employment_type: Optional[str]   = None

    def model_post_init(self, __context: Any) -> None:
        if self.annual_income  and self.income_annum == 5000000:
            object.__setattr__(self, 'income_annum', self.annual_income)
        if self.credit_score   and self.cibil_score  == 650:
            object.__setattr__(self, 'cibil_score',  self.credit_score)
        if self.loan_tenure    and self.loan_term     == 10:
            object.__setattr__(self, 'loan_term',    self.loan_tenure)


class PredictionResponse(BaseModel):
    prediction:      str
    confidence:      float
    confidence_pct:  str
    risk_level:      str
    risk_score:      float
    feature_weights: Dict[str, float]
    model_version:   str
    prediction_id:   int


class OverviewMetrics(BaseModel):
    total_predictions:   int
    approved:            int
    rejected:            int
    anomalies:           int
    avg_confidence:      float
    approval_rate:       float
    health_score:        float
    avg_psi:             float
    high_drift_features: int
    risk_level:          str
    last_updated:        str


class HealthStatus(BaseModel):
    status:              str
    model_version:       str
    accuracy:            float
    precision:           float
    recall:              float
    f1_score:            float
    health_score:        float
    failure_probability: float
    error_rate:          float
    latency_p99_ms:      int
    throughput_per_sec:  float
    uptime_seconds:      int


class DriftFeature(BaseModel):
    feature:        str
    psi_score:      float
    ks_stat:        float
    p_value:        float
    mean_delta_pct: float
    var_delta_pct:  float
    status:         str


class DriftSummary(BaseModel):
    features_drifted:  int
    total_features:    int
    avg_psi:           float
    ks_failures:       int
    concept_drift:     bool
    label_shift_pct:   float
    current_approval:  float
    baseline_approval: float
    drift_delta:       float
    features:          List[DriftFeature]
    analysed_at:       str


class RootCause(BaseModel):
    text: str


class ExplainResponse(BaseModel):
    summary:         str
    severity:        str = "LOW"
    root_causes:     List[Any]
    recommendations: List[str]
    risk_level:      str
    urgency:         str = "Monitor"
    generated_at:    str
    context_used:    Any
    source:          str = "rule-based"
    latency_ms:      int = 150
    token_count:     int = 800


class ModelInfo(BaseModel):
    key:        str
    name:       str
    version:    str
    accuracy:   float
    n_features: int


class ModelsListResponse(BaseModel):
    models:       List[ModelInfo]
    active_model: str

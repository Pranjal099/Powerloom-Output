"""FastAPI prediction service for G018 MP243 Power Loom Output.

Author: Pranjal (Service Lead) — S24CSEU1027
Structured JSON logging: Eshaan Mishra (Ops Lead) — S24CSEU0795
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:
    from pythonjsonlogger.jsonlogger import JsonFormatter
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("powerloom.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(
        JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_stdout_handler)


def _load_model_path() -> Path:
    """Resolve the champion model path from params.yaml."""
    params_file = ROOT / "params.yaml"
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        rel_path = params.get("paths", {}).get("model", "models/model.joblib")
        return ROOT / rel_path
    return ROOT / "models/model.joblib"


MODEL_PATH = _load_model_path()
model: Any = None


def get_model() -> Any:
    """Load the champion model from disk if not already cached."""
    global model
    if model is None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Preload the champion model on application startup."""
    get_model()
    yield


app = FastAPI(
    title="Power Loom Output Prediction Service",
    description="MP243 API predicting cloth output (metres_per_shift) for Guwahati Weavers' Cluster",
    version="1.0.0",
    lifespan=lifespan,
)


class LoomPredictionRequest(BaseModel):
    """Input feature schema with client business validation rules."""

    loom_age_yr: float
    thread_count: float
    power_cuts_min: float = Field(
        ..., ge=0, description="Power cuts in minutes (must be >= 0)"
    )
    weaver_exp_yr: float
    humidity_pct: float
    shift_hours: float = Field(
        ..., le=20, description="Shift duration in hours (must be <= 20)"
    )
    fabric: str
    shed: str


class LoomPredictionResponse(BaseModel):
    """Output prediction schema."""

    prediction: float


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Liveness and readiness health check endpoint."""
    loaded = get_model() is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=LoomPredictionResponse)
def predict(request: LoomPredictionRequest) -> LoomPredictionResponse:
    """Predict cloth woven per shift in metres."""
    loaded_model = get_model()
    if loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Champion model artifact not found at {MODEL_PATH}",
        )

    payload = request.model_dump()
    # Normalize categorical strings to match clean_train preprocessing
    payload["fabric"] = payload["fabric"].strip().lower()
    payload["shed"] = payload["shed"].strip().lower()

    input_df = pd.DataFrame([payload])
    try:
        pred_value = float(loaded_model.predict(input_df)[0])
    except Exception as exc:
        logger.error("Model inference failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {exc}"
        ) from exc

    prediction = round(pred_value, 4)
    logger.info(
        "Prediction generated",
        extra={
            "prediction": prediction,
            "fabric": payload["fabric"],
            "shed": payload["shed"],
            "shift_hours": payload["shift_hours"],
        },
    )
    return LoomPredictionResponse(prediction=prediction)

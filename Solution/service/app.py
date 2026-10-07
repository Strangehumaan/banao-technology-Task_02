"""Kestrel routing service: one JSON endpoint and one screen.

    uv run uvicorn service.app:app --port 8000      (from the Solution folder)

POST /route   one request record -> team, confidence, action, reasons
GET  /health  is a model loaded, and what it was validated at
GET  /        the screen that calls /route

No paid API is used. If the model file is missing the service still starts,
tries to train from the data pack, and otherwise answers 503 with a clear
message instead of crashing.
"""
from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from kestrel_router.explain import explain
from kestrel_router.train import MODEL_PATH, train_and_save

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("kestrel")
STATIC = Path(__file__).resolve().parent / "static"

STATE: dict = {"pipeline": None, "meta": None, "error": None}


def load_model() -> None:
    try:
        if not MODEL_PATH.exists():
            log.warning("No model at %s; training from the data pack (takes ~20 s)...", MODEL_PATH)
            train_and_save(verbose=False)
        bundle = joblib.load(MODEL_PATH)
        STATE.update(pipeline=bundle["pipeline"], meta=bundle["meta"], error=None)
    except FileNotFoundError as e:
        STATE.update(pipeline=None, meta=None, error=str(e))
        log.error("Model unavailable: %s", e)
    except Exception as e:  # keep the service up whatever happens
        STATE.update(pipeline=None, meta=None, error=f"{type(e).__name__}: {e}")
        log.exception("Model failed to load")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Kestrel request router", version="1.0", lifespan=lifespan)


class RequestIn(BaseModel):
    request_text: str = Field(..., min_length=3, max_length=2000,
                              examples=["water purifier leaking water from bottom"])
    channel: Literal["ivr", "chat", "whatsapp", "email"] = "chat"
    product_family: Literal["Water Purifier", "Air Fryer", "Mixer Grinder", "Induction Cooktop",
                            "Room Heater", "Ceiling Fan", "Robot Vacuum"] = "Water Purifier"
    warranty_status: Literal["in_warranty", "shield", "out_of_warranty"] = "in_warranty"
    request_id: str | None = None


@app.get("/health")
def health():
    meta = STATE["meta"] or {}
    return {
        "status": "ok" if STATE["pipeline"] is not None else "degraded",
        "model_loaded": STATE["pipeline"] is not None,
        "error": STATE["error"],
        "trained_at": meta.get("trained_at"),
        "holdout_accuracy": meta.get("holdout", {}).get("accuracy"),
        "clarify_threshold": meta.get("clarify_threshold"),
        "paid_api_calls": 0,
    }


@app.post("/route")
def route(req: RequestIn):
    if STATE["pipeline"] is None:
        return JSONResponse(status_code=503, content={
            "error": "The routing model is not available right now, so this request was not routed.",
            "what_to_do": "Route it manually for now. To fix: put the Kestrel data pack in the Data "
                          "folder and run `uv run python -m kestrel_router.train`, then restart.",
            "detail": STATE["error"],
        })
    record = pd.DataFrame([req.model_dump()])
    out = explain(STATE["pipeline"], record, STATE["meta"]["clarify_threshold"])
    return {"request_id": req.request_id, **out}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

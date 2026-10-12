"""Read-only serving of a versioned historical forecast, plus observed-error checks."""
import json
from pathlib import Path
from typing import Literal
from datetime import date
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict


class Observation(BaseModel):
    date: date
    value: float = Field(allow_inf_nan=False)
    model_config = ConfigDict(extra="forbid")


class MonitorRequest(BaseModel):
    source: Literal["GCAG", "GISTEMP"]
    observations: list[Observation] = Field(min_length=1, max_length=12)
    model_config = ConfigDict(extra="forbid")


def create_app(root: Path):
    root = Path(root).resolve()
    app = FastAPI(title="Climate Forecasting Lab", version="1.0.0", description="Versioned historical forecasts; as-of October 2016, not current climate predictions.")

    def artifacts():
        if not (root/"artifacts/manifest.json").exists():
            raise HTTPException(503, "Run the training pipeline first")
        manifest = json.loads((root/"artifacts/manifest.json").read_text())
        frame = pd.read_csv(root/"artifacts/forecast.csv")
        return manifest, frame

    @app.get("/health")
    def health():
        m, _ = artifacts()
        return {"status": "ready", "historical_snapshot": True, "data_sha256": m["raw_sha256"], "as_of": "2016-10-01"}

    @app.get("/")
    def dashboard():
        path = root/"reports/dashboard.html"
        if not path.exists():
            raise HTTPException(503, "Run the pipeline first")
        return FileResponse(path)

    @app.get("/forecast")
    def forecast(source: Literal["GCAG", "GISTEMP"] = "GISTEMP", months: int = Query(12, ge=1, le=12), level: int = Query(90)):
        if level not in (80, 90):
            raise HTTPException(422, "Supported interval levels are 80 and 90")
        m, frame = artifacts()
        rows = frame[(frame.source == source) & (frame.horizon <= months) & (frame.level == level/100)]
        return {"historical_snapshot": True, "data_sha256": m["raw_sha256"], "interval_note": "Empirically calibrated, no exact coverage guarantee under temporal dependence", "forecasts": rows.to_dict("records")}

    @app.get("/metrics")
    def metrics(source: Literal["GCAG", "GISTEMP"] = "GISTEMP"):
        m, _ = artifacts()
        return m["sources"][source]

    @app.post("/monitor")
    def monitor(payload: MonitorRequest):
        _, frame = artifacts()
        dates = [o.date.replace(day=1).isoformat() for o in payload.observations]
        if len(set(dates)) != len(dates):
            raise HTTPException(422, "Duplicate observation month")
        issued = frame[(frame.source == payload.source) & (frame.level == .9)].set_index("date")
        if any(d not in issued.index for d in dates):
            raise HTTPException(422, "Observation outside the issued forecast horizon")
        result = []
        for d, observation in zip(dates, payload.observations):
            row = issued.loc[d]
            result.append({"date": d, "actual": observation.value, "prediction": float(row.prediction),
                           "error": float(observation.value-row.prediction),
                           "outside_90_interval": bool(observation.value < row.lower or observation.value > row.upper)})
        return {"observations": result, "note": "An interval exceedance is a review flag, not proof of drift. Values are not persisted or used for training."}
    return app

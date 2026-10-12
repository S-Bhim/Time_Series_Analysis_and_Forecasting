import json
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from climate_ts.api import create_app


@pytest.fixture
def client(tmp_path):
    (tmp_path/"artifacts").mkdir()
    (tmp_path/"artifacts/manifest.json").write_text(json.dumps({"raw_sha256": "test-hash", "sources": {"GCAG": {}, "GISTEMP": {}}}))
    pd.DataFrame([dict(source=s, date="2016-11-01", horizon=1, prediction=.7, lower=.4, upper=1., level=.9)
                  for s in ["GCAG", "GISTEMP"]]).to_csv(tmp_path/"artifacts/forecast.csv", index=False)
    return TestClient(create_app(tmp_path))


def test_api_input_validation_and_snapshot(client):
    assert client.get("/health").json()["historical_snapshot"]
    assert client.get("/forecast?months=13").status_code == 422
    assert client.get("/forecast?source=invalid").status_code == 422
    assert client.get("/forecast?level=95").status_code == 422
    out = client.get("/forecast?source=GCAG&months=1&level=90").json()
    assert len(out["forecasts"]) == 1
    assert out["forecasts"][0]["prediction"] == .7
    assert client.get("/forecast?level=80").status_code == 200
    assert client.get("/forecast?level=85").status_code == 422


def test_monitor_reports_interval_exceedances(client):
    payload = {"source": "GCAG", "observations": [{"date": "2016-11-15", "value": 1.2}]}
    out = client.post("/monitor", json=payload)
    assert out.status_code == 200
    assert out.json()["observations"][0]["outside_90_interval"]
    assert out.json()["observations"][0]["error"] == pytest.approx(.5)
    payload["observations"] *= 2
    assert client.post("/monitor", json=payload).status_code == 422
    payload["observations"] = [{"date": "2026-01-01", "value": 1.2}]
    assert client.post("/monitor", json=payload).status_code == 422


def test_missing_artifacts_are_not_healthy(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").status_code == 503
    assert client.get("/").status_code == 503

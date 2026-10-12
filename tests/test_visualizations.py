import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from climate_ts.data import load_data
from climate_ts.exploration import explore, CHARTS

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def prepared():
    return explore(load_data(ROOT/"data/raw/monthly_csv.csv")[0])

def test_eda_preserves_missing_months_and_excludes_partial_annual_mean(prepared):
    for source,d in prepared.items():
        assert np.isnan(d["heatmap"].loc[2016,11])
        assert np.isnan(d["heatmap"].loc[2016,12])
        assert d["annual"].index.max().year==2015
        assert d["heatmap"].loc[2016,10]==pytest.approx(d["series"].iloc[-1])

def test_rolling_summaries_are_trailing_and_have_minimum_history(prepared):
    d=prepared["GCAG"]
    assert d["mean12"].iloc[:11].isna().all()
    assert d["mean60"].iloc[:59].isna().all()
    assert d["mean12"].iloc[11]==pytest.approx(d["series"].iloc[:12].mean())
    assert d["mean60"].iloc[100]==pytest.approx(d["series"].iloc[41:101].mean())

def test_dependence_and_decomposition_are_complete(prepared):
    for d in prepared.values():
        assert len(d["acf"])==len(d["pacf"])==36
        assert np.isfinite(d["acf"]).all() and np.isfinite(d["pacf"]).all()
        np.testing.assert_allclose(d["stl"].trend+d["stl"].seasonal+d["stl"].resid,d["series"],atol=1e-12)

def test_all_candidate_forecasts_are_available_and_winner_matches():
    config=json.loads((ROOT/"config.json").read_text())
    f=pd.read_csv(ROOT/"artifacts/candidate_forecasts.csv")
    winner=pd.read_csv(ROOT/"artifacts/forecast.csv")
    assert len(f)==2*len(config["candidates"])*12
    assert not f.duplicated(["source","model","horizon"]).any()
    assert f.groupby(["source","model"]).size().eq(12).all()
    assert set(f.model)==set(config["candidates"])
    both=winner.merge(f,on=["source","model","date","horizon"],suffixes=("_winner","_candidate"))
    assert len(both)==len(winner)
    np.testing.assert_allclose(both.prediction_winner,both.prediction_candidate,atol=1e-12)

def test_dashboard_bundles_complete_offline_payload():
    html=(ROOT/"reports/dashboard.html").read_text(encoding="utf-8")
    payload=html.split('<script id="project-data" type="application/json">')[1].split('</script>')[0]
    d=json.loads(payload)
    assert set(d["exploration"])=={"GCAG","GISTEMP"}
    assert all(set(figs)==set(CHARTS) for figs in d["exploration"].values())
    assert len(d["candidate_forecasts"])==240
    assert '<script src=' not in html
    assert 'Choose models to compare' in html

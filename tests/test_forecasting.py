import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from climate_ts.models import feature_row, supervised, fit_forecast
from climate_ts.evaluation import split_origins, finite_quantile, calibration_widths, add_intervals, backtest, score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/"config.json").read_text())


def synthetic(n=720):
    rng = np.random.default_rng(12)
    return pd.Series(np.arange(n)*.001 + np.sin(np.arange(n)*2*np.pi/12)*.1 + rng.normal(0,.02,n),
                     index=pd.date_range("1900-01-01", periods=n, freq="MS"))


def test_chronological_blocks_are_disjoint_and_cover_all_targets():
    origins = split_origins(1642, CONFIG)
    phases = [set(t for o in origins[p] for t in range(o,o+12)) for p in ("selection", "calibration", "test")]
    assert len(phases[0]) == 180 and len(phases[1]) == 120 and len(phases[2]) == 60
    assert max(phases[0]) < min(phases[1]) and max(phases[1]) < min(phases[2])
    assert max(phases[2]) == 1641


def test_features_exclude_the_target_and_all_future_values():
    series = synthetic(72)
    X, _ = supervised(series)
    poisoned = series.copy()
    poisoned.iloc[40:] = 99999
    X_poisoned, _ = supervised(poisoned)
    np.testing.assert_array_equal(X[:17], X_poisoned[:17])  # targets 24..40 have identical histories
    assert X[16, 0] == series.iloc[39]
    assert X[16, 6] == pytest.approx(series.iloc[37:40].mean())


@pytest.mark.parametrize("model", ["naive", "seasonal_naive", "drift", "ets_damped", "arima_111", "sarima_111_100", "ridge_1", "boosting_3"])
def test_candidate_forecasts_are_finite_and_have_correct_horizon(model):
    pred, _, _ = fit_forecast(model, synthetic(120), 12)
    assert pred.shape == (12,) and np.isfinite(pred).all()


def test_backtest_does_not_consume_future_truth(monkeypatch):
    seen = []
    def recorder(name, train, horizon, seed):
        seen.append(train.copy())
        return np.repeat(train.iloc[-1], horizon), None, []
    monkeypatch.setattr("climate_ts.evaluation.fit_forecast", recorder)
    original = synthetic()
    altered = original.copy()
    altered.iloc[600:] = -999
    a, _ = backtest(original, "GCAG", ["naive"], [600], "test", CONFIG)
    b, _ = backtest(altered, "GCAG", ["naive"], [600], "test", CONFIG)
    pd.testing.assert_series_equal(seen[0], seen[1])
    np.testing.assert_array_equal(a.prediction, b.prediction)
    assert not np.array_equal(a.actual, b.actual)
    assert seen[0].index.max() < a.date.min()


def test_finite_sample_quantile_and_insufficient_calibration():
    assert finite_quantile(range(1,11), .9) == 10
    assert finite_quantile(range(1,11), .8) == 9
    with pytest.raises(ValueError):
        finite_quantile(range(1,11), .95)


def test_intervals_use_calibration_errors_only():
    cal = pd.DataFrame({"horizon": [1]*10, "error": np.arange(1,11)})
    widths = calibration_widths(cal, [.8,.9])
    issued = pd.DataFrame({"horizon": [1], "prediction": [4.0], "actual": [1000.]})
    out = add_intervals(issued, widths)
    assert out.lower.tolist() == [-5.,-6.]
    assert out.upper.tolist() == [13.,14.]
    assert not out.covered.any()


def test_metrics_handle_zero_and_negative_anomalies():
    frame = pd.DataFrame({"source": ["A"]*3, "phase": ["test"]*3, "model": ["m"]*3,
        "origin": ["x"]*3, "error": [-1.,0.,1.], "mase_scale": [2.]*3})
    result = score(frame).iloc[0]
    assert result.MAE == pytest.approx(2/3)
    assert result.RMSE == pytest.approx(np.sqrt(2/3))
    assert result.MASE == pytest.approx(1/3)

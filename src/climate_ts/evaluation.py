"""Disjoint chronological selection, calibration and evaluation blocks."""
import math
import numpy as np
import pandas as pd
from .models import fit_forecast


def split_origins(n, config):
    h = config["horizon"]
    if h != 12:
        raise ValueError("Backtest protocol requires 12-month blocks")
    sizes = [config[k] for k in ("selection_months", "calibration_months", "test_months")]
    if any(v <= 0 or v % h for v in sizes) or n-sum(sizes) < config["train_window"]:
        raise ValueError("Invalid split sizes or insufficient history")
    selection, calibration, test = sizes
    test_start = n-test
    cal_start = test_start-calibration
    dev_start = cal_start-selection
    return {"selection": list(range(dev_start, cal_start, h)),
            "calibration": list(range(cal_start, test_start, h)),
            "test": list(range(test_start, n, h))}


def backtest(series, source, names, origins, phase, config):
    rows, runs = [], []
    for origin in origins:
        train = series.iloc[max(0, origin-config["train_window"]):origin]
        target = series.iloc[origin:origin+config["horizon"]]
        scale = float(np.mean(np.abs(train.to_numpy()[12:]-train.to_numpy()[:-12])))
        if scale <= 0:
            raise ValueError("Seasonal MASE scale is zero")
        for name in names:
            try:
                prediction, _, notes = fit_forecast(name, train, len(target), config["seed"])
                runs.append(dict(source=source, phase=phase, model=name, origin=str(train.index[-1].date()),
                                 train_start=str(train.index[0].date()), n_train=len(train), status="ok", message="; ".join(notes)))
                for h, (date, actual, pred) in enumerate(zip(target.index, target, prediction), 1):
                    rows.append(dict(source=source, phase=phase, model=name, origin=train.index[-1],
                                     date=date, horizon=h, actual=float(actual), prediction=float(pred),
                                     error=float(actual-pred), mase_scale=scale))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                runs.append(dict(source=source, phase=phase, model=name, origin=str(train.index[-1].date()),
                                 train_start=str(train.index[0].date()), n_train=len(train), status="failed", message=str(exc)))
    return pd.DataFrame(rows), pd.DataFrame(runs)


def score(predictions, groups=("source", "phase", "model")):
    records = []
    for keys, g in predictions.groupby(list(groups)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        e = g.error.to_numpy()
        records.append(dict(zip(groups, keys)) | {"MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt(np.mean(e**2))), "MASE": float((g.error.abs()/g.mase_scale).mean()),
            "bias": float(-e.mean()), "n": len(g), "origins": g.origin.nunique()})
    return pd.DataFrame(records)


def finite_quantile(errors, level):
    a = np.sort(np.abs(np.asarray(errors, dtype=float)))
    if len(a) == 0 or not np.isfinite(a).all() or not 0 < level < 1:
        raise ValueError("Invalid errors or interval level")
    rank = math.ceil((len(a)+1)*level)
    if rank > len(a):
        raise ValueError("Too few calibration origins for this interval level")
    return float(a[rank-1])


def calibration_widths(calibration, levels):
    return pd.DataFrame([{"horizon": h, "level": level, "radius": finite_quantile(g.error, level), "n_calibration": len(g)}
                         for h, g in calibration.groupby("horizon") for level in levels])


def add_intervals(predictions, widths):
    out = predictions.merge(widths, on="horizon", how="left", validate="many_to_many")
    out["lower"] = out.prediction-out.radius
    out["upper"] = out.prediction+out.radius
    if "actual" in out:
        out["covered"] = (out.actual >= out.lower) & (out.actual <= out.upper)
        alpha = 1-out.level
        out["interval_score"] = 2*out.radius + 2/alpha*((out.lower-out.actual).clip(lower=0)+(out.actual-out.upper).clip(lower=0))
    return out


def paired_block_bootstrap(predictions, champion, seed=42):
    """Resample annual forecast-origin blocks, retaining dependence within each year."""
    per_origin = predictions.assign(ae=predictions.error.abs()).groupby(["origin", "model"]).ae.mean().unstack()
    difference = (per_origin.seasonal_naive-per_origin[champion]).to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, (5000, len(difference)), replace=True).mean(axis=1)
    return {"mae_gain_vs_seasonal_naive": float(difference.mean()),
            "bootstrap_lower_95": float(np.quantile(means, .025)), "bootstrap_upper_95": float(np.quantile(means, .975)),
            "blocks": len(difference), "note": "Exploratory paired annual-block bootstrap; only five test blocks, weak uncertainty resolution"}

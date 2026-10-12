import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from .data import load_data, series_for
from .models import fit_forecast, FEATURE_NAMES
from .evaluation import split_origins, backtest, score, calibration_widths, add_intervals, paired_block_bootstrap
from .diagnostics import analyse_series, error_diagnostics


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")


def run(root):
    root = Path(root).resolve()
    started = time.perf_counter()
    config = json.loads((root/"config.json").read_text())
    clean, quarantine, audit = load_data(root/"data/raw/monthly_csv.csv")
    processed = root/"data/processed"
    artifacts = root/"artifacts"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed/"monthly_clean.csv", index=False)
    quarantine.to_csv(processed/"quarantine.csv", index=False)
    write_json(artifacts/"data_audit.json", audit)
    from .exploration import explore, export_visualizations
    print("Generating exploratory visualizations from audited observations", flush=True)
    prepared = explore(clean)
    export_visualizations(root, clean, prepared)
    all_predictions, all_runs, forecasts, interval_rows, widths_rows, decomposition = [], [], [], [], [], []
    candidate_forecasts, final_fit_log = [], []
    findings = {}
    for source in sorted(clean.source.unique()):
        series = series_for(clean, source)
        origins = split_origins(len(series), config)
        print(f"{source}: selecting among {len(config['candidates'])} candidates on {len(origins['selection'])} origins", flush=True)
        selection, logs = backtest(series, source, config["candidates"], origins["selection"], "selection", config)
        all_runs.append(logs)
        leaderboard = score(selection)
        eligible = leaderboard[leaderboard.n == config["selection_months"]]
        if eligible.empty:
            raise RuntimeError("No candidate completed every selection fold")
        champion = eligible.sort_values(["MAE", "RMSE", "model"]).iloc[0].model
        print(f"{source}: locked champion {champion}; calibrating intervals and evaluating holdout", flush=True)
        calibration, logs = backtest(series, source, [champion], origins["calibration"], "calibration", config)
        all_runs.append(logs)
        if len(calibration) != config["calibration_months"]:
            raise RuntimeError("Incomplete calibration for champion")
        widths = calibration_widths(calibration, config["interval_levels"])
        widths_rows.append(widths.assign(source=source, model=champion))
        test, logs = backtest(series, source, config["candidates"], origins["test"], "test", config)
        all_runs.append(logs)
        champion_test = test[test.model == champion]
        if len(champion_test) != config["test_months"] or len(test[test.model == "seasonal_naive"]) != config["test_months"]:
            raise RuntimeError("Incomplete holdout for champion or seasonal baseline")
        interval_test = add_intervals(champion_test, widths)
        interval_rows.append(interval_test)
        # Monitoring threshold is calibrated BEFORE the test period.
        interval_test["outside_interval"] = ~interval_test.covered
        fit_series = series.iloc[-config["train_window"]:]
        prediction, fitted_model, notes = fit_forecast(champion, fit_series, config["horizon"], config["seed"])
        joblib.dump({"model_name": champion, "estimator": fitted_model, "history": fit_series,
                     "raw_sha256": audit["raw_sha256"], "config": config}, artifacts/f"{source.lower()}_model.joblib")
        if champion.startswith("ridge_"):
            pd.DataFrame({"feature": FEATURE_NAMES, "coefficient_per_train_std": fitted_model[-1].coef_}).to_csv(
                artifacts/f"{source.lower()}_ridge_coefficients.csv", index=False)
        dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=config["horizon"], freq="MS")
        future = pd.DataFrame({"source": source, "model": champion, "origin": series.index[-1], "date": dates,
                                "horizon": np.arange(1, config["horizon"]+1), "prediction": prediction})
        forecasts.append(add_intervals(future, widths))
        for candidate in config["candidates"]:
            if candidate == champion:
                candidate_prediction, candidate_notes = prediction, notes
            else:
                candidate_prediction, _, candidate_notes = fit_forecast(candidate, fit_series, config["horizon"], config["seed"])
            candidate_forecasts.append(pd.DataFrame({"source": source, "model": candidate,
                "origin": series.index[-1], "date": dates, "horizon": np.arange(1, config["horizon"]+1),
                "prediction": candidate_prediction}))
            final_fit_log.append({"source": source, "model": candidate, "status": "ok", "warnings": "; ".join(candidate_notes)})
        diagnostic, decomp = analyse_series(series)
        decomposition.append(decomp.assign(source=source))
        coverage = interval_test.groupby("level").agg(coverage=("covered", "mean"), mean_interval_score=("interval_score", "mean"), mean_radius=("radius", "mean"))
        test_scores = score(test)
        winner_score = test_scores[test_scores.model == champion].iloc[0]
        baseline_score = test_scores[test_scores.model == "seasonal_naive"].iloc[0]
        findings[source] = {"champion": champion, "selection_MAE": float(eligible[eligible.model == champion].iloc[0].MAE),
            "test_MAE": float(winner_score.MAE), "test_RMSE": float(winner_score.RMSE), "test_MASE": float(winner_score.MASE),
            "seasonal_naive_test_MAE": float(baseline_score.MAE),
            "MAE_improvement_percent": float(100*(1-winner_score.MAE/baseline_score.MAE)),
            "interval_evaluation": coverage.reset_index().to_dict("records"), "inference": diagnostic,
            "forecast_error_diagnostics": error_diagnostics(champion_test),
            "paired_comparison": paired_block_bootstrap(test, champion, config["seed"]),
            "production_fit_warnings": notes,
            "splits": {phase: {"first_origin": str(series.index[o[0]-1].date()),
                "target_start": str(series.index[o[0]].date()), "target_end": str(series.index[o[-1]+11].date()),
                "origins": len(o)} for phase, o in origins.items()}}
        all_predictions.extend([selection, calibration, test])
        write_json(artifacts/f"{source.lower()}_model_card.json", findings[source])
    predictions = pd.concat(all_predictions, ignore_index=True)
    runs = pd.concat(all_runs, ignore_index=True)
    future = pd.concat(forecasts, ignore_index=True)
    interval_test = pd.concat(interval_rows, ignore_index=True)
    tables = {"predictions": predictions, "scores": score(predictions),
              "horizon_scores": score(predictions, ("source", "phase", "model", "horizon")),
              "run_log": runs, "forecast": future, "interval_evaluation": interval_test,
              "calibration_widths": pd.concat(widths_rows, ignore_index=True),
              "decomposition": pd.concat(decomposition, ignore_index=True),
              "candidate_forecasts": pd.concat(candidate_forecasts, ignore_index=True),
              "final_fit_log": pd.DataFrame(final_fit_log),
              "descriptive_statistics": clean.groupby("source").value.describe().reset_index()}
    for name, frame in tables.items():
        frame.to_csv(artifacts/f"{name}.csv", index=False)
    with sqlite3.connect(artifacts/"experiments.sqlite") as conn:
        for name, frame in {"observations": clean, "quarantine": quarantine, **tables}.items():
            frame.to_sql(name, conn, if_exists="replace", index=False)
    aligned = clean.pivot(index="date", columns="source", values="value")
    comparison = {"level_correlation": float(aligned.corr().iloc[0, 1]),
        "difference_correlation": float(aligned.diff().dropna().corr().iloc[0, 1]),
        "mean_GCAG_minus_GISTEMP": float((aligned.GCAG-aligned.GISTEMP).mean()),
        "note": "Sources have distinct reference baselines and shared observational information. They are not independent replications; do not pool raw anomalies."}
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "plotly", "fastapi"]}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.perf_counter()-started,
        "python": platform.python_version(), "versions": versions, "config": config,
        "config_sha256": hashlib.sha256((root/"config.json").read_bytes()).hexdigest(),
        "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/"src/climate_ts").iterdir()) if p.suffix in (".py", ".html")},
        "raw_sha256": audit["raw_sha256"], "fit_count": len(runs), "failed_fits": int((runs.status == "failed").sum()),
        "final_candidate_fits": len(final_fit_log), "eda_chart_types": 11, "exported_eda_figures": 15,
        "sources": findings, "source_comparison": comparison,
        "warning": "Historical research snapshot, not a current climate forecast or emissions scenario projection"}
    write_json(artifacts/"manifest.json", manifest)
    from .report import build_report
    build_report(root, clean, audit, manifest, tables, prepared)
    print(f"Finished in {manifest['duration_seconds']:.1f}s. Open reports/dashboard.html", flush=True)
    return manifest

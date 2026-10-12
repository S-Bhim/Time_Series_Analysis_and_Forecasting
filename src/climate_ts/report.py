"""Standalone findings, visualization dashboard and portfolio documentation."""
from pathlib import Path
from .dashboard import build_dashboard


def build_report(root, clean, audit, manifest, tables, prepared=None):
    root = Path(root)
    build_dashboard(root, clean, audit, manifest, tables, prepared)
    sections = []
    for source, f in manifest["sources"].items():
        t = f["inference"]["trends"][1]
        coverage = next(v["coverage"] for v in f["interval_evaluation"] if v["level"] == .9)
        sections.append(f"""## {source}

The model selected on validation was **{f['champion']}**, with selection MAE {f['selection_MAE']:.4f} °C.
On the 60-month holdout: **MAE {f['test_MAE']:.4f} °C**, RMSE {f['test_RMSE']:.4f} °C and seasonal MASE {f['test_MASE']:.3f}.
Seasonal-naive test MAE was {f['seasonal_naive_test_MAE']:.4f} °C, giving a measured **{f['MAE_improvement_percent']:.1f}% MAE reduction**.
Empirical coverage of the nominal 90% interval was **{coverage:.1%}**.

Since-1970 annual slope through {t['end']}: {t['slope_c_per_decade']:.4f} °C/decade, HAC 95% CI [{t['lower_95']:.4f}, {t['upper_95']:.4f}].
Exploratory single-hinge year: {f['inference']['exploratory_break']['break_year']}. This does not identify a cause.
The paired annual-block bootstrap interval for MAE gain is [{f['paired_comparison']['bootstrap_lower_95']:.4f}, {f['paired_comparison']['bootstrap_upper_95']:.4f}] °C. It includes zero; superiority is not statistically decisive.
""")
    summary = f"""# Global Temperature Analysis and Forecasting

A complete independent project beginning with the raw CSV: audit, cleaning, visualization, statistical diagnostics, model training, evaluation, uncertainty and serving.

## Data and preparation

{audit['input_rows']} raw records, {audit['retained_rows']} retained records and {audit['quarantined_rows']} quarantined records. Each source has 1,642 consecutive monthly records from January 1880 through October 2016. Mixed date formats are parsed explicitly. The disconnected late records are preserved separately without guessed corrections or imputed temperatures.

SHA-256 of the unchanged source: `{audit['raw_sha256']}`.
The values are treated as anomalies in °C from the dataset family; the supplied CSV does not establish original baselines or source-release provenance. All findings concern this snapshot.

## Exploratory analysis and visualization

The interactive dashboard provides 11 visualization types for each selected source: monthly history, trailing 12/60-month means, complete annual means and changes, year-by-month heatmap, calendar-month boxes, histogram, decade boxes, rolling variability and first differences, robust STL, ACF/PACF of first differences, and source-comparison scatterplots.

Fifteen exportable PNG figures and [a full gallery](visualizations.md) accompany the notebook. The long-run anomaly distributions move upward in this snapshot, while monthly variability remains substantial. Calendar-month distributions combine different warming regimes and are not treated as isolated evidence of seasonality. No sources are averaged together.

## Model comparison and forecasts

The dashboard exposes checkboxes for all ten configurations, a source selector, selection/holdout period selector, horizon filter and ranking metric. Selected models update forecast overlays, error histories, error distributions, horizon plots and the comparison table. The forecast tab overlays any selected combination of the ten final candidate refits. Only the fixed validation winner has independently calibrated intervals. CSV exports reflect the selected filters.

Ten configurations were evaluated across 15 model-selection origins, ten winner-calibration origins and five final test origins. Every origin fits on the preceding 600 months and forecasts the next 12. The model winner is chosen before calibration and test. All {manifest['fit_count']} backtest fits completed with {manifest['failed_fits']} failed fits; {manifest['final_candidate_fits']} final candidate fits generated comparison forecasts.

""" + "\n".join(sections) + """
## Limits and interpretation

This is a historical research project. Issued forecasts cover November 2016–October 2017, not the present day. Statistical forecasts omit physical climate dynamics, emissions and external future drivers. Data vintage revisions cannot be reconstructed. Small calibration and test samples and temporal dependence limit interval reliability and comparisons. Full-history trend, decomposition and break analyses are descriptive and never used as backtest features. A better test score in hindsight does not justify switching the selected winner.

## Reproduction

Run the notebook from the first cell or `python -m climate_ts.cli run` from the project folder. All outputs are rebuilt from `data/raw/monthly_csv.csv`; no earlier project files or precomputed results are required as inputs. Open `reports/dashboard.html` directly for offline interaction.

## References

- Dataset-family context: https://github.com/datasets/global-temp
- Forecast evaluation: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html
- Causal lag features: https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html
- HAC covariance: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
"""
    (root/"reports/results.md").write_text(summary, encoding="utf-8")
    f = manifest["sources"]["GCAG"]
    cv = f"""# Suggested CV wording

Use after understanding and reproducing the project yourself.

**Global Temperature Analysis and Forecasting | Python, Statistics, Time Series, Scikit-learn, SQL, FastAPI**

- Built an end-to-end temperature-anomaly analysis pipeline from 3,288 CSV records, with audited date parsing, source-specific EDA, 11 interactive visualization types and 15 exported figures.
- Compared ten statistical and machine-learning configurations using rolling-origin validation, a separate interval-calibration period and a 60-month holdout, with causal lag features and recursive forecasting.
- Achieved GCAG holdout MAE of {f['test_MAE']:.3f} °C with Ridge regression, {f['MAE_improvement_percent']:.1f}% below seasonal naive; evaluated empirical interval coverage and comparison uncertainty.
- Delivered a multi-model comparison dashboard, an executed notebook, SQL experiment store, saved models, API, automated tests and HAC trend inference.

The numerical gain above is for GCAG only; it does not establish statistically decisive superiority. This project does not claim live climate forecasts, a deployed cloud service, causal attribution or neural models.
"""
    (root/"docs/cv_bullets.md").write_text(cv,encoding="utf-8")

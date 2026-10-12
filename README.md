# Global Temperature Analysis and Forecasting

A complete standalone project built from the supplied monthly global temperature CSV. It starts with raw-data inspection and cleaning, develops a full exploratory visualization analysis, then performs statistical inference, forecasting, model comparison, uncertainty calibration and local serving. No earlier project, notebook or saved experiment is required to reproduce it.

**Start with [the interactive dashboard](https://s-bhim.github.io/Time_Series_Analysis_and_Forecasting/reports/dashboard.html), [the executed notebook](notebooks/01_end_to_end.ipynb), or [the results report](reports/results.md).** The HTML dashboard is self-contained and works offline. No installation is needed to view it in a browser.

## What is included

- An unchanged raw CSV, deterministic date parsing, continuity checks and a separate quarantine table.
- Independent GCAG and GISTEMP modelling. Their anomaly baselines are not assumed interchangeable.
- Eleven interactive EDA visualization types per selected source and 15 exported PNG figures: monthly history, moving averages, complete annual summaries, heatmaps, monthly and decade distributions, variability, first differences, robust STL, ACF/PACF and source comparison.
- Ten candidate configurations: naive, seasonal naive, drift, damped-trend exponential smoothing, ARIMA, SARIMA, two Ridge penalties and two gradient-boosting depths.
- Fifteen rolling selection origins, ten interval-calibration origins and five holdout origins. Each fit uses the preceding 50 years to forecast the next 12 months.
- MAE, RMSE, seasonal MASE, signed bias, horizon-specific scores, interval coverage and interval scores.
- HAC trend inference, ADF/KPSS stationarity checks, robust STL, exploratory structural-break analysis and a paired annual-block bootstrap comparison.
- Reusable Python modules, an executed Jupyter notebook, an offline dashboard, SQLite tables and SQL examples, saved model objects, a FastAPI service, tests and a CI configuration.

## Use the dashboard

Open `reports/dashboard.html` in a browser. If you already have an older view open, refresh or reopen the file.

1. **Data & visualizations:** start with all 11 exploratory views. Switch GCAG/GISTEMP, inspect the data-quality table, zoom historical charts and use legends to focus on individual traces.
2. **Compare models:** check any combination of the ten model boxes. Use Select all, Suggested comparison or Clear selection. Selections update the prediction overlays, ranking, horizon errors, error histories, distributions and metrics table together.
3. Choose selection or holdout targets, filter the maximum included horizon, and rank by MAE, RMSE or MASE. Download the filtered comparison as CSV.
4. **Forecasts:** the same model selections overlay independently fitted forecasts from October 2016. Choose 1–12 months. Toggle the fixed validation winner's 80% or 90% interval. Other candidates have point forecasts only; the interface explains when the winner is not selected.
5. **Method & results:** review the chronological design and interpretation limits. Exploring alternatives in the UI never changes the saved validation winner.

The standalone HTML includes all plotting code and data. It does not require an API connection or internet access. Static images are collected in [the visualization gallery](reports/visualizations.md).

## Results from the supplied snapshot

| Source | Locked model | Test MAE (°C) | MAE reduction vs seasonal naive | 90% interval coverage |
|---|---|---:|---:|---:|
| GCAG | Ridge, alpha 1 | 0.0982 | 17.2% | 86.7% |
| GISTEMP | Ridge, alpha 1 | 0.1109 | 3.9% | 91.7% |

These are historical test results on November 2011–October 2016. Selection was completed before November 2001. The selected model need not be the best model in hindsight: SARIMA has lower GISTEMP test MAE, but switching to it after inspecting that test would invalidate the selection protocol. Both bootstrap intervals for improvement include zero, so these gains do not establish statistically decisive superiority.

The original CSV has 3,288 rows. Four disconnected records are quarantined; 3,284 retained rows cover January 1880–October 2016. The issued forecasts cover November 2016–October 2017. **This is a historical research project, not a current climate forecast.** The supplied snapshot's original release and anomaly reference periods are unverified.

## Run locally

Use Python 3.12. Open a terminal in this folder.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m climate_ts.cli audit
python -m climate_ts.cli run
python -m pytest -q
python -m climate_ts.cli serve
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell activation is unavailable, invoke `.venv\Scripts\python.exe` directly instead of changing system policy. The training run took roughly a minute in the tested environment, but hardware and library versions affect timing.

If Node.js is installed, `node tests/dashboard_logic.test.cjs` runs the separate offline dashboard logic tests. Those tests verify model selection, filters, metric arithmetic and CSV exports against a DOM/plotting stub; they do not claim browser rendering validation.

For the exact tested package versions, install `requirements-lock.txt` before the editable project. That lock records the Python 3.12 environment used here; different platforms may need compatible versions. `artifacts/manifest.json` records the data hash, code hashes, configuration, package versions and experiment duration.

After serving, open [the local dashboard](http://127.0.0.1:8000) and [API documentation](http://127.0.0.1:8000/docs). The API serves the checked-in forecast snapshot; it does not train a model on each request.

```text
GET /health
GET /forecast?source=GISTEMP&months=12&level=90
GET /metrics?source=GCAG
POST /monitor
```

Example monitoring body:

```json
{"source":"GISTEMP","observations":[{"date":"2016-11-01","value":0.8}]}
```

The monitoring value above is an illustrative input, not an asserted observation. The endpoint compares supplied observations with the previously issued interval, validates dates and reports exceedances. It does not persist data, retrain, or declare a formal distribution shift.

## Notebook and study guide

Open `notebooks/01_end_to_end.ipynb` in Jupyter or VS Code and select the environment above. Run from the first cell: it reads the raw CSV, performs its own audit and EDA, creates the visualization gallery, computes diagnostics, defines the experiment, trains the complete pipeline and evaluates results. It has 43 cells, including 22 executed code cells. Precomputed experiment files are outputs, not prerequisites. For Google Colab, upload and extract the project ZIP, install the package, and change the working directory to the project root before running cells.

Read [methodology](docs/methodology.md) before presenting the project. [Interview preparation](docs/interview_guide.md) explains the main decisions. [CV bullets](docs/cv_bullets.md) contain evidence-based wording to use after you understand and can reproduce the work.

## Project layout

```text
data/raw/               Original input, preserved byte-for-byte
data/processed/         Monthly modelling records and quarantine
src/climate_ts/         Data, EDA, models, evaluation, diagnostics, dashboard, API
config.json             Fixed evaluation and model configuration
notebooks/              Executed end-to-end walkthrough
artifacts/              Predictions, scores, model cards, models, logs, SQLite store
reports/                Offline dashboard, findings and visualization gallery
reports/figures/        Fifteen exported EDA figures
docs/                   Methodology, SQL queries, interview guide and CV wording
tests/                  Data, leakage, model, interval and API checks
.github/workflows/      Automated test configuration
```

## Reproduction and deployment boundaries

The pipeline overwrites generated artifacts in this project folder. Copy an experiment folder before changing its configuration if you want to preserve a previous run. Raw inputs are never rewritten. The current ingestion policy and report are designed for this supplied two-source snapshot; replacing it with a new dataset requires reviewing the quarantine rule, time ranges and provenance.

Saved `.joblib` files contain fitted estimators and their training history. Load only trusted local files, using compatible package versions. They support inspection and reuse; public API endpoints serve safe precomputed CSV/JSON outputs.

A Dockerfile is provided as a deployment starting point (`docker build -t climate-ts .`, then `docker run --rm -p 8000:8000 climate-ts`). Docker and hosted CI are not claimed as executed in this environment. No cloud service has been published. Add authentication, monitoring, artifact lifecycle management and a genuine updated data source before operational deployment.

The underlying data license was not included with the supplied file. Verify redistribution rights and original provenance before making a public repository. No personal CV contact information is included in this project.

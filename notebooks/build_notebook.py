"""Generate the standalone notebook. Run from any directory after installing dev dependencies."""
from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# Global Temperature Analysis and Forecasting
## Complete project: raw data → visual analysis → inference → modelling → evaluation → application

This notebook starts from `data/raw/monthly_csv.csv` and builds the entire project. **No previous notebook or precomputed model result is required.** Its workflow is data inspection, date auditing, cleaning, exploratory visualizations, statistical diagnostics, chronological experiment design, model training, uncertainty evaluation and serving.

Research questions:
1. How do the two reported anomaly series change over time?
2. Which forecast configurations generalize under a chronological evaluation?
3. How reliable are calibrated prediction intervals and inferred trends?

The CSV is a historical snapshot. Retained observations end in October 2016, so final forecasts cover November 2016–October 2017. Baselines and the original release metadata are not supplied. Treat the results as a study of this dataset.

Install with `python -m pip install -e ".[dev]"`, then run all cells from the project root or notebook folder. The full training cell runs all models and regenerates reports. Saved notebook outputs are also included.''')
code('''%matplotlib inline
from pathlib import Path
import sys, json, sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "config.json").exists(): ROOT = ROOT.parent
assert (ROOT / "data/raw/monthly_csv.csv").exists(), "Open from the project or notebooks folder"
sys.path.insert(0, str(ROOT / "src"))
from climate_ts.data import load_data, series_for
from climate_ts.exploration import explore, export_visualizations
from climate_ts.diagnostics import analyse_series
from climate_ts.models import supervised, fit_forecast, FEATURE_NAMES
from climate_ts.evaluation import split_origins, finite_quantile
config = json.loads((ROOT / "config.json").read_text())
pd.set_option("display.max_columns", 20)
plt.rcParams.update({"figure.figsize": (12, 4), "axes.spines.top": False, "axes.spines.right": False})''')
md('''## 1. Load and inspect the original CSV

Before parsing or modelling, examine the raw column types, first and last records, counts, missingness and numerical range. `Source` identifies a separate series. `Date` contains mixed date formats. `Mean` is treated as an anomaly in °C from the dataset family, not absolute global temperature.''')
code('''raw = pd.read_csv(ROOT / "data/raw/monthly_csv.csv")
display(raw.head(8))
display(raw.tail(8))
display(pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing": raw.isna().sum()}))
display(raw.groupby("Source").Mean.describe())
print("Raw shape:", raw.shape)''')
md('''## 2. Audit dates and prepare the modelling table

Parse ISO strings as year-month-day and the other accepted format as day-month-year. A blanket mixed-date parser with `dayfirst=True` can misread older ISO dates. Reduce parsed dates to monthly keys, sort within source and reject duplicate source-month keys.

The input has a disconnected tail after 11 missing months. Preserve those records in quarantine instead of guessing dates. Retain the uninterrupted prefix, with no interpolation or modification of temperature values. Raw file hashing makes provenance auditable.''')
code('''clean, quarantine, audit = load_data(ROOT / "data/raw/monthly_csv.csv")
display(pd.Series({k:audit[k] for k in ["input_rows","retained_rows","quarantined_rows","start","end","format_counts"]}))
display(quarantine)
display(clean[clean.raw_date.isin(["1899-12-06","06-12-1900"])][["source","raw_date","date","value"]])
print("Raw SHA-256:", audit["raw_sha256"])
assert len(clean) + len(quarantine) == len(raw)
display(clean.groupby("source").value.describe())''')
md('''## 3. Exploratory data analysis and visualizations

Start with a single source so reference-baseline differences do not get averaged away. Both sources are available in the exported gallery and the interactive dashboard. The following cell computes every EDA summary directly from `clean`, then exports 15 publication-sized PNG figures. The plotting implementation is in `src/climate_ts/exploration.py`.

Change `SOURCE` to inspect the other source. This choice affects the walkthrough examples; the final pipeline always evaluates both sources.''')
code('''SOURCE = "GISTEMP"
series = series_for(clean, SOURCE)
prepared = explore(clean)
figure_index = export_visualizations(ROOT, clean, prepared)
prefix = SOURCE.lower()
print("Exported EDA figures:", len(figure_index))
def show_figure(name):
    display(Image(filename=str(ROOT / "reports/figures" / f"{prefix}_{name}.png")))''')
md('''### 3.1 Monthly trends, smoothing and annual means

The monthly series shows short-run noise. Trailing 12- and 60-month means help examine the slower pattern without centered windows. Only full 12-month years enter annual aggregation; the incomplete final year is excluded from annual trend inference.''')
code('''show_figure("history")
display(prepared[SOURCE]["annual"].tail().rename("complete_annual_mean"))''')
md('''### 3.2 Year-by-month heatmap

Each row is a year and each column a calendar month. The color scale is centered at zero within the source. Read horizontal and vertical patterns separately: a warming shift across years is different from calendar seasonality. Unavailable cells are blank.''')
code('''show_figure("heatmap")
display(prepared[SOURCE]["heatmap"].tail(3))''')
md('''### 3.3 Distribution and decade shifts

The histogram describes the pooled historical observations. Decade boxes reveal how that pooled distribution combines regimes with different levels. Do not interpret it as a stationary probability distribution or use a whole-history outlier rule to delete recent high anomalies.''')
code('''show_figure("distributions")''')
md('''### 3.4 Calendar-month distributions

These boxes contain observations across the full history. Differences in their locations can reflect the mix of warming regimes as well as a monthly effect. Global anomaly series should not be assumed to have the seasonal cycle of local absolute temperatures.''')
code('''show_figure("months")''')
md('''### 3.5 Variability and first differences

Trailing 12-month standard deviation summarizes local variation. First differences show changes from one month to the next. These views help distinguish changes in level from short-term variation.''')
code('''show_figure("variability")''')
md('''### 3.6 Robust STL decomposition

Decompose observations into a smooth trend, a 12-month seasonal component and a remainder. Full-history STL uses later data to estimate earlier components, so it is **descriptive only** and never enters the forecast backtest features.''')
code('''show_figure("decomposition")''')
md('''### 3.7 Serial dependence: ACF and PACF

These diagnostics use first differences and 36 monthly lags. Dotted lines are approximate 95% bounds. Isolated exceedances across many lags should not be treated as automatic model-order decisions.''')
code('''show_figure("autocorrelation")''')
md('''### 3.8 Compare sources in levels and changes

Level correlation is affected by the common trend. First-difference correlation describes co-movement in monthly changes. Distinct baselines and shared underlying observations prevent treating the two sources as independent replications.''')
code('''display(Image(filename=str(ROOT / "reports/figures/both sources_comparison.png")))
wide = clean.pivot(index="date", columns="source", values="value")
display(pd.DataFrame({"level_correlation":[wide.corr().iloc[0,1]],
                      "difference_correlation":[wide.diff().dropna().corr().iloc[0,1]],
                      "mean_GCAG_minus_GISTEMP":[(wide.GCAG-wide.GISTEMP).mean()]}))''')
md('''## 4. Statistical inference and stationarity

OLS slopes use complete annual means and HAC covariance with three annual lags. The full-history and since-1970 windows are descriptive choices, not causal designs. ADF has a unit-root null; KPSS has a stationarity null. Their assumptions and approximate p-value bounds matter.

An exploratory continuous-hinge search considers years with at least 20 observations on each side. The best year by BIC is a descriptive location, with search uncertainty omitted.''')
code('''diagnostic, decomposition = analyse_series(series)
display(pd.DataFrame(diagnostic["trends"]))
display(pd.DataFrame(diagnostic["stationarity"]))
print("STL seasonal strength:", diagnostic["seasonal_strength"])
display(pd.Series(diagnostic["exploratory_break"]))
print("Stationarity warnings:", diagnostic["stationarity_notes"])''')
md('''## 5. Design the experiment before training

Each model uses a rolling 600-month training window and forecasts the next 12 months. Fifteen annual origins select the configuration, ten subsequent origins calibrate the fixed winner's intervals, and five final origins evaluate it. Target blocks do not overlap.

Earlier test observations may enter later annual refits once those observations would be available. Hyperparameters and the chosen configuration remain fixed. This evaluates annual updating, not a single five-year forecast.''')
code('''origins = split_origins(len(series), config)
split_rows=[]
for phase, positions in origins.items():
    split_rows.append({"phase":phase,"origins":len(positions),"target_start":series.index[positions[0]],"target_end":series.index[positions[-1]+11]})
display(pd.DataFrame(split_rows))
display(pd.Series(config))''')
md('''## 6. Build causal features and compare model families

The ten fixed configurations cover naive, seasonal naive, drift, damped ETS, ARIMA, seasonal ARIMA, two Ridge penalties and two boosting depths. Model complexity is evaluated rather than assumed beneficial.

For machine learning, lags and rolling summaries end at t−1. Calendar terms and time index are known at the forecast origin. Scaling is learned only within the training fold. Later multi-step features use earlier predictions recursively.''')
code('''origin = origins["selection"][0]
train = series.iloc[origin-config["train_window"]:origin]
X, y = supervised(train)
display(pd.DataFrame(X, columns=FEATURE_NAMES).head())
demo_prediction, demo_model, notes = fit_forecast("ridge_1", train, 12, config["seed"])
display(pd.DataFrame({"date":series.index[origin:origin+12],"prediction":demo_prediction,
                      "actual_for_scoring_only":series.iloc[origin:origin+12].to_numpy()}))''')
md('''## 7. Train and run the complete pipeline

This cell starts from the raw CSV and regenerates all data audit, EDA, training, calibration, holdout, model, SQL, dashboard and report artifacts. It requires no existing results. It overwrites generated files within this project folder. Copy the folder before changing the experiment if you need to preserve an earlier run.''')
code('''from climate_ts.pipeline import run
manifest = run(ROOT)
assert manifest["raw_sha256"] == audit["raw_sha256"]
print("Backtest fits:", manifest["fit_count"], "Failed fits:", manifest["failed_fits"])
print("Final model refits:", manifest["final_candidate_fits"])''')
md('''## 8. Evaluate selection and holdout results

MAE and RMSE are in °C. Seasonal MASE uses the origin's training seasonal-difference scale. Positive bias means overprediction. MAPE is avoided because anomaly values can cross zero. Keep the winner chosen before testing even when another model performs better in hindsight.''')
code('''scores = pd.read_csv(ROOT / "artifacts/scores.csv")
display(scores[(scores.source == SOURCE) & (scores.phase == "selection")].sort_values("MAE"))
display(scores[(scores.source == SOURCE) & (scores.phase == "test")].sort_values("MAE"))
display(pd.DataFrame({s:{k:f[k] for k in ["champion","test_MAE","test_RMSE","MAE_improvement_percent"]}
                      for s,f in manifest["sources"].items()}).T)''')
code('''pred = pd.read_csv(ROOT / "artifacts/predictions.csv", parse_dates=["date","origin"])
MODELS = ["ridge_1", "seasonal_naive", "arima_111", "sarima_111_100"]
test = pred[(pred.source == SOURCE) & (pred.phase == "test")]
fig, axes = plt.subplots(2,1,figsize=(12,8),sharex=True)
actual = test[test.model == "naive"]
axes[0].plot(actual.date,actual.actual,color="black",label="Actual",lw=1.5)
for model in MODELS:
    g=test[test.model == model]
    axes[0].plot(g.date,g.prediction,label=model,lw=1)
    axes[1].plot(g.date,g.error,label=model,lw=1)
axes[0].set(title=f"{SOURCE}: selected model comparisons on holdout",ylabel="Anomaly (°C)")
axes[1].set(ylabel="Actual − forecast (°C)",xlabel="Date")
axes[1].axhline(0,color="gray",lw=.7)
axes[0].legend(ncol=3);fig.tight_layout();plt.show()''')
md('''## 9. Calibrate prediction intervals and inspect uncertainty

The fixed winner has ten independent forecast-origin calibration errors per horizon, although temporal dependence can remain. For level c, the absolute-error radius uses rank ceil((n+1)c). With ten calibration errors, finite 80% and 90% intervals are available; 95% is rejected.

Coverage is measured on the held-out targets. Serial dependence and rolling refitting prevent claiming exact exchangeable-sample conformal guarantees. The paired annual-block bootstrap uses only five test blocks and provides weak uncertainty resolution.''')
code('''cal = pred[(pred.source==SOURCE)&(pred.phase=="calibration")]
print("One-month 90% empirical radius:", finite_quantile(cal[cal.horizon==1].error,.9))
intervals = pd.read_csv(ROOT / "artifacts/interval_evaluation.csv")
display(intervals.groupby(["source","level"]).agg(coverage=("covered","mean"),interval_score=("interval_score","mean")))
display(pd.DataFrame({s:f["paired_comparison"] for s,f in manifest["sources"].items()}).T)''')
md('''## 10. Compare final forecasts from all model configurations

Each candidate is independently refitted through October 2016. Choose any models for an overlay, matching the dashboard's checkboxes. Only the fixed validation winner has intervals from separate calibration. Selecting overlays does not change model-selection decisions.''')
code('''future = pd.read_csv(ROOT / "artifacts/candidate_forecasts.csv",parse_dates=["date"])
issued = pd.read_csv(ROOT / "artifacts/forecast.csv",parse_dates=["date"])
g=future[future.source==SOURCE]
fig,ax=plt.subplots(figsize=(12,5))
ax.plot(series.iloc[-60:].index,series.iloc[-60:],label="Observed",color="#8999a6")
for model in MODELS:
    p=g[g.model==model];ax.plot(p.date,p.prediction,label=model)
band=issued[(issued.source==SOURCE)&(issued.level==.9)]
ax.fill_between(band.date,band.lower,band.upper,alpha=.12,label="90% winner interval")
ax.set(title=f"{SOURCE}: historical forecasts issued from October 2016",ylabel="Anomaly (°C)")
ax.legend(ncol=3);fig.tight_layout();plt.show()
display(g.pivot(index="date",columns="model",values="prediction").round(4))''')
md('''## 11. Model interpretation, SQL and API

Ridge coefficients below are effects per training-standard-deviation change while holding other predictors fixed. Correlated lag variables make causal interpretations inappropriate. The SQLite store lets you recompute metrics independently. The API serves the historical forecast snapshot and validates user inputs.''')
code('''display(pd.read_csv(ROOT / f"artifacts/{prefix}_ridge_coefficients.csv"))
with sqlite3.connect(ROOT / "artifacts/experiments.sqlite") as connection:
    sql_scores=pd.read_sql_query("SELECT source,model,AVG(ABS(error)) AS MAE,COUNT(*) AS targets FROM predictions WHERE phase='test' GROUP BY source,model ORDER BY source,MAE",connection)
display(sql_scores)
assert sql_scores.targets.eq(60).all()''')
code('''from fastapi.testclient import TestClient
from climate_ts.api import create_app
client = TestClient(create_app(ROOT))
display(client.get("/health").json())
response=client.get("/forecast?source=GISTEMP&months=3&level=90")
assert response.status_code==200
display(pd.DataFrame(response.json()["forecasts"]))
assert client.get("/forecast?months=13").status_code==422
print("API query and input-validation checks passed")''')
md('''## 12. Use the complete application

Open `reports/dashboard.html` directly in a browser; it embeds its data and plotting library. Start in **Data & visualizations**, then use **Compare models** to select multiple checkboxes. Switch source, evaluation period, included horizons and ranking metric. The forecast tab retains the same model selections, lets you change forecast length and shows the winner's 80% or 90% interval. Export filtered CSV tables from either comparison view.

The project includes 15 exported PNG figures, a visualization gallery, saved estimators, a local API, SQL queries and tests. Run `python -m pytest -q` to verify the software. Run `python -m climate_ts.cli serve` for the local service.

Limits: historical and unverified data vintage, unresolved tail dates, absent anomaly reference metadata, small calibration/test samples, descriptive rather than causal inference, and no cloud deployment. Read the method and interview guides before using the suggested CV wording.

References: [statsmodels forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html), [scikit-learn lag features](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html), [HAC inference](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html), [dataset-family context](https://github.com/datasets/global-temp).''')
nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
nbf.write(nb,root/"notebooks/01_end_to_end.ipynb")
print(f"Created {len(cells)} cells; {sum(c.cell_type=='code' for c in cells)} code cells")

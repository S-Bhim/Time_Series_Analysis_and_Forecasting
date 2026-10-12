# Global Temperature Analysis and Forecasting

A complete independent project beginning with the raw CSV: audit, cleaning, visualization, statistical diagnostics, model training, evaluation, uncertainty and serving.

## Data and preparation

3288 raw records, 3284 retained records and 4 quarantined records. Each source has 1,642 consecutive monthly records from January 1880 through October 2016. Mixed date formats are parsed explicitly. The disconnected late records are preserved separately without guessed corrections or imputed temperatures.

SHA-256 of the unchanged source: `c6272a009f1ada973116261f8d4a5f17e3f1a654b686ed4921e9c27d3e89d8ee`.
The values are treated as anomalies in °C from the dataset family; the supplied CSV does not establish original baselines or source-release provenance. All findings concern this snapshot.

## Exploratory analysis and visualization

The interactive dashboard provides 11 visualization types for each selected source: monthly history, trailing 12/60-month means, complete annual means and changes, year-by-month heatmap, calendar-month boxes, histogram, decade boxes, rolling variability and first differences, robust STL, ACF/PACF of first differences, and source-comparison scatterplots.

Fifteen exportable PNG figures and [a full gallery](visualizations.md) accompany the notebook. The long-run anomaly distributions move upward in this snapshot, while monthly variability remains substantial. Calendar-month distributions combine different warming regimes and are not treated as isolated evidence of seasonality. No sources are averaged together.

## Model comparison and forecasts

The dashboard exposes checkboxes for all ten configurations, a source selector, selection/holdout period selector, horizon filter and ranking metric. Selected models update forecast overlays, error histories, error distributions, horizon plots and the comparison table. The forecast tab overlays any selected combination of the ten final candidate refits. Only the fixed validation winner has independently calibrated intervals. CSV exports reflect the selected filters.

Ten configurations were evaluated across 15 model-selection origins, ten winner-calibration origins and five final test origins. Every origin fits on the preceding 600 months and forecasts the next 12. The model winner is chosen before calibration and test. All 420 backtest fits completed with 0 failed fits; 20 final candidate fits generated comparison forecasts.

## GCAG

The model selected on validation was **ridge_1**, with selection MAE 0.0998 °C.
On the 60-month holdout: **MAE 0.0982 °C**, RMSE 0.1314 °C and seasonal MASE 0.740.
Seasonal-naive test MAE was 0.1186 °C, giving a measured **17.2% MAE reduction**.
Empirical coverage of the nominal 90% interval was **86.7%**.

Since-1970 annual slope through 2015: 0.1686 °C/decade, HAC 95% CI [0.1504, 0.1868].
Exploratory single-hinge year: 1972. This does not identify a cause.
The paired annual-block bootstrap interval for MAE gain is [-0.0080, 0.0488] °C. It includes zero; superiority is not statistically decisive.

## GISTEMP

The model selected on validation was **ridge_1**, with selection MAE 0.1173 °C.
On the 60-month holdout: **MAE 0.1109 °C**, RMSE 0.1380 °C and seasonal MASE 0.729.
Seasonal-naive test MAE was 0.1153 °C, giving a measured **3.9% MAE reduction**.
Empirical coverage of the nominal 90% interval was **91.7%**.

Since-1970 annual slope through 2015: 0.1754 °C/decade, HAC 95% CI [0.1577, 0.1930].
Exploratory single-hinge year: 1972. This does not identify a cause.
The paired annual-block bootstrap interval for MAE gain is [-0.0219, 0.0338] °C. It includes zero; superiority is not statistically decisive.

## Limits and interpretation

This is a historical research project. Issued forecasts cover November 2016–October 2017, not the present day. Statistical forecasts omit physical climate dynamics, emissions and external future drivers. Data vintage revisions cannot be reconstructed. Small calibration and test samples and temporal dependence limit interval reliability and comparisons. Full-history trend, decomposition and break analyses are descriptive and never used as backtest features. A better test score in hindsight does not justify switching the selected winner.

## Reproduction

Run the notebook from the first cell or `python -m climate_ts.cli run` from the project folder. All outputs are rebuilt from `data/raw/monthly_csv.csv`; no earlier project files or precomputed results are required as inputs. Open `reports/dashboard.html` directly for offline interaction.

## References

- Dataset-family context: https://github.com/datasets/global-temp
- Forecast evaluation: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html
- Causal lag features: https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html
- HAC covariance: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html

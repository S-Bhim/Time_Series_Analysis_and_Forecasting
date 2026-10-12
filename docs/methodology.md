# Methodology and design decisions

## Research questions

1. Which fixed model configuration best forecasts the next 12 monthly anomalies when evaluated chronologically?
2. How much forecast error remains at each horizon, and how well do independently calibrated intervals cover subsequent observations?
3. What long-run trends and possible changes in slope are visible in this supplied snapshot, allowing for correlated errors?

The target is each source's reported anomaly, treated as °C. It is not absolute global temperature, a local weather forecast, or a direct measurement relative to a known preindustrial baseline. No causal attribution is attempted.

## Data contract

The CSV must contain exactly `Source,Date,Mean`, finite numeric values, the two expected sources, valid dates and unique source-month keys. ISO strings use `%Y-%m-%d`; other accepted strings use `%d-%m-%Y`. In this input, calling a single permissive mixed-date parser with `dayfirst=True` can misread the older ISO dates. Dates are reduced to monthly keys only after format-specific parsing.

The parser retains each source's uninterrupted prefix and quarantines records after the first missing month. The input has an 11-month gap before two late records per source. Those records might have incorrect dates, but row order is not sufficient evidence to repair them. No imputation is applied. A hash proves the raw copy matches the user's input. No upstream download replaces supplied values.

## Exploratory visualization workflow

The project constructs its entire EDA from the audited observations before fitting forecasts. Interactive charts cover raw history, causal trailing means, complete annual means and changes, year-month heatmaps, calendar-month boxes, histograms, decade boxes, rolling standard deviations, first differences, full STL, first-difference ACF/PACF and source comparison. Missing cells remain unavailable, and partial years are excluded from annual summaries. Fifteen PNG figures and a gallery provide portable outputs. Monthly boxes span warming regimes and are not interpreted as an isolated seasonal-effect test.

The dashboard provides explicit multiple-model checkboxes across all ten configurations. It recomputes comparison metrics from the selected source, evaluation period and horizon subset. That interactive filtering is exploratory; it never changes the validation winner. The future tab overlays all selected final candidate refits, with separately calibrated intervals available only for the fixed winner.

## Predeclared evaluation protocol

All candidate definitions and the 600-month rolling training window are fixed in configuration. The window focuses on the preceding 50 years instead of assuming one forecast relation holds unchanged across the entire century. It is a design assumption, not a window selected on test performance. Window-length sensitivity remains a future experiment requiring its own validation design.

| Phase | Target months | Origins | Purpose |
|---|---|---:|---|
| Selection | Nov 1986–Oct 2001 | 15 | Choose configuration by MAE |
| Calibration | Nov 2001–Oct 2011 | 10 | Estimate winner's interval radii |
| Test | Nov 2011–Oct 2016 | 5 | Evaluate locked recipe |

Every origin produces 12 predictions at once. Origins are a year apart, so forecast target blocks do not overlap. Training is repeated using only history available at the origin. Observations from an earlier test block become available at later origins, consistent with annual refitting. No observation from the block being predicted is available to that fit. Model identities and hyperparameters are frozen before calibration and test.

Candidates that fail any selection fold cannot win; failures remain in the log. There is no silent fallback substitution. State-space optimizers can retry with Powell from the initial fit's parameters if the first optimizer fails to converge. Warning messages are retained even after a successful retry. A missing champion calibration or test forecast stops the pipeline.

## Candidate configurations

| Candidate | Specification |
|---|---|
| Naive | Repeat latest observed value |
| Seasonal naive | Repeat the last 12 months |
| Drift | Extrapolate training-endpoint slope |
| ETS damped | Additive damped trend, no seasonal component |
| ARIMA | (1,1,1) with drift via SARIMAX constant |
| SARIMA | (1,1,1)(1,0,0)[12] with drift |
| Ridge | Standardized causal lag features, alpha 1 or 10 |
| Gradient boosting | Histogram boosting, depth 3 or 5, 140 iterations, no random early-stop split |

The grid is deliberately limited and declared before testing. It is not exhaustive auto-ARIMA or a claim of globally optimal tuning. Ten configurations represent seven model families.

For machine learning, predictors are lags 1, 2, 3, 6, 12 and 24; preceding 3- and 12-month means; preceding 12-month standard deviation; change between the most recent and year-earlier value; calendar sine/cosine; and time in years within the fitting window. Training examples start only after 24 historical months. Ridge scaling is fitted inside each training fold. For steps 2–12, previous predictions are recursively inserted into the history. Boosting early stopping is disabled to avoid a random internal validation split.

LSTM is not included merely to make the project look advanced. With two correlated series, about 1,600 months each and a strong need for auditability, a carefully evaluated statistical/ML benchmark is a defensible first advanced project. A neural extension would need separate validation, adequate tuning and a demonstrated benefit.

## Metrics and comparison uncertainty

For errors `e = actual - forecast`, MAE is `mean(abs(e))`; RMSE is `sqrt(mean(e²))`. Bias is `mean(forecast - actual)`, so positive means overprediction. Seasonal MASE divides absolute errors by the mean absolute 12-month difference in that origin's training window, then averages the scaled errors. MASE below one compares with the *training* seasonal error scale; it does not mean the same thing as beating the seasonal-naive model on the test period.

MAPE is unsuitable because anomalies are negative or near zero. Each source and horizon is reported separately. The paired bootstrap samples five annual-origin blocks with replacement, retaining the 12 within-block errors. Its 95% interval is exploratory: five blocks cannot give strong precision, and cross-block dependence can remain. Both observed gain intervals include zero in this run.

## Prediction intervals

For each horizon `h`, retain ten absolute errors of the fixed champion from the separate calibration years. For nominal coverage `c`, use order statistic `ceil((n+1)*c)`. The interval is `forecast ± radius[h,c]`. With n=10 this supports finite 80% and 90% intervals; 95% is rejected rather than represented by a misleading finite radius.

Radii are frozen before test scoring. Evaluate empirical coverage, width and the interval score `width + (2/alpha)*(miss below + miss above)`. These are conformal-style empirical intervals, but calibration forecasts use rolling refits and time series violate exchangeability. No exact distribution-free guarantee is claimed. Nonmonotone widths by horizon are retained transparently rather than smoothed after observing test results.

Future snapshot forecasts use the winner refitted to the last 600 months and the same pre-test calibration radii. They extend the historical series by 12 months; their accuracy is not evaluated against the quarantined records.

## Statistical inference and exploratory diagnostics

- Annual regressions use only full 12-month years. Slopes and 95% intervals use OLS with HAC covariance and three annual lags. Two prespecified descriptive windows are full history and since 1970; neither establishes causes.
- ADF and KPSS are applied to level data with a trend and first differences with an intercept. Their nulls differ. Report warnings because table-bound p-values are approximate bounds, not precise tail probabilities.
- Robust STL uses period 12. It estimates a seasonal pattern; the project does not assume anomaly series have the strong seasonality of absolute temperature. Full-history STL is for EDA only and never enters forecast features.
- A single continuous hinge is searched across complete annual years with a minimum 20-year margin at either end. BIC selects a descriptive candidate; ordinary coefficient uncertainty after this search would ignore selection. No change-point p-value or causal event label is assigned.
- A Ljung–Box statistic is reported on concatenated holdout forecast errors only as a descriptive dependence check. These are multi-horizon forecast errors, not independent in-sample innovations.
- Raw and differenced source correlations are both shown to expose the effect of a common trend. Source estimates may share underlying observations and use distinct anomaly reference periods.

## Engineering and reproducibility

The package separates ingestion, forecasting, evaluation, inference, artifact generation and serving. CSV tables allow independent recalculation; SQLite supports SQL querying; a manifest stores data/configuration/code hashes and package versions. Saved models retain the final estimator and its history. The API serves a fixed forecast snapshot and offers a stateless interval-exceedance check. Automated tests exercise malformed inputs, mixed dates, feature causality, unseen-target isolation, small-sample interval quantiles, model outputs and API validation.

References and full empirical results are in `reports/results.md`. Source baselines and original release remain unknown; use findings as a portfolio demonstration of method rather than official climate evidence.

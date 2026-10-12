# Explain and defend the project

## A 60-second explanation

I built a complete global-temperature time series project from a raw CSV. I audited mixed dates and a disconnected tail, then visualized monthly trends, moving averages, annual patterns, heatmaps, distributions, decomposition and serial dependence for both sources. I compared ten model configurations using chronological rolling-origin validation, calibrated intervals in a separate period and evaluated a fixed winner on 60 unseen months. Ridge won validation for both sources. I used HAC uncertainty for trends and built a dashboard that compares any selected combination of models, plus a reproducible notebook, SQL store, local API and automated tests.

## Questions to practise

**Why is this advanced?** The main work is experimental validity: data provenance, time-aware validation, fold-local preprocessing, recursive forecasting, distinct calibration, explicit uncertainty limitations and reproducible serving. Complexity is not measured by adding a neural model.

**Why not a random train/test split?** Adjacent months and their lag features are dependent. Random splits expose models to future regimes and lead to an evaluation that does not resemble forecasting.

**How did you avoid leakage?** Every fit receives only the preceding 600 months. Scaling is learned there. Features for a target use prior values. During a 12-month forecast, later features use earlier predictions, not the actual target values. Full-history EDA never feeds forecasting features.

**Is using prior test observations in later refits cheating?** No, under this annual update policy those observations would already be known. Hyperparameters and model choice remain fixed. It would be a different experiment to forecast all five years from one initial origin.

**What won, and how much better was it?** Ridge alpha 1 won selection for both sources. Holdout MAE was about 0.098 °C for GCAG and 0.111 °C for GISTEMP, 17.2% and 3.9% below seasonal naive. The bootstrap intervals span zero, so I describe measured improvements without asserting decisive significance.

**Why not switch to SARIMA for GISTEMP?** It did slightly better in hindsight on the test set, but choosing it using that result would turn the holdout into another tuning set. A subsequent model revision needs new evaluation data or a properly nested design.

**Why no MAPE?** Temperature anomalies cross zero and can be negative. Percentage errors become unstable or hard to interpret. MAE in °C and MASE are more suitable.

**What does 90% mean here?** It is the nominal calibration target. Actual test coverage was 86.7% and 91.7%, respectively. Serial dependence, shifting distributions and ten calibration errors per horizon prevent claiming exact guaranteed coverage.

**Why HAC standard errors?** Annual trend residuals can be autocorrelated and heteroskedastic. Newey–West HAC covariance allows some of that dependence instead of treating all errors as independent. The bandwidth choice and linear trend specification still matter.

**Does the break in 1972 prove a cause?** No. It is the best descriptive single-hinge fit among searched years. Search uncertainty and model specification matter, and no exogenous causal variables are present.

**Can the dashboard forecast 2027?** No. The retained observations end in 2016. The application demonstrates an as-of-2016 forecasting workflow. A current forecast requires authenticated updated data and a new evaluation.

**Is the API production deployed?** It is implemented and locally tested. It serves versioned precomputed forecasts. Docker and CI configuration are included, but cloud hosting, uptime, live ingestion and operational retraining are not claimed.

## Suggested walkthrough order

1. Show the data audit and one example of each date format.
2. Walk through the visualization gallery and explain the difference between trends, seasonality and short-term variation.
3. Explain the selection/calibration/test timeline before showing model scores.
4. Demonstrate a causal lag row and recursive 12-step forecasting.
5. Use dashboard checkboxes to compare several models, including a failure or limitation.
6. Show nominal versus empirical interval coverage.
7. Query the SQLite experiment tables and make one API call.
8. Explain the tests and the result manifest.

## Sensible next experiments

Authenticate and refresh the data vintage; document anomaly reference periods; assess rolling-window lengths inside validation; add external predictors only with release-date-aware availability; try direct multi-horizon models; evaluate neural models only after a sufficiently rigorous tuning budget; use temporally adaptive calibration; preserve immutable run directories and deploy with access controls if needed. These are extensions, not implemented claims.

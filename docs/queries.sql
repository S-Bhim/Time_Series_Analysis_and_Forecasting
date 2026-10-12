-- Run against artifacts/experiments.sqlite. Positive bias means overprediction.
SELECT source, model, MAE, RMSE, MASE, bias, n
FROM scores WHERE phase = 'test' ORDER BY source, MAE;

-- Check historical source/date coverage without pooling source anomalies.
SELECT source, MIN(date) AS first_month, MAX(date) AS last_month, COUNT(*) AS months
FROM observations GROUP BY source;

-- Independently recompute nominal interval coverage on the test block.
SELECT source, level, AVG(CASE WHEN actual BETWEEN lower AND upper THEN 1.0 ELSE 0.0 END) AS coverage,
       AVG(upper - lower) AS mean_width
FROM interval_evaluation GROUP BY source, level;

-- Audit optimiser warnings or failures; no silent baseline substitutions.
SELECT source, phase, model, origin, status, message
FROM run_log WHERE status <> 'ok' OR COALESCE(message, '') <> '';

-- Locate holdout review flags; statistical surprises, not confirmed data errors.
SELECT source, date, actual, prediction, lower, upper
FROM interval_evaluation WHERE level = 0.9 AND (actual < lower OR actual > upper);

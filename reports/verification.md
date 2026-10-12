# Standalone project verification

- The notebook starts with the raw CSV and performs inspection, cleaning, visual exploration, statistical diagnostics, training, evaluation, forecasting, SQL queries and API checks in order.
- All 22 code cells in the 43-cell notebook executed and have saved outputs. Notebook structural validation passed; no error outputs are present.
- The complete pipeline executed 420 backtest fits with zero failed fits, plus 20 final candidate fits.
- All 28 Python tests passed, including causal-feature checks, invalid input handling, EDA missingness and rolling-window checks, candidate-forecast consistency, API validation and dashboard data packaging.
- All 35 offline JavaScript assertions passed. They cover multiple-model selection, select-all/clear/default actions, source retention, period and horizon filters, metric recomputation, interval controls including a one-month forecast, and CSV exports.
- The JavaScript tests use a minimal DOM and Plotly stub. Automated browser inspection of the local-file page was blocked by browser security, so current browser rendering is not claimed as verified.
- Fifteen exported EDA figures were visually inspected, including a contact-sheet review and detailed heatmap/ACF checks.
- Every MAE and RMSE score was independently recomputed from the 5,040 backtest prediction rows and matched to tolerance 1e-12. All candidate forecasts total 240 rows, covering two sources, ten configurations and 12 horizons.
- SQLite row counts agree with CSV artifacts. The source CSV remains unchanged and its SHA-256 is recorded.
- Non-failing library warnings concern future ACF return conventions and TestClient's httpx deprecation. Docker and hosted CI were not executed. No cloud deployment is claimed.

Refresh or reopen the dashboard file if an older version is open in your browser. Use the Compare models tab to access all ten model checkboxes.

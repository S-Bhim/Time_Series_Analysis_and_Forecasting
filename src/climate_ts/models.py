"""All forecasts depend only on the history passed to fit_forecast."""
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

LAGS = (1, 2, 3, 6, 12, 24)
FEATURE_NAMES = [f"lag_{v}" for v in LAGS] + ["mean_3", "mean_12", "std_12", "change_12", "month_sin", "month_cos", "time_years"]


def feature_row(history, target_date, t):
    a = np.asarray(history, dtype=float)
    if len(a) < 24:
        raise ValueError("24 historical observations required")
    month = pd.Timestamp(target_date).month
    return [a[-lag] for lag in LAGS] + [a[-3:].mean(), a[-12:].mean(), a[-12:].std(),
           a[-1] - a[-13], np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), t/12]


def supervised(series):
    a = series.to_numpy(dtype=float)
    return np.asarray([feature_row(a[:i], series.index[i], i) for i in range(24, len(a))]), a[24:]


def fit_forecast(name, series, horizon=12, seed=42):
    if not 1 <= horizon <= 12:
        raise ValueError("Supported horizon is 1-12 months")
    a = series.to_numpy(dtype=float)
    if len(a) < 36 or not np.isfinite(a).all():
        raise ValueError("At least 36 finite observations required")
    fitted = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if name == "naive":
            pred = np.repeat(a[-1], horizon)
        elif name == "seasonal_naive":
            pred = a[-12:][:horizon]
        elif name == "drift":
            pred = a[-1] + np.arange(1, horizon+1) * (a[-1]-a[0])/(len(a)-1)
        elif name == "ets_damped":
            fitted = ExponentialSmoothing(a, trend="add", damped_trend=True, initialization_method="estimated").fit()
            pred = fitted.forecast(horizon)
        elif name in ("arima_111", "sarima_111_100"):
            seasonal = (1, 0, 0, 12) if name.startswith("sarima") else (0, 0, 0, 0)
            model = SARIMAX(a, order=(1, 1, 1), seasonal_order=seasonal, trend="c", enforce_stationarity=True, enforce_invertibility=True)
            fitted = model.fit(disp=False, maxiter=180)
            if not fitted.mle_retvals.get("converged", True):
                fitted = model.fit(start_params=fitted.params, method="powell", disp=False, maxiter=250)
            if not fitted.mle_retvals.get("converged", True):
                raise RuntimeError(f"{name}: optimizer did not converge")
            pred = fitted.forecast(horizon)
        elif name.startswith(("ridge_", "boosting_")):
            X, y = supervised(series)
            if name.startswith("ridge_"):
                fitted = make_pipeline(StandardScaler(), Ridge(alpha=float(name.split("_")[1])))
            else:
                fitted = HistGradientBoostingRegressor(max_iter=140, max_depth=int(name.split("_")[1]),
                    learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=20, early_stopping=False, random_state=seed)
            fitted.fit(X, y)
            future = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
            history, pred = a.tolist(), []
            for j, date in enumerate(future):
                value = float(fitted.predict(np.asarray([feature_row(history, date, len(a)+j)]))[0])
                pred.append(value)
                history.append(value)  # Never use future actual values in recursive forecasts.
        else:
            raise ValueError(f"Unknown model: {name}")
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (horizon,) or not np.isfinite(pred).all():
        raise RuntimeError(f"Invalid predictions for {name}")
    return pred, fitted, sorted(set(str(w.message) for w in captured))

"""Descriptive inference is separate from model selection."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox


def analyse_series(series):
    # Only complete annual means enter trend inference and break exploration.
    annual = series.resample("YS").agg(["mean", "count"])
    annual = annual[annual["count"] == 12]["mean"]
    trends = []
    for label, sample in [("full_complete_years", annual), ("since_1970", annual[annual.index.year >= 1970])]:
        t = (sample.index.year.to_numpy()-sample.index.year.min())/10
        fitted = sm.OLS(sample.to_numpy(), sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        ci = fitted.conf_int()[1]
        trends.append({"window": label, "start": int(sample.index.year.min()), "end": int(sample.index.year.max()),
            "n_years": len(sample), "slope_c_per_decade": float(fitted.params[1]), "lower_95": float(ci[0]),
            "upper_95": float(ci[1]), "p_value": float(fitted.pvalues[1]), "HAC_lags_years": 3})
    stationarity = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for label, a, reg in [("level_with_trend", series, "ct"), ("first_difference", series.diff().dropna(), "c")]:
            adf = adfuller(a, regression=reg, autolag="AIC")
            kp = kpss(a, regression=reg, nlags="auto")
            stationarity.append({"series": label, "adf_stat": float(adf[0]), "adf_p": float(adf[1]),
                                 "kpss_stat": float(kp[0]), "kpss_p": float(kp[1]), "adf_lags": int(adf[2])})
        notes = sorted(set(str(w.message) for w in captured))
    stl = STL(series, period=12, robust=True).fit()
    decomp = pd.DataFrame({"date": series.index, "observed": series.values, "trend": stl.trend,
                           "seasonal": stl.seasonal, "residual": stl.resid}).reset_index(drop=True)
    season_strength = max(0., 1-float(np.var(stl.resid)/np.var(stl.resid+stl.seasonal)))
    # Single continuous hinge, searched on full descriptive annual series.
    x = annual.index.year.to_numpy(dtype=float)
    base = sm.OLS(annual.to_numpy(), sm.add_constant(x-x[0])).fit()
    candidates = []
    for year in x[20:-20]:
        fit = sm.OLS(annual.to_numpy(), np.column_stack([np.ones(len(x)), x-x[0], np.maximum(0, x-year)])).fit()
        candidates.append({"break_year": int(year), "bic": float(fit.bic),
                           "pre_slope_c_per_decade": float(fit.params[1]*10),
                           "post_slope_c_per_decade": float((fit.params[1]+fit.params[2])*10)})
    best = min(candidates, key=lambda v: v["bic"])
    best["linear_bic"] = float(base.bic)
    best["note"] = "Exploratory best single hinge among annual candidates; search uncertainty omitted; not causal attribution or a formal change-point confidence interval"
    return {"trends": trends, "stationarity": stationarity, "stationarity_notes": notes,
            "seasonal_strength": season_strength, "exploratory_break": best}, decomp


def error_diagnostics(test_predictions):
    e = test_predictions.sort_values("date").error.to_numpy()
    lb = acorr_ljungbox(e, lags=[12], return_df=True).iloc[0]
    return {"mean_error": float(e.mean()), "ljung_box_12_p": float(lb.lb_pvalue),
            "note": "Descriptive serial correlation of concatenated multi-horizon errors; not an in-sample innovation whiteness test"}

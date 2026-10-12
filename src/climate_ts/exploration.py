"""Exploratory analysis from cleaned observations, independent of forecasting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from .data import series_for

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CHARTS = {
    "history": ("Monthly temperature anomalies", "Every retained observation, in chronological order. Use the range slider to zoom."),
    "rolling": ("Trailing averages", "12- and 60-month means use only the current and preceding observations. No centered smoothing."),
    "annual": ("Annual means and year-to-year change", "Only years with all 12 observations enter annual summaries. The partial final year is omitted."),
    "heatmap": ("Year × month heatmap", "Each cell is one month's anomaly. Blank cells mean unavailable observations, never zero."),
    "monthly": ("Distributions by calendar month", "Month groups span different warming regimes. Differences do not by themselves establish a seasonal effect."),
    "distribution": ("Anomaly distribution", "The distribution pools a changing historical series; it is not a stationary probability model."),
    "decades": ("Distributions across decades", "Decade groups expose shifts in location and spread. The last decade is incomplete."),
    "volatility": ("Rolling variability and monthly changes", "Trailing 12-month standard deviation and first differences describe different kinds of variation."),
    "stl": ("Robust STL decomposition", "Observed, trend, 12-month seasonal component and remainder. Full-history decomposition is descriptive only."),
    "correlation": ("ACF and PACF of first differences", "First 36 lags with approximate 95% bounds. Differencing and dependence diagnostics inform model reasoning."),
    "sources": ("Compare the two sources", "Inspect both levels and changes. Shared observations and different reference periods limit interpretation."),
}


def explore(clean):
    """Return chart-ready summaries without any training or artifact dependencies."""
    result = {}
    for source in sorted(clean.source.unique()):
        s = series_for(clean, source)
        annual = s.resample("YS").agg(["mean", "count"])
        annual = annual.loc[annual["count"] == 12, "mean"]
        frame = s.rename("value").reset_index()
        frame["year"], frame["month"] = frame.date.dt.year, frame.date.dt.month
        heat = frame.pivot(index="year", columns="month", values="value").reindex(columns=range(1, 13))
        decomposition = STL(s, period=12, robust=True).fit()
        a, aci = acf(s.diff().dropna(), nlags=36, alpha=.05)
        p, pci = pacf(s.diff().dropna(), nlags=36, alpha=.05, method="ywm")
        result[source] = dict(series=s, frame=frame, annual=annual, heatmap=heat,
            mean12=s.rolling(12, min_periods=12).mean(), mean60=s.rolling(60, min_periods=60).mean(),
            volatility=s.rolling(12, min_periods=12).std(), difference=s.diff(), stl=decomposition,
            acf=a[1:], acf_bounds=aci[1:]-a[1:, None], pacf=p[1:], pacf_bounds=pci[1:]-p[1:, None])
    return result


def plotly_figures(clean, prepared=None):
    prepared = prepared or explore(clean)
    wide = clean.pivot(index="date", columns="source", values="value")
    output = {}
    for source, d in prepared.items():
        s, frame, annual = d["series"], d["frame"], d["annual"]
        figures = {}
        figures["history"] = go.Figure(go.Scatter(x=s.index, y=s, mode="lines", name=source, line=dict(color="#147d92", width=1)))
        figures["history"].update_xaxes(rangeslider_visible=True)
        figures["rolling"] = go.Figure()
        for label, values, color in [("Monthly", s, "#ced6de"), ("12-month mean", d["mean12"], "#147d92"), ("60-month mean", d["mean60"], "#bd5839")]:
            figures["rolling"].add_trace(go.Scatter(x=s.index, y=values, name=label, line=dict(color=color, width=1.8)))
        figures["annual"] = make_subplots(rows=2, cols=1, shared_xaxes=True)
        figures["annual"].add_trace(go.Scatter(x=annual.index, y=annual, name="Complete annual means"), row=1, col=1)
        figures["annual"].add_trace(go.Bar(x=annual.index, y=annual.diff(), name="Year-to-year change"), row=2, col=1)
        figures["heatmap"] = go.Figure(go.Heatmap(x=MONTHS, y=d["heatmap"].index, z=d["heatmap"].values,
            colorscale="RdBu_r", zmid=0, colorbar=dict(title="°C"), hoverongaps=False))
        figures["monthly"] = go.Figure()
        for m, label in enumerate(MONTHS, 1):
            figures["monthly"].add_trace(go.Box(y=frame.loc[frame.month == m, "value"], name=label, boxpoints=False, showlegend=False))
        figures["distribution"] = go.Figure(go.Histogram(x=s, nbinsx=45, marker_color="#147d92", name="Months"))
        figures["distribution"].update_xaxes(title="Anomaly (°C)")
        figures["distribution"].update_yaxes(title="Number of months")
        figures["decades"] = go.Figure()
        for decade, g in frame.groupby((frame.year//10)*10):
            figures["decades"].add_trace(go.Box(y=g.value, name=str(decade)+"s", boxpoints=False, showlegend=False))
        figures["volatility"] = make_subplots(rows=2, cols=1, shared_xaxes=True)
        for row, key, name in [(1, "volatility", "Trailing 12-month standard deviation"), (2, "difference", "Monthly change")]:
            figures["volatility"].add_trace(go.Scatter(x=s.index, y=d[key], name=name, line=dict(width=1)), row=row, col=1)
        figures["stl"] = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.06)
        for row, key, values in [(1, "Observed", s), (2, "Trend", d["stl"].trend), (3, "Seasonal", d["stl"].seasonal), (4, "Remainder", d["stl"].resid)]:
            figures["stl"].add_trace(go.Scatter(x=s.index, y=values, name=key, line=dict(width=1)), row=row, col=1)
            figures["stl"].update_yaxes(title_text=key, row=row, col=1)
        figures["correlation"] = make_subplots(rows=2, cols=1, shared_xaxes=True)
        for row, key in [(1, "acf"), (2, "pacf")]:
            figures["correlation"].add_trace(go.Bar(x=list(range(1,37)), y=d[key], name=key.upper()), row=row, col=1)
            for bound in [0,1]:
                figures["correlation"].add_trace(go.Scatter(x=list(range(1,37)), y=d[key+"_bounds"][:, bound],
                    mode="lines", line=dict(dash="dot", color="#9b4c40"), showlegend=False), row=row, col=1)
        figures["correlation"].update_xaxes(title_text="Lag in months", row=2, col=1)
        figures["sources"] = make_subplots(rows=1, cols=2, subplot_titles=["Reported anomaly levels", "First differences"])
        for col, points in [(1, wide), (2, wide.diff().dropna())]:
            figures["sources"].add_trace(go.Scatter(x=points.GCAG, y=points.GISTEMP, mode="markers",
                marker=dict(size=4, opacity=.4, color="#147d92"), showlegend=False), row=1, col=col)
            figures["sources"].update_xaxes(title_text="GCAG (°C)", row=1, col=col)
            figures["sources"].update_yaxes(title_text="GISTEMP (°C)", row=1, col=col)
        for key, fig in figures.items():
            fig.update_layout(template="plotly_white", margin=dict(l=62,r=30,t=45,b=50),
                height=620 if key == "stl" else 520 if key == "heatmap" else 440,
                font=dict(family="Arial", color="#23384b"), legend=dict(orientation="h", y=1.12), hovermode="closest")
            if key in ["history", "rolling", "monthly", "decades"]:
                fig.update_yaxes(title="Anomaly (°C)")
        output[source] = {key: json.loads(fig.to_json()) for key, fig in figures.items()}
    return output


def export_visualizations(root, clean, prepared=None):
    import matplotlib.pyplot as plt
    root = Path(root)
    target = root/"reports/figures"
    target.mkdir(parents=True, exist_ok=True)
    prepared = prepared or explore(clean)
    entries = []
    def save(fig, source, name, caption):
        fig.suptitle(f"{source}: {caption}", fontsize=15)
        fig.tight_layout(rect=(0,0,1,.97))
        path = target/f"{source.lower()}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        entries.append((path.name, f"{source}: {caption}"))
    for source,d in prepared.items():
        s, a, f, heat = d["series"], d["annual"], d["frame"], d["heatmap"]
        fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
        axes[0].plot(s.index,s,color="#bdc7d0",lw=.6,label="Monthly")
        axes[0].plot(s.index,d["mean12"],color="#147d92",label="Trailing 12 months")
        axes[0].plot(s.index,d["mean60"],color="#bd5839",label="Trailing 60 months")
        axes[0].legend(ncol=3); axes[0].set_ylabel("Anomaly (°C)")
        axes[1].plot(a.index,a,color="#147d92");axes[1].set_ylabel("Complete annual mean (°C)")
        save(fig,source,"history","Monthly history, trailing means and complete annual means")
        fig,ax=plt.subplots(figsize=(11,10))
        lim=float(np.nanmax(np.abs(heat.values)))
        im=ax.imshow(heat.values,aspect="auto",origin="lower",cmap="RdBu_r",vmin=-lim,vmax=lim)
        ax.set_xticks(range(12),MONTHS); idx=list(range(0,len(heat),10));ax.set_yticks(idx,heat.index[idx]);fig.colorbar(im,ax=ax,label="Anomaly (°C)")
        save(fig,source,"heatmap","Year-by-month anomalies (unavailable months remain blank)")
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        axes[0].hist(s,bins=45,color="#147d92");axes[0].set(xlabel="Anomaly (°C)",ylabel="Number of months")
        groups=list(f.groupby((f.year//10)*10))
        axes[1].boxplot([g.value for _,g in groups],tick_labels=[str(v) for v,_ in groups],showfliers=False)
        axes[1].tick_params(axis="x",rotation=60);axes[1].set(xlabel="Decade",ylabel="Anomaly (°C)")
        save(fig,source,"distributions","Distribution and decade shifts")
        fig,ax=plt.subplots(figsize=(12,4))
        ax.boxplot([f.loc[f.month==m,"value"] for m in range(1,13)],tick_labels=MONTHS,showfliers=False)
        ax.set_ylabel("Anomaly (°C)")
        save(fig,source,"months","Calendar-month distributions across the full history")
        fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
        for ax,label,values in zip(axes,["Observed","Trend","Seasonal","Remainder"],[s,d["stl"].trend,d["stl"].seasonal,d["stl"].resid]):
            ax.plot(s.index,values,color="#147d92",lw=.7);ax.set_ylabel(label+" (°C)")
        save(fig,source,"decomposition","Robust STL, descriptive full-history decomposition")
        fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        for ax,key in zip(axes,["acf","pacf"]):
            ax.bar(range(1,37),d[key],color="#147d92");ax.plot(range(1,37),d[key+"_bounds"],color="#bd5839",ls="--",lw=.8);ax.set_ylabel(key.upper())
        axes[1].set_xlabel("Lag in months")
        save(fig,source,"autocorrelation","First-difference ACF/PACF with approximate 95% bounds")
        fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        axes[0].plot(s.index,d["volatility"],color="#147d92");axes[0].set_ylabel("Trailing 12-month SD (°C)")
        axes[1].plot(s.index,d["difference"],color="#bd5839",lw=.7);axes[1].set_ylabel("Monthly change (°C)")
        save(fig,source,"variability","Rolling standard deviation and first differences")
    wide=clean.pivot(index="date",columns="source",values="value")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,label,points in [(axes[0],"Levels",wide),(axes[1],"First differences",wide.diff().dropna())]:
        ax.scatter(points.GCAG,points.GISTEMP,s=7,alpha=.35,color="#147d92");ax.set(title=label,xlabel="GCAG (°C)",ylabel="GISTEMP (°C)")
    save(fig,"Both sources","comparison","Levels and changes; source baselines may differ")
    text="# Visualization gallery\n\nAll figures are generated directly from the audited CSV. These are descriptive analyses, not forecast features.\n\n"
    text += "\n\n".join(f"## {caption}\n\n![{caption}](<figures/{name}>)" for name,caption in entries)
    (root/"reports/visualizations.md").write_text(text,encoding="utf-8")
    return entries

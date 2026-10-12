"""Strict, format-specific ingestion. Never silently repair observations."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_data(path: str | Path):
    path = Path(path)
    raw = pd.read_csv(path)
    if list(raw.columns) != ["Source", "Date", "Mean"]:
        raise ValueError("Expected exactly Source,Date,Mean columns")
    if raw.empty or raw.isna().any().any():
        raise ValueError("Empty input or missing source/date/value")
    if set(raw.Source) != {"GCAG", "GISTEMP"}:
        raise ValueError("Expected GCAG and GISTEMP sources")
    values = pd.to_numeric(raw.Mean, errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("Values must be finite")
    dates = raw.Date.astype(str)
    iso = dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    dmy = dates.str.fullmatch(r"\d{2}-\d{2}-\d{4}")
    if not (iso | dmy).all():
        raise ValueError("Unsupported date format")
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    parsed.loc[iso] = pd.to_datetime(dates[iso], format="%Y-%m-%d")
    parsed.loc[dmy] = pd.to_datetime(dates[dmy], format="%d-%m-%Y")
    clean = pd.DataFrame({"source": raw.Source, "date": parsed.dt.to_period("M").dt.to_timestamp(),
                          "value": values, "raw_date": dates, "csv_row": raw.index + 2})
    if clean.duplicated(["source", "date"]).any():
        raise ValueError("Duplicate source-month records; inspect instead of averaging")
    kept, excluded, gaps = [], [], {}
    for source, group in clean.groupby("source"):
        group = group.sort_values("date").reset_index(drop=True)
        full = pd.date_range(group.date.min(), group.date.max(), freq="MS")
        missing = full.difference(group.date)
        gaps[source] = missing.strftime("%Y-%m").tolist()
        # Explicit conservative policy: keep only the uninterrupted prefix.
        cutoff = missing.min() if len(missing) else group.date.max() + pd.offsets.MonthBegin()
        kept.append(group[group.date < cutoff])
        tail = group[group.date >= cutoff].copy()
        tail["reason"] = "Disconnected tail after first missing month; date provenance unresolved"
        excluded.append(tail)
    clean = pd.concat(kept).sort_values(["source", "date"]).reset_index(drop=True)
    quarantine = pd.concat(excluded).reset_index(drop=True)
    counts = clean.groupby("source").size()
    if counts.min() < 720:
        raise ValueError("At least 720 continuous months per source are required")
    audit = {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_rows": len(raw), "retained_rows": len(clean), "quarantined_rows": len(quarantine),
        "format_counts": {"YYYY-MM-DD": int(iso.sum()), "DD-MM-YYYY": int(dmy.sum())},
        "missing_months": gaps, "policy": "Retain contiguous prefix per source; no imputation or inferred date repairs",
        "start": clean.date.min().strftime("%Y-%m"), "end": clean.date.max().strftime("%Y-%m"),
        "rows_per_source": {k: int(v) for k, v in counts.items()},
        "units": "Temperature anomaly, degrees Celsius (inferred from dataset family; original baseline metadata absent)",
        "provenance": "User-supplied legacy snapshot; not authenticated against an upstream release",
    }
    return clean, quarantine, audit


def series_for(clean, source):
    s = clean[clean.source == source].set_index("date").value.sort_index().asfreq("MS")
    if s.isna().any():
        raise ValueError("Modelling series must be contiguous")
    return s

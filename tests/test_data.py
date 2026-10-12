from pathlib import Path
import pandas as pd
import pytest
from climate_ts.data import load_data, series_for

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/"data/raw/monthly_csv.csv"


def test_mixed_dates_and_quarantine_are_not_silently_repaired():
    clean, quarantine, audit = load_data(RAW)
    assert audit["input_rows"] == 3288
    assert audit["retained_rows"] == 3284
    assert audit["quarantined_rows"] == 4
    assert clean.loc[clean.raw_date == "1899-12-06", "date"].eq(pd.Timestamp("1899-12-01")).all()
    assert clean.loc[clean.raw_date == "06-12-1900", "date"].eq(pd.Timestamp("1900-12-01")).all()
    assert set(quarantine.raw_date) == {"09-10-2017", "06-11-2017"}
    assert set(audit["rows_per_source"].values()) == {1642}
    assert series_for(clean, "GCAG").index[-1] == pd.Timestamp("2016-10-01")
    # No value modification is hidden in date handling.
    raw = pd.read_csv(RAW)
    restored = pd.concat([clean, quarantine]).sort_values("csv_row").value.to_numpy()
    assert (raw.Mean.to_numpy() == restored).all()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_date", "infinite", "wrong_source"])
def test_invalid_data_fails_loudly(tmp_path, mutation):
    frame = pd.read_csv(RAW)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[10]]])
    elif mutation == "missing":
        frame.loc[10, "Mean"] = float("nan")
    elif mutation == "bad_date":
        frame.loc[10, "Date"] = "not-a-date"
    elif mutation == "infinite":
        frame.loc[10, "Mean"] = float("inf")
    else:
        frame.loc[10, "Source"] = "UNKNOWN"
    path = tmp_path/"bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, TypeError)):
        load_data(path)

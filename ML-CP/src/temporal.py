"""Calendar-based expanding validation; observations on a date stay together."""
import numpy as np
import pandas as pd


def calendar_folds(dates, validation_years=(2020, 2021, 2022)):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("Valuation dates must not be missing")
    for year in validation_years:
        start, end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
        train = np.flatnonzero((dates < start).to_numpy())
        valid = np.flatnonzero(((dates >= start) & (dates < end)).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError(f"No training or validation observations for {year}")
        if not dates.iloc[train].max() < dates.iloc[valid].min():
            raise ValueError("Training dates must strictly precede validation dates")
        yield train, valid

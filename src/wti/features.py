"""Point-in-time weekly panel: rolled returns, curve slope and abnormal inventory changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decision_dates(trading_days: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DatetimeIndex:
    """Last trading day of each period (default: calendar week ending Friday)."""
    days = pd.Series(trading_days, index=trading_days)
    return pd.DatetimeIndex(days.groupby(days.index.to_period(freq)).last().to_numpy())


def abnormal_inventory_change(stocks: pd.Series, years: int = 5) -> pd.Series:
    """Weekly log change in stocks (%) minus its mean in the same ISO week of the previous years.

    Inventories have a strong refinery-driven seasonal pattern. The seasonal mean uses
    strictly earlier years only, so the series is known at each week-ending date.
    """
    change = 100.0 * np.log(stocks).diff()
    iso = stocks.index.isocalendar()
    frame = pd.DataFrame(
        {"change": change, "year": iso.year.to_numpy(), "week": iso.week.clip(upper=52).to_numpy()},
        index=stocks.index,
    )
    table = frame.pivot_table(index="year", columns="week", values="change", aggfunc="mean")
    seasonal = []
    for year, week in zip(frame["year"], frame["week"], strict=True):
        past = table.loc[(table.index < year) & (table.index >= year - years), week]
        seasonal.append(past.mean() if past.notna().sum() == years else np.nan)
    return (frame["change"] - np.asarray(seasonal)).rename("inv")


def align_release(weekly: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 6) -> pd.Series:
    """Latest weekly value already published at each decision date.

    EIA releases the Weekly Petroleum Status Report on the Wednesday after the
    week-ending Friday (Thursday in holiday weeks). Treating a week as known only
    `lag_days` after its end date is conservative for every regular release.
    """
    published = pd.DataFrame({"available": weekly.index + pd.Timedelta(days=lag_days), "value": weekly.to_numpy()})
    published = published.dropna().sort_values("available")
    target = pd.DataFrame({"date": dates})
    merged = pd.merge_asof(target, published, left_on="date", right_on="available", direction="backward")
    return pd.Series(merged["value"].to_numpy(), index=dates, name=weekly.name)


def weekly_panel(
    rolled: pd.DataFrame, slope: pd.Series, inventory: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 6
) -> pd.DataFrame:
    """VAR input observed at each decision date (all in percent).

    r     : log return of the rolled contract from the previous decision close
    slope : annualised C2/C4 roll yield at the decision close
    inv   : abnormal inventory change of the latest published week
    """
    cumulative = rolled["logret"].cumsum()
    panel = pd.DataFrame(index=dates)
    panel["inv"] = align_release(inventory, dates, lag_days)
    panel["slope"] = 100.0 * slope.reindex(dates)
    panel["r"] = 100.0 * cumulative.reindex(dates).diff()
    return panel.dropna()

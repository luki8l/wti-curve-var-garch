"""Perturbing data after a cut-off date must not change anything decided before it."""

import numpy as np
import pandas as pd

from wti import garch, varmodel
from wti.backtest import Costs, simulate
from wti.features import abnormal_inventory_change, align_release


def make_panel(seed=0, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-07", periods=n, freq="W-FRI")
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + rng.normal(size=3)
    return pd.DataFrame(y, index=idx, columns=["inv", "slope", "r"])


def test_var_forecasts_ignore_future_data():
    panel = make_panel()
    cut = panel.index[300]
    shocked = panel.copy()
    shocked.loc[shocked.index > cut] += 50.0
    a = varmodel.walk_forward(panel, 2, panel.index[150])
    b = varmodel.walk_forward(shocked, 2, panel.index[150])
    assert np.allclose(a.loc[:cut], b.loc[:cut])
    assert not np.allclose(a.loc[cut:].iloc[1:], b.loc[cut:].iloc[1:])


def test_garch_forecasts_ignore_future_data():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2000-01-03", periods=1500)
    y = pd.Series(rng.standard_t(6, len(idx)), index=idx)
    cut = idx[1200]
    shocked = y.copy()
    shocked.loc[shocked.index > cut] *= 4.0
    a = garch.walk_forward(y, "GJR-t", idx[1000], refit_every=50)
    b = garch.walk_forward(shocked, "GJR-t", idx[1000], refit_every=50)
    assert np.allclose(a.loc[:cut, "h1"], b.loc[:cut, "h1"])


def test_positions_start_after_execution_lag():
    idx = pd.bdate_range("2020-01-01", periods=30)
    daily = pd.DataFrame({"ret": 0.01, "roll": False, "price": 50.0, "rf": 0.0}, index=idx)
    targets = pd.Series([1.0, -1.0], index=[idx[4], idx[9]])
    bt = simulate(daily, targets, lag=1, costs=Costs(0, 0.01, 0))
    assert (bt["pos"].iloc[:5] == 0).all()
    assert bt["pos"].iloc[5] == 1.0 and bt["pos"].iloc[10] == -1.0
    # exposure decided on day 4 earns returns only from day 6 on
    assert bt["gross"].iloc[5] == 0.0 and np.isclose(bt["gross"].iloc[6], 0.01)


def test_inventory_week_is_used_only_after_release():
    weeks = pd.date_range("2010-01-01", periods=10, freq="W-FRI")
    stocks = pd.Series(np.arange(10.0), index=weeks)
    thursday_after = weeks[3] + pd.Timedelta(days=6)
    aligned = align_release(stocks, pd.DatetimeIndex([thursday_after - pd.Timedelta(days=1), thursday_after]))
    assert aligned.iloc[0] == 2.0 and aligned.iloc[1] == 3.0


def test_seasonal_mean_uses_prior_years_only():
    weeks = pd.date_range("2000-01-07", "2010-12-31", freq="W-FRI")
    rng = np.random.default_rng(5)
    stocks = pd.Series(300_000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(weeks)))), index=weeks)
    a = abnormal_inventory_change(stocks)
    shocked = stocks.copy()
    shocked.loc["2009":] *= 1.5
    b = abnormal_inventory_change(shocked)
    assert np.allclose(a.loc[:"2008"].dropna(), b.loc[:"2008"].dropna())

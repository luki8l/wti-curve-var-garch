import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from wti import evaluation, metrics, varmodel
from wti.backtest import Costs, simulate

from .test_no_lookahead import make_panel


def test_var_ols_and_irf_match_statsmodels():
    panel = make_panel(seed=2)
    B, sigma = varmodel.ols(panel.to_numpy(), 2)
    ref = VAR(panel).fit(2)
    assert np.allclose(B, ref.params.to_numpy())
    assert np.allclose(sigma, ref.sigma_u.to_numpy())
    assert np.allclose(varmodel.irf(B, sigma, 2, 10), ref.irf(10).orth_irfs)


def test_costs_on_trades_and_rolls():
    idx = pd.bdate_range("2020-01-01", periods=6)
    daily = pd.DataFrame(
        {"ret": 0.0, "roll": [False, False, False, True, False, False], "price": 50.0, "rf": 0.0}, index=idx
    )
    bt = simulate(daily, pd.Series([2.0], index=[idx[0]]), lag=1, costs=Costs(1.0, 0.01, 0.0025))
    rate = 0.0125 / 50.0
    assert np.isclose(bt["cost"].iloc[1], 2.0 * rate)  # buy 2x
    assert np.isclose(bt["cost"].iloc[3], 2 * 2.0 * rate)  # roll: sell old, buy new
    assert np.isclose(bt["cost"].sum(), 6.0 * rate)


def test_drawdown_and_cagr():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.20, 0.0], index=idx)
    assert np.isclose(metrics.drawdown(r).min(), -0.5)
    assert np.isclose(metrics.wealth(r).iloc[-1], 1.1 * 0.5 * 1.2)


def test_sharpe_inference_is_sensible():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=2520)
    good = pd.Series(rng.normal(0.0008, 0.01, len(idx)), index=idx)
    assert metrics.probabilistic_sharpe(good) > 0.95
    many_trials = rng.normal(0, 0.5, 200)
    assert metrics.deflated_sharpe(good, many_trials)["dsr"] < metrics.probabilistic_sharpe(good)


def test_var_backtest_counts_hits():
    r = pd.Series([-3.0, 0.1, 0.2, -2.5, 0.3] * 20)
    var = pd.Series([-2.0] * 100)
    res = evaluation.var_backtest(r, var, 0.05)
    assert res["hits"] == 40 and res["kupiec_pvalue"] < 0.01

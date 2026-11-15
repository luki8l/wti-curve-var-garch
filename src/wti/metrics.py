"""Performance and risk statistics. Inputs are daily simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DAYS = 252
EULER_GAMMA = 0.5772156649


def wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    w = wealth(returns)
    return w / w.cummax() - 1.0


def cagr(returns: pd.Series) -> float:
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    return float(wealth(returns).iloc[-1] ** (1.0 / years) - 1.0)


def sharpe(excess: pd.Series) -> float:
    return float(excess.mean() / excess.std() * np.sqrt(DAYS))


def sharpe_se(excess: pd.Series) -> float:
    """Standard error of the per-period Sharpe ratio under non-normal iid returns (Mertens 2002)."""
    sr = excess.mean() / excess.std()
    g3, g4 = stats.skew(excess), stats.kurtosis(excess, fisher=False)
    return float(np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (len(excess) - 1)))


def probabilistic_sharpe(excess: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period SR > benchmark_sr), Bailey and Lopez de Prado (2012)."""
    sr = excess.mean() / excess.std()
    return float(stats.norm.cdf((sr - benchmark_sr) / sharpe_se(excess)))


def deflated_sharpe(excess: pd.Series, trial_sharpes: np.ndarray) -> dict:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014).

    `trial_sharpes` are the annualised Sharpe ratios of every configuration tried.
    The hurdle is the expected maximum Sharpe of that many unskilled trials.
    """
    n_trials = len(trial_sharpes)
    var_trials = np.var(np.asarray(trial_sharpes) / np.sqrt(DAYS), ddof=1)
    hurdle = np.sqrt(var_trials) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return {
        "n_trials": n_trials,
        "hurdle_sharpe_ann": float(hurdle * np.sqrt(DAYS)),
        "dsr": probabilistic_sharpe(excess, hurdle),
    }


def alpha_beta(excess: pd.Series, bench_excess: pd.Series, lags: int = 5) -> dict:
    data = pd.concat([excess, bench_excess], axis=1).dropna()
    res = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_ann": float(res.params.iloc[0] * DAYS),
        "alpha_t": float(res.tvalues.iloc[0]),
        "beta": float(res.params.iloc[1]),
    }


def summary(bt: pd.DataFrame, bench: pd.DataFrame | None = None) -> dict:
    """Headline statistics of a backtest frame (columns: excess, total, pos, cost, turnover)."""
    excess, total = bt["excess"], bt["total"]
    weekly = (1 + excess).resample("W-FRI").prod() - 1
    weekly = weekly[weekly != 0]
    years = (excess.index[-1] - excess.index[0]).days / 365.25
    out = {
        "cagr": cagr(total),
        "ann_return": float(excess.mean() * DAYS),
        "ann_vol": float(excess.std() * np.sqrt(DAYS)),
        "sharpe": sharpe(excess),
        "sharpe_se": sharpe_se(excess) * np.sqrt(DAYS),
        "max_dd": float(drawdown(total).min()),
        "calmar": cagr(total) / abs(float(drawdown(total).min())),
        "hit_rate_w": float((weekly > 0).mean()),
        "avg_win_w": float(weekly[weekly > 0].mean()),
        "avg_loss_w": float(weekly[weekly < 0].mean()),
        "skew": float(stats.skew(excess)),
        "kurt": float(stats.kurtosis(excess)),
        "var95": float(-np.quantile(excess, 0.05)),
        "es95": float(-excess[excess <= np.quantile(excess, 0.05)].mean()),
        "var99": float(-np.quantile(excess, 0.01)),
        "es99": float(-excess[excess <= np.quantile(excess, 0.01)].mean()),
        "turnover": float(bt["turnover"].sum() / years),
        "cost_drag": float(bt["cost"].sum() / years),
        "net_exposure": float(bt["pos"].mean()),
        "gross_exposure": float(bt["pos"].abs().mean()),
        "time_long": float((bt["pos"] > 0).mean()),
    }
    if bench is not None:
        out.update(alpha_beta(excess, bench["excess"]))
    return out

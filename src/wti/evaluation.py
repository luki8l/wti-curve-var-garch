"""Forecast evaluation and Value-at-Risk backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _hac_mean_test(x: pd.Series, lags: int) -> tuple[float, float]:
    """t-statistic of the mean of x with Newey-West standard errors."""
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0]), float(res.params[0])


def oos_r2(actual: pd.Series, model: pd.Series, bench: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R^2 against a benchmark forecast."""
    return float(1.0 - ((actual - model) ** 2).sum() / ((actual - bench) ** 2).sum())


def clark_west(actual: pd.Series, model: pd.Series, bench: pd.Series, lags: int = 4) -> dict:
    """Clark-West (2007) test of equal MSPE for nested models (one-sided)."""
    adj = (actual - bench) ** 2 - ((actual - model) ** 2 - (bench - model) ** 2)
    t, _ = _hac_mean_test(adj, lags)
    return {"cw_t": t, "cw_pvalue": float(1.0 - stats.norm.cdf(t))}


def pesaran_timmermann(actual: pd.Series, forecast: pd.Series) -> dict:
    """Pesaran-Timmermann (1992) test of directional accuracy (one-sided)."""
    a, f = (actual > 0).astype(float), (forecast > 0).astype(float)
    n = len(a)
    p_hat = float((a == f).mean())
    pa, pf = a.mean(), f.mean()
    p_star = pa * pf + (1 - pa) * (1 - pf)
    var_p = p_star * (1 - p_star) / n
    var_star = (
        (2 * pa - 1) ** 2 * pf * (1 - pf) / n
        + (2 * pf - 1) ** 2 * pa * (1 - pa) / n
        + 4 * pa * pf * (1 - pa) * (1 - pf) / n**2
    )
    stat = (p_hat - p_star) / np.sqrt(var_p - var_star)
    return {"hit_rate": p_hat, "pt_stat": float(stat), "pt_pvalue": float(1.0 - stats.norm.cdf(stat))}


def qlike(realised_sq: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """QLIKE loss, robust to noise in the squared-return proxy (Patton 2011)."""
    return np.log(forecast_var) + realised_sq / forecast_var


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int = 5) -> dict:
    """DM test of equal expected loss; negative t means model a has lower loss."""
    t, mean = _hac_mean_test(loss_a - loss_b, lags)
    return {"dm_mean_diff": mean, "dm_t": t, "dm_pvalue": float(2 * (1 - stats.norm.cdf(abs(t))))}


def var_backtest(returns: pd.Series, var: pd.Series, alpha: float) -> dict:
    """Kupiec unconditional coverage and Christoffersen independence tests.

    `var` is the (negative) return quantile; a hit is a return below it.
    """
    hits = (returns < var).astype(int).to_numpy()
    n, x = len(hits), int(hits.sum())
    pi = x / n

    def ll(p: float, k: int, m: int) -> float:
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k * np.log(p) + (m - k) * np.log(1 - p)

    lr_uc = -2 * (ll(alpha, x, n) - ll(pi, x, n))
    prev, curr = hits[:-1], hits[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi_all = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (ll(pi_all, n01 + n11, n00 + n01 + n10 + n11) - (ll(pi0, n01, n00 + n01) + ll(pi1, n11, n10 + n11)))
    return {
        "alpha": alpha,
        "n": n,
        "hits": x,
        "hit_rate": pi,
        "kupiec_lr": lr_uc,
        "kupiec_pvalue": float(1 - stats.chi2.cdf(lr_uc, 1)),
        "christoffersen_lr": lr_ind,
        "christoffersen_pvalue": float(1 - stats.chi2.cdf(lr_ind, 1)),
        "cc_pvalue": float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2)),
    }

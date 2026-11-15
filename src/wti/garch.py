"""Univariate volatility models: specification search and walk-forward forecasts.

Returns are daily log returns in percent. Forecasts are made at the close of day
t with parameters estimated on data strictly before the last refit date, and
the GARCH recursion only uses returns up to t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

SPECS = {
    "GARCH-N": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    "GARCH-t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "GJR-t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "EGARCH-t": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "GJR-skewt": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "skewt"},
}


def fit(returns: pd.Series, spec: str):
    return arch_model(returns, mean="Constant", **SPECS[spec]).fit(disp="off")


def compare_specs(returns: pd.Series, specs: list[str] | None = None) -> pd.DataFrame:
    """Log-likelihood and information criteria of each specification on one sample."""
    rows = {}
    for name in specs or list(SPECS):
        res = fit(returns, name)
        params = res.params
        persistence = params.get("alpha[1]", 0.0) + params.get("beta[1]", 0.0) + 0.5 * params.get("gamma[1]", 0.0)
        if SPECS[name]["vol"] == "EGARCH":
            persistence = params["beta[1]"]
        rows[name] = {
            "loglik": res.loglikelihood,
            "aic": res.aic,
            "bic": res.bic,
            "persistence": persistence,
            "gamma": params.get("gamma[1]", np.nan),
            "gamma_t": res.tvalues.get("gamma[1]", np.nan),
            "nu": params.get("nu", params.get("eta", np.nan)),
        }
    return pd.DataFrame(rows).T


def walk_forward(
    returns: pd.Series, spec: str, start: str | pd.Timestamp, refit_every: int = 21, horizon: int = 5
) -> pd.DataFrame:
    """Out-of-sample variance forecasts from an expanding estimation window.

    Every `refit_every` days the model is re-estimated on all returns before the
    refit date; until the next refit the parameters are held fixed and the
    variance is filtered forward. Row t holds forecasts made at the close of t:
    `h1` for day t+1 and `hsum` for the sum over days t+1..t+horizon.
    """
    y = returns.dropna()
    first = y.index.searchsorted(pd.Timestamp(start))
    blocks = []
    for k in range(first, len(y), refit_every):
        end = min(k + refit_every, len(y))
        model = arch_model(y.iloc[:end], mean="Constant", **SPECS[spec])
        res = model.fit(last_obs=y.index[k], disp="off")
        fc = res.forecast(horizon=horizon, start=y.index[k], reindex=False)
        block = pd.DataFrame(index=fc.variance.index)
        block["h1"] = fc.variance["h.1"]
        block["hsum"] = fc.variance.sum(axis=1)
        block["mu"] = res.params["mu"]
        block["nu"] = res.params.get("nu", np.inf)
        blocks.append(block)
    return pd.concat(blocks)


def ewma_variance(returns: pd.Series, lam: float = 0.94, seed_days: int = 60) -> pd.Series:
    """RiskMetrics forecast for t+1 made at t (seeded with the first `seed_days` returns)."""
    r2 = returns.pow(2).to_numpy()
    out = np.empty_like(r2)
    var = r2[:seed_days].mean()
    for t, x in enumerate(r2):
        var = lam * var + (1 - lam) * x
        out[t] = var
    return pd.Series(out, index=returns.index, name="ewma")


def rolling_variance(returns: pd.Series, window: int = 63) -> pd.Series:
    """Historical variance of the last `window` days as forecast for t+1."""
    return returns.pow(2).rolling(window).mean().rename(f"roll{window}")


def t_quantile(alpha: float, nu: np.ndarray | float) -> np.ndarray:
    """Quantile of a unit-variance Student-t (nu = inf gives the normal)."""
    nu = np.asarray(nu, dtype=float)
    q = np.where(np.isfinite(nu), stats.t.ppf(alpha, np.where(np.isfinite(nu), nu, 30.0)), stats.norm.ppf(alpha))
    scale = np.where(np.isfinite(nu), np.sqrt((nu - 2.0) / np.where(np.isfinite(nu), nu, 3.0)), 1.0)
    return q * scale


def t_expected_shortfall(alpha: float, nu: np.ndarray | float) -> np.ndarray:
    """Left-tail expected shortfall (as a negative number) of a unit-variance Student-t."""
    nu = np.asarray(nu, dtype=float)
    x = stats.t.ppf(alpha, nu)
    es_std_t = -(stats.t.pdf(x, nu) / alpha) * (nu + x**2) / (nu - 1.0)
    return es_std_t * np.sqrt((nu - 2.0) / nu)

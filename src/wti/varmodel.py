"""Reduced-form VAR: estimation, robust Granger tests, impulse responses, walk-forward forecasts.

Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + u_t, estimated equation by equation with OLS.
Variable order in the panel is also the Cholesky order used for orthogonalised IRFs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR


def design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Regressor matrix [1, Y_{t-1}, ..., Y_{t-p}] and targets Y_t."""
    T = Y.shape[0]
    lags = [Y[p - i : T - i] for i in range(1, p + 1)]
    X = np.hstack([np.ones((T - p, 1)), *lags])
    return X, Y[p:]


def ols(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients B ((1 + k p) x k) and residual covariance (dof-adjusted)."""
    X, target = design(Y, p)
    B, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ B
    sigma = resid.T @ resid / (len(target) - X.shape[1])
    return B, sigma


def forecast_next(Y: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    x = np.concatenate([[1.0], *[Y[-i] for i in range(1, p + 1)]])
    return x @ B


def walk_forward(
    panel: pd.DataFrame, p: int, start: str | pd.Timestamp, window: int | None = None, target: str = "r"
) -> pd.Series:
    """One-step-ahead forecasts of `target` made at each date t >= start.

    The VAR is re-estimated every period on data up to and including t (expanding,
    or the last `window` observations), so the forecast for t+1 uses no later data.
    """
    Y = panel.to_numpy()
    col = panel.columns.get_loc(target)
    first = panel.index.searchsorted(pd.Timestamp(start))
    out = {}
    for t in range(first, len(panel)):
        lo = 0 if window is None else max(0, t + 1 - window)
        sample = Y[lo : t + 1]
        B, _ = ols(sample, p)
        out[panel.index[t]] = forecast_next(sample, B, p)[col]
    return pd.Series(out, name=f"{target}_hat")


def select_order(panel: pd.DataFrame, max_lags: int) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = VAR(panel).select_order(max_lags)
    table = pd.DataFrame(result.ics)
    table.index.name = "lags"
    return table


def ma_coefficients(B: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """MA(inf) matrices Psi_0..Psi_H of the VAR."""
    k = B.shape[1]
    A = [B[1 + i * k : 1 + (i + 1) * k].T for i in range(p)]
    psi = [np.eye(k)]
    for h in range(1, horizon + 1):
        psi.append(sum(A[i] @ psi[h - 1 - i] for i in range(min(p, h))))
    return np.array(psi)


def irf(B: np.ndarray, sigma: np.ndarray, p: int, horizon: int, kind: str = "cholesky") -> np.ndarray:
    """Responses [h, variable, shock] to one-standard-deviation shocks.

    'cholesky': orthogonalised, recursive order = column order.
    'generalized': Pesaran-Shin, invariant to the ordering.
    """
    psi = ma_coefficients(B, p, horizon)
    if kind == "cholesky":
        return psi @ np.linalg.cholesky(sigma)
    if kind == "generalized":
        return psi @ sigma / np.sqrt(np.diag(sigma))
    raise ValueError(kind)


def wild_bootstrap_irf(
    panel: pd.DataFrame, p: int, horizon: int, kind: str = "cholesky", draws: int = 999, seed: int = 0
) -> np.ndarray:
    """Recursive-design wild bootstrap (Rademacher) for IRF bands.

    Residuals of oil-market VARs are conditionally heteroskedastic (GARCH effects),
    which the usual iid residual bootstrap ignores. Returns draws [n, h, var, shock].
    """
    rng = np.random.default_rng(seed)
    Y = panel.to_numpy()
    B, _ = ols(Y, p)
    X, target = design(Y, p)
    resid = target - X @ B
    k = Y.shape[1]
    out = np.empty((draws, horizon + 1, k, k))
    for d in range(draws):
        eta = rng.choice([-1.0, 1.0], size=(len(resid), 1))
        Ystar = Y.copy()
        for t in range(p, len(Y)):
            x = np.concatenate([[1.0], *[Ystar[t - i] for i in range(1, p + 1)]])
            Ystar[t] = x @ B + eta[t - p] * resid[t - p]
        Bs, Ss = ols(Ystar, p)
        out[d] = irf(Bs, Ss, p, horizon, kind)
    return out


def granger_robust(panel: pd.DataFrame, p: int, target: str = "r") -> pd.DataFrame:
    """Does each variable help predict `target`? Wald tests with HC3 standard errors."""
    names = list(panel.columns)
    lagged = pd.concat({f"{n}_L{i}": panel[n].shift(i) for i in range(1, p + 1) for n in names}, axis=1)
    data = pd.concat([panel[target].rename("y"), lagged], axis=1).dropna()
    model = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit(cov_type="HC3")

    def wald(columns: list[str]) -> dict:
        test = model.wald_test(", ".join(f"{c} = 0" for c in columns), scalar=True)
        return {"wald": float(test.statistic), "df": len(columns), "pvalue": float(test.pvalue)}

    rows = {n: wald([f"{n}_L{i}" for i in range(1, p + 1)]) for n in names}
    rows["all"] = wald(list(lagged.columns))
    return pd.DataFrame(rows).T


def lag_sum_test(panel: pd.DataFrame, p: int, target: str, regressor: str, null: float) -> dict:
    """Long-run coefficient sum_i b_i of `regressor` lags in the `target` equation (HC3).

    For target r (weekly %) and regressor slope (% p.a.), full pass-through of the
    roll yield into expected futures returns implies a sum of 1/52.
    """
    names = list(panel.columns)
    lagged = pd.concat({f"{n}_L{i}": panel[n].shift(i) for i in range(1, p + 1) for n in names}, axis=1)
    data = pd.concat([panel[target].rename("y"), lagged], axis=1).dropna()
    model = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit(cov_type="HC3")
    weights = pd.Series(0.0, index=model.params.index)
    weights[[f"{regressor}_L{i}" for i in range(1, p + 1)]] = 1.0
    est = float(weights @ model.params)
    se = float(np.sqrt(weights @ model.cov_params() @ weights))
    return {
        "sum": est,
        "se": se,
        "t_zero": est / se,
        "p_zero": float(2 * (1 - stats.norm.cdf(abs(est / se)))),
        "null": null,
        "t_null": (est - null) / se,
        "p_null": float(2 * (1 - stats.norm.cdf(abs((est - null) / se)))),
    }


def diagnostics(panel: pd.DataFrame, p: int, lags: int = 12) -> dict:
    """Stability and residual tests of the full-sample statsmodels VAR."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(panel).fit(p)
        whiteness = res.test_whiteness(nlags=lags, adjusted=True)
    roots = np.abs(res.roots)
    ljung_box = {
        f"lb12_p_{c}": float(acorr_ljungbox(res.resid[c], lags=[lags]).lb_pvalue.iloc[0]) for c in panel.columns
    }
    return {
        **ljung_box,
        "stable": bool(res.is_stable()),
        "min_root_modulus": float(roots.min()),
        "portmanteau_stat": float(whiteness.test_statistic),
        "portmanteau_pvalue": float(whiteness.pvalue),
        "nobs": int(res.nobs),
    }

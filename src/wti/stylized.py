"""Stylized facts of daily WTI futures returns (Cont 2001), as model diagnostics."""

from __future__ import annotations

import warnings

import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf


def _excess_kurtosis_at(logret: pd.Series, rule: str) -> float:
    return float(stats.kurtosis(logret.resample(rule).sum().loc[lambda s: s != 0]))


def facts(logret_pct: pd.Series) -> pd.Series:
    """One column of the stylized-facts table for a sample of daily log returns (%)."""
    r = logret_pct.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(r, lags=[10], return_df=True)
        lm_stat, lm_p, *_ = het_arch(r, nlags=10)
        garch_n = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    z = garch_n.std_resid.dropna()
    abs_acf = acf(r.abs(), nlags=100, fft=True)
    return pd.Series(
        {
            "acf1_r": acf(r, nlags=1, fft=True)[1],
            "ljung_box10_p": float(lb["lb_pvalue"].iloc[0]),
            "skew": float(stats.skew(r)),
            "kurt_daily": float(stats.kurtosis(r)),
            "jarque_bera_p": float(stats.jarque_bera(r).pvalue),
            "kurt_weekly": _excess_kurtosis_at(r, "W-FRI"),
            "kurt_monthly": _excess_kurtosis_at(r, "ME"),
            "acf1_abs": abs_acf[1],
            "acf50_abs": abs_acf[50],
            "acf100_abs": abs_acf[100],
            "arch_lm10_p": float(lm_p),
            "kurt_std_resid": float(stats.kurtosis(z)),
            "leverage_corr": float(r.corr(r.abs().shift(-1))),
        }
    )


def acf_table(logret_pct: pd.Series, nlags: int = 100) -> pd.DataFrame:
    r = logret_pct.dropna()
    return pd.DataFrame(
        {"r": acf(r, nlags=nlags, fft=True), "abs_r": acf(r.abs(), nlags=nlags, fft=True)},
        index=pd.RangeIndex(0, nlags + 1, name="lag"),
    )

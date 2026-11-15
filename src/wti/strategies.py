"""Target exposures of the candidate strategies at each decision date.

The candidates form a 3 x 2 design: direction {always long, sign of the VAR
return forecast, sign of the curve slope} x sizing {1x, GARCH volatility target}.
The main strategy (carry_vol) was selected on the 1996-2009 validation period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wti.backtest import vol_scale

NAMES = {
    "benchmark": "Long WTI, 1x (benchmark)",
    "long_vol": "Long WTI, vol-targeted",
    "var": "VAR signal, 1x",
    "var_vol": "VAR signal, vol-targeted",
    "carry": "Carry signal, 1x",
    "carry_vol": "Carry signal, vol-targeted",
}
MAIN = "carry_vol"


def carry_direction(slope: pd.Series, band: float = 0.0, long_only: bool = False) -> pd.Series:
    """+1 in backwardation, -1 in contango; flat inside +/- band (same units as slope)."""
    direction = pd.Series(np.where(slope > band, 1.0, np.where(slope < -band, -1.0, 0.0)), index=slope.index)
    return direction.clip(lower=0.0) if long_only else direction


def targets(
    rhat: pd.Series,
    sigma_ann: pd.Series,
    slope: pd.Series,
    target_vol: float,
    max_leverage: float,
    signal: str = "sign",
    rhat_scale: pd.Series | None = None,
) -> pd.DataFrame:
    """Exposures for all candidates on the common decision dates.

    signal='sign' uses the direction of the VAR forecast only; signal='scaled' sizes
    it by |forecast| / rhat_scale (capped at 1), a robustness variant.
    """
    index = rhat.index.intersection(sigma_ann.dropna().index).intersection(slope.index)
    rhat, sigma_ann, slope = rhat.loc[index], sigma_ann.loc[index], slope.loc[index]
    scale = vol_scale(sigma_ann, target_vol, max_leverage)
    if signal == "sign":
        var_direction = np.sign(rhat)
    elif signal == "scaled":
        var_direction = (rhat / rhat_scale.reindex(index)).clip(-1.0, 1.0).fillna(0.0)
    else:
        raise ValueError(signal)
    carry = carry_direction(slope)
    return pd.DataFrame(
        {
            "benchmark": 1.0,
            "long_vol": scale,
            "var": var_direction,
            "var_vol": var_direction * scale,
            "carry": carry,
            "carry_vol": carry * scale,
        },
        index=index,
    )

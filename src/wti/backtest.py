"""Daily backtest of weekly target exposures in the rolled WTI contract.

Timing: a target computed at the close of decision day d is traded at the
settlement of the `lag`-th trading day after d and earns returns from the next
day on. Position notional is a multiple of capital; capital sits in T-bills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Per-side trading costs of one CL contract (1,000 bbl), in USD per barrel.

    slippage_ticks: execution against the settlement price, in 0.01 USD ticks.
    commission: broker and exchange fees per barrel (2.50 USD per contract = 0.0025).
    """

    slippage_ticks: float = 1.0
    tick: float = 0.01
    commission: float = 0.0025

    def rate(self, price: pd.Series) -> pd.Series:
        """Cost as a fraction of traded notional; price-dependent because ticks are fixed in USD."""
        return (self.slippage_ticks * self.tick + self.commission) / price


def daily_frame(rolled: pd.DataFrame, tbill: pd.Series) -> pd.DataFrame:
    """Rolled futures returns plus the T-bill return earned on collateral (actual/360)."""
    frame = rolled[["ret", "roll", "price"]].copy()
    rate = tbill.reindex(frame.index.union(tbill.index)).ffill().reindex(frame.index) / 100.0
    days = frame.index.to_series().diff().dt.days.fillna(1).to_numpy()
    frame["rf"] = rate.shift().bfill().to_numpy() * days / 360.0
    return frame


def vol_scale(sigma_ann: pd.Series, target: float, cap: float) -> pd.Series:
    """Exposure that targets `target` annualised volatility, capped at `cap` times capital."""
    return (target / sigma_ann).clip(upper=cap)


def simulate(daily: pd.DataFrame, targets: pd.Series, lag: int = 1, costs: Costs | None = None) -> pd.DataFrame:
    """Run a target-exposure series through the execution and cost model.

    targets: exposure (multiple of capital, sign = direction) indexed by decision dates.
    Returns daily columns pos (held at the close), gross, cost, excess, total, turnover.
    """
    costs = costs or Costs()
    index = daily.index
    decision_pos = index.get_indexer(targets.index)
    if (decision_pos < 0).any():
        raise ValueError("decision dates must be trading days of the daily frame")
    exec_pos = decision_pos + lag
    keep = exec_pos < len(index)
    pos = pd.Series(np.nan, index=index)
    pos.iloc[exec_pos[keep]] = targets.to_numpy()[keep]
    pos = pos.ffill().fillna(0.0)

    held = pos.shift().fillna(0.0)
    trade = (pos - held).abs()
    roll_trade = 2.0 * held.abs() * daily["roll"].astype(float)
    turnover = trade + roll_trade
    cost = turnover * costs.rate(daily["price"])

    out = pd.DataFrame(index=index)
    out["pos"] = pos
    out["gross"] = held * daily["ret"]
    out["cost"] = cost
    out["excess"] = out["gross"] - cost
    out["total"] = daily["rf"] + out["excess"]
    out["turnover"] = turnover
    return out

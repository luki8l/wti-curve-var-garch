"""NYMEX WTI (CL) contract calendar, data validation and the rolled return series.

EIA publishes settlements by position on the curve (contract 1-4), not by
delivery month. After the front contract expires every label shifts by one, so
the return of a held contract has to be computed across labels on that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cl_last_trade_date(delivery: pd.Period, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Last trading day of the CL contract for a delivery month.

    CME rule: trading terminates 3 business days before the 25th calendar day of
    the month prior to delivery; if the 25th is not a business day, 4 business
    days before the 25th. Business days are the observed settlement dates, so
    exchange holidays are handled implicitly.
    """
    prior = delivery - 1
    reference = pd.Timestamp(prior.year, prior.month, 25)
    before = calendar[calendar < reference]
    return before[-3 if reference in calendar else -4]


def roll_days(calendar: pd.DatetimeIndex, observed: pd.DatetimeIndex | None = None) -> pd.DatetimeIndex:
    """First day after each CL expiry on which the shifted labels are observed.

    `calendar` is the full exchange calendar (used for the rule); `observed` are the
    days with a complete curve. If the first day after expiry has no complete curve,
    the shift becomes visible on the next complete day.
    """
    observed = calendar if observed is None else observed
    first = calendar[0].to_period("M") + 1
    last = calendar[-1].to_period("M") + 1
    days = []
    for delivery in pd.period_range(first, last, freq="M"):
        prior = delivery - 1
        if (calendar < pd.Timestamp(prior.year, prior.month, 25)).sum() < 4:
            continue  # calendar starts too late in that month to apply the rule
        expiry = cl_last_trade_date(delivery, calendar)
        later = observed[observed > expiry]
        if len(later) and later[0] != observed[0]:
            days.append(later[0])
    return pd.DatetimeIndex(days)


def roll_diagnostics(curve: pd.DataFrame, rolls: pd.DatetimeIndex) -> pd.DataFrame:
    """Check the calendar rule against the data on every roll day.

    On a roll day today's c2-c3 spread continues yesterday's c3-c4 spread; on other
    days it continues yesterday's c2-c3 spread. `score` < 0 means the data favour a
    roll on that day. `info` is |c2-c3 - (c3-c4)| the day before: when the curve is
    linear both hypotheses look alike, and a misplaced roll would also cost little.
    The front contract is left out because it is distorted around expiry.
    """
    s23 = curve["c2"] - curve["c3"]
    s34 = curve["c3"] - curve["c4"]
    score = (s23 - s34.shift()).abs() - (s23 - s23.shift()).abs()
    info = (s23.shift() - s34.shift()).abs()
    pos = curve.index.get_indexer(rolls)
    return pd.DataFrame(
        {
            "score": score.iloc[pos].to_numpy(),
            "score_day_before": score.iloc[pos - 1].to_numpy(),
            "score_day_after": score.iloc[np.minimum(pos + 1, len(curve) - 1)].to_numpy(),
            "info": info.iloc[pos].to_numpy(),
        },
        index=rolls,
    )


def matched_log_returns(curve: pd.DataFrame, rolls: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily log returns of the contracts quoted yesterday as c2, c3, c4 (same contract)."""
    logp = np.log(curve.clip(lower=1e-6))
    is_roll = curve.index.isin(rolls)
    out = {}
    for k in (2, 3, 4):
        today = logp[f"c{k}"].where(~is_roll, logp[f"c{k - 1}"])
        out[f"c{k}"] = today - logp[f"c{k}"].shift()
    return pd.DataFrame(out)


def isolated_spikes(curve: pd.DataFrame, rolls: pd.DatetimeIndex, threshold: float = 5.0) -> pd.DataFrame:
    """Flag single-day recording errors in contracts 2-4.

    Contracts on one curve move almost in parallel. A recording error shows up as one
    contract jumping away from the others and jumping back the next day. The jump is
    measured in dollars relative to the typical calendar spread of the previous month,
    so genuine dislocations (April 2020) are not flagged.
    """
    matched = matched_log_returns(curve, rolls)
    deviation = matched.sub(matched.median(axis=1), axis=0).mul(curve[["c2", "c3", "c4"]].shift())
    spread = pd.concat([(curve["c2"] - curve["c3"]).abs(), (curve["c3"] - curve["c4"]).abs()], axis=1)
    scale = spread.mean(axis=1).rolling(21, min_periods=5).median().shift().clip(lower=0.05)
    z = deviation.div(scale, axis=0)
    flagged = (z.abs() > threshold) & (z.shift(-1).abs() > threshold) & (np.sign(z) != np.sign(z.shift(-1)))
    hits = flagged.stack()
    hits = hits[hits]
    return pd.DataFrame([{"date": d, "contract": k, "price": curve.at[d, k], "z": z.at[d, k]} for d, k in hits.index])


def clean_curve(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """Complete-curve days with isolated recording errors removed.

    Returns the cleaned curve, the roll days and the table of removed observations.
    """
    complete = raw.dropna()
    rolls = roll_days(raw.index, complete.index)
    spikes = isolated_spikes(complete, rolls)
    if len(spikes):
        complete = complete.drop(index=pd.DatetimeIndex(spikes["date"]).unique())
        rolls = roll_days(raw.index, complete.index)
    return complete, rolls, spikes


def rolled_returns(curve: pd.DataFrame, rolls: pd.DatetimeIndex, hold: int = 2) -> pd.DataFrame:
    """Daily returns of holding the `hold`-th nearby contract, rolled when labels shift.

    On a roll day the contract held overnight is quoted one label closer to the front,
    so its return is c{hold-1}_t / c{hold}_{t-1} - 1. At that close the position moves
    into the new c{hold}; this is when roll costs are charged. Holding the second
    contract keeps the strategy out of the expiry month (April 2020: the front contract
    settled at -37.63 USD while the second contract stayed positive).
    """
    if hold < 2:
        raise ValueError("hold >= 2 required: the front contract cannot be followed past expiry")
    held = curve[f"c{hold}"]
    is_roll = curve.index.isin(rolls)
    today = held.where(~is_roll, curve[f"c{hold - 1}"])
    out = pd.DataFrame(index=curve.index)
    out["price"] = held
    out["roll"] = is_roll
    out["ret"] = today / held.shift() - 1.0
    out["logret"] = np.log(today) - np.log(held.shift())
    return out.iloc[1:]


def roll_yield(curve: pd.DataFrame, near: int = 2, far: int = 4) -> pd.Series:
    """Annualised log slope of the curve; positive = backwardation."""
    months = far - near
    return ((np.log(curve[f"c{near}"]) - np.log(curve[f"c{far}"])) * 12.0 / months).rename("slope")

import numpy as np
import pandas as pd
import pytest

from wti.futures import roll_days


@pytest.fixture
def synthetic_curve():
    """Contracts with known prices, quoted by curve position like the EIA files.

    Every delivery month follows the same random walk plus a fixed calendar spread,
    so the true return of any single contract is known exactly.
    """
    rng = np.random.default_rng(1)
    calendar = pd.bdate_range("2015-01-01", "2016-12-31")
    common = np.exp(np.cumsum(rng.normal(0, 0.01, len(calendar)))) * 50.0
    months = pd.period_range("2015-02", "2017-08", freq="M")
    prices = {m: pd.Series(common + 0.3 * i, index=calendar) for i, m in enumerate(months)}

    rolls = roll_days(calendar)
    front = pd.Series(0, index=calendar)  # index into `months` of the front contract
    for day in rolls:
        front.loc[day:] += 1
    curve = pd.DataFrame(index=calendar, columns=["c1", "c2", "c3", "c4"], dtype=float)
    for day in calendar:
        for j in range(4):
            curve.at[day, f"c{j + 1}"] = prices[months[front.loc[day] + j]].loc[day]
    return {"curve": curve, "rolls": rolls, "prices": prices, "months": months, "front": front}

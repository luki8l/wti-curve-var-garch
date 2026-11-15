import numpy as np
import pandas as pd

from wti.futures import cl_last_trade_date, isolated_spikes, roll_yield, rolled_returns


def test_may_2020_contract_expired_on_21_april():
    # 25 April 2020 was a Saturday, so trading ended 4 business days before it.
    calendar = pd.bdate_range("2020-03-01", "2020-05-31").drop(pd.Timestamp("2020-04-10"))  # Good Friday
    assert cl_last_trade_date(pd.Period("2020-05", "M"), calendar) == pd.Timestamp("2020-04-21")


def test_rule_counts_three_days_when_25th_is_a_business_day():
    calendar = pd.bdate_range("2022-04-01", "2022-06-30")
    # 25 May 2022 was a Wednesday: 24, 23, 20 May -> 20 May
    assert cl_last_trade_date(pd.Period("2022-06", "M"), calendar) == pd.Timestamp("2022-05-20")


def test_rolled_return_follows_one_contract_through_label_shifts(synthetic_curve):
    s = synthetic_curve
    out = rolled_returns(s["curve"], s["rolls"], hold=2)
    # the contract held overnight is the one quoted as c2 the day before
    for day in list(s["rolls"][:6]) + list(out.index[10:15]):
        prev = s["curve"].index[s["curve"].index.get_loc(day) - 1]
        contract = s["months"][s["front"].loc[prev] + 1]
        expected = s["prices"][contract].loc[day] / s["prices"][contract].loc[prev] - 1
        assert np.isclose(out.at[day, "ret"], expected)
    assert out["roll"].sum() == len(s["rolls"])


def test_roll_yield_sign():
    curve = pd.DataFrame({"c1": [52.0], "c2": [51.0], "c3": [50.0], "c4": [49.0]})
    assert roll_yield(curve).iloc[0] > 0  # backwardation
    assert np.isclose(roll_yield(curve).iloc[0], np.log(51 / 49) * 6)


def test_isolated_recording_error_is_flagged(synthetic_curve):
    s = synthetic_curve
    curve = s["curve"].copy()
    bad_day = curve.index[200]
    curve.at[bad_day, "c4"] *= 0.88
    flagged = isolated_spikes(curve, s["rolls"])
    assert list(zip(flagged["date"], flagged["contract"], strict=True)) == [(bad_day, "c4")]
    assert isolated_spikes(s["curve"], s["rolls"]).empty

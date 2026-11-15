"""End-to-end analysis. Every number in the report is produced here.

python -m wti.analysis            # run and cache results in output/
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from wti import evaluation, garch, metrics, stylized, varmodel
from wti.backtest import Costs, daily_frame, simulate
from wti.data import load_curve, load_optional, load_stocks, load_tbill, verify_manifest
from wti.features import abnormal_inventory_change, decision_dates, weekly_panel
from wti.futures import clean_curve, roll_diagnostics, roll_yield, rolled_returns
from wti.paths import CONFIG, OUTPUT, ensure_dirs
from wti.strategies import MAIN, NAMES, carry_direction, targets

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arch")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@dataclass
class Inputs:
    curve: pd.DataFrame
    rolls: pd.DatetimeIndex
    spikes: pd.DataFrame
    rolled: pd.DataFrame
    daily: pd.DataFrame
    slope_daily: pd.Series
    panel: pd.DataFrame
    roll_diag: pd.DataFrame
    ovx: pd.Series | None
    manifest: dict = field(default_factory=dict)


def build_inputs(cfg: dict, slope_far: int | None = None, seasonal_years: int | None = None) -> Inputs:
    manifest = verify_manifest()
    end = pd.Timestamp(cfg["sample"]["end"])
    curve, rolls, spikes = clean_curve(load_curve())
    curve = curve.loc[:end]
    rolls = rolls[rolls <= end]
    rolled = rolled_returns(curve, rolls, cfg["futures"]["hold"])
    far = slope_far or cfg["futures"]["slope_far"]
    slope = roll_yield(curve, cfg["futures"]["slope_near"], far)
    inv = abnormal_inventory_change(load_stocks(), seasonal_years or cfg["inventory"]["seasonal_years"])
    dates = decision_dates(rolled.index)
    panel = weekly_panel(rolled, slope, inv, dates, cfg["inventory"]["release_lag_days"])
    panel = panel[cfg["var"]["variables"]]
    ovx = load_optional("fred_OVXCLS.csv")
    return Inputs(
        curve=curve,
        rolls=rolls,
        spikes=spikes,
        rolled=rolled,
        daily=daily_frame(rolled, load_tbill()),
        slope_daily=slope,
        panel=panel,
        roll_diag=roll_diagnostics(curve, rolls),
        ovx=ovx,
        manifest=manifest,
    )


def period(frame, bounds):
    return frame.loc[pd.Timestamp(bounds[0]) : pd.Timestamp(bounds[1])]


# ----------------------------------------------------------------------------- volatility
def volatility(inp: Inputs, cfg: dict) -> dict:
    y = 100.0 * inp.rolled["logret"]
    is_end = cfg["sample"]["in_sample_end"]
    specs = garch.compare_specs(y.loc[:is_end])
    gcfg = cfg["garch"]
    main = garch.walk_forward(y, gcfg["spec"], gcfg["start"], gcfg["refit_every"], horizon=1)
    normal = garch.walk_forward(y, "GARCH-N", gcfg["start"], gcfg["refit_every"], horizon=1)
    ewma = garch.ewma_variance(y)
    roll63 = garch.rolling_variance(y, 63)
    # forecast made at t for t+1, aligned with the realised square on t+1
    fc = pd.DataFrame(
        {
            gcfg["spec"]: main["h1"],
            "GARCH-N": normal["h1"],
            "EWMA": ewma.reindex(main.index),
            "Rolling 63d": roll63.reindex(main.index),
        }
    ).shift(1)
    realised = (y**2).reindex(fc.index)
    data = pd.concat([fc, realised.rename("rv")], axis=1).dropna()
    evals = {}
    for name, bounds in cfg["periods"].items():
        sub = period(data, bounds)
        losses = {m: evaluation.qlike(sub["rv"], sub[m]) for m in fc.columns}
        rows = {}
        for m in fc.columns:
            row = {"qlike": float(losses[m].mean())}
            if m != gcfg["spec"]:
                row.update(evaluation.diebold_mariano(losses[gcfg["spec"]], losses[m]))
            rows[m] = row
        evals[name] = pd.DataFrame(rows).T
    sigma_daily = np.sqrt(main["h1"]) / 100.0
    ovx_corr = None
    if inp.ovx is not None:
        both = pd.concat([sigma_daily * np.sqrt(252), inp.ovx / 100.0], axis=1, join="inner").dropna()
        ovx_corr = float(both.corr().iloc[0, 1])
    return {
        "specs": specs,
        "forecast": main,
        "sigma_ann": (sigma_daily * np.sqrt(252)).rename("sigma_ann"),
        "eval": evals,
        "ovx_corr": ovx_corr,
    }


def value_at_risk(inp: Inputs, cfg: dict, vol: dict, positions: dict[str, pd.Series]) -> dict:
    """1-day VaR from the GARCH-t forecast, backtested for the long benchmark and the strategy."""
    fc = vol["forecast"].shift(1)  # forecast for day t made at t-1
    logret = inp.rolled["logret"]
    out = {}
    for name, pos in positions.items():
        held = pos.shift(1).reindex(fc.index)
        rows = {}
        for alpha in (0.01, 0.05):
            q = garch.t_quantile(alpha, fc["nu"].to_numpy())
            sigma = np.sqrt(fc["h1"]) / 100.0
            mu = fc["mu"] / 100.0
            # alpha-quantile of w * r with r = mu + sigma * z and z symmetric:
            # w * mu + |w| * sigma * q_alpha for long and short exposure alike
            var = held * mu + held.abs() * sigma * q
            strat = held * logret.reindex(fc.index)
            data = pd.concat([strat.rename("r"), var.rename("var"), held.rename("w")], axis=1).dropna()
            data = data[data["w"] != 0]
            for pname, bounds in cfg["periods"].items():
                sub = period(data, bounds)
                rows[(pname, alpha)] = evaluation.var_backtest(sub["r"], sub["var"], alpha)
        out[name] = pd.DataFrame(rows).T
    return out


# ----------------------------------------------------------------------------- VAR
def var_analysis(inp: Inputs, cfg: dict, horizon: int = 12, draws: int = 999) -> dict:
    vcfg = cfg["var"]
    panel_is = inp.panel.loc[: cfg["sample"]["in_sample_end"]]
    p = vcfg["lags"]
    order = varmodel.select_order(panel_is, 8)
    B, sigma = varmodel.ols(panel_is.to_numpy(), p)
    names = ["const"] + [f"{c}_L{i}" for i in range(1, p + 1) for c in panel_is.columns]
    coef = pd.DataFrame(B, columns=panel_is.columns, index=names)
    granger_is = varmodel.granger_robust(panel_is, p, "r")
    point = {
        "cholesky": varmodel.irf(B, sigma, p, horizon, "cholesky"),
        "generalized": varmodel.irf(B, sigma, p, horizon, "generalized"),
    }
    boot = {k: varmodel.wild_bootstrap_irf(panel_is, p, horizon, k, draws=draws, seed=7) for k in point}
    forecast = varmodel.walk_forward(inp.panel, p, vcfg["start"], vcfg["window"])
    order_fits = {}
    for lags in (1, 2, 4, 8):
        d = varmodel.diagnostics(panel_is, lags)
        order_fits[lags] = {k: d[k] for k in ("lb12_p_inv", "lb12_p_slope", "lb12_p_r", "portmanteau_pvalue")}
    nxt = panel_is["r"].shift(-1)
    regimes = {}
    for label, mask in (("backwardation", panel_is["slope"] > 0), ("contango", panel_is["slope"] <= 0)):
        x = nxt[mask].dropna()
        t, mean = evaluation._hac_mean_test(x, 4)
        regimes[label] = {"weeks": len(x), "mean_next_week": mean, "t": t, "ann": mean * 52}
    return {
        "order_fits": pd.DataFrame(order_fits).T,
        "passthrough": varmodel.lag_sum_test(panel_is, p, "r", "slope", 1 / 52),
        "regimes": pd.DataFrame(regimes).T,
        "order": order,
        "coef": coef,
        "sigma": sigma,
        "granger_is": granger_is,
        "diagnostics": varmodel.diagnostics(panel_is, p),
        "irf": point,
        "irf_boot": boot,
        "forecast": forecast,
        "forecast_eval": forecast_eval(inp.panel, forecast, cfg),
    }


def forecast_eval(panel: pd.DataFrame, forecast: pd.Series, cfg: dict) -> pd.DataFrame:
    """Out-of-sample accuracy of the VAR return forecast against the prevailing mean."""
    actual = panel["r"].shift(-1).reindex(forecast.index)
    prevailing = panel["r"].expanding().mean().reindex(forecast.index)
    data = pd.concat([actual.rename("a"), forecast.rename("f"), prevailing.rename("b")], axis=1).dropna()
    rows = {}
    for name, bounds in cfg["periods"].items():
        sub = period(data, bounds)
        row = {"n": len(sub), "oos_r2": evaluation.oos_r2(sub["a"], sub["f"], sub["b"])}
        row.update(evaluation.clark_west(sub["a"], sub["f"], sub["b"]))
        row.update(evaluation.pesaran_timmermann(sub["a"], sub["f"]))
        row["share_long"] = float((sub["f"] > 0).mean())
        rows[name] = row
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- backtests
def run_strategies(
    inp: Inputs,
    cfg: dict,
    rhat: pd.Series,
    sigma_ann: pd.Series,
    *,
    signal: str = "sign",
    target_vol: float | None = None,
    max_leverage: float | None = None,
    lag: int | None = None,
    costs: Costs | None = None,
    rebalance: str = "W",
    band: float = 0.0,
    long_only: bool = False,
    panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Backtest all candidates; keyword arguments are one-at-a-time robustness deviations."""
    scfg, ecfg = cfg["sizing"], cfg["execution"]
    panel = inp.panel if panel is None else panel
    dates = rhat.index
    if rebalance == "M":
        dates = pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(dates.to_period("M")).last())
    rhat = rhat.reindex(dates)
    scale = 2.0 * rhat.expanding(min_periods=52).std() if signal == "scaled" else None
    tgt = targets(
        rhat,
        sigma_ann.reindex(dates),
        panel["slope"].reindex(dates),
        target_vol or scfg["target_vol"],
        max_leverage or scfg["max_leverage"],
        signal=signal,
        rhat_scale=scale,
    )
    if band or long_only:
        size = tgt["long_vol"]
        tgt["carry_vol"] = carry_direction(panel["slope"].reindex(tgt.index), band, long_only) * size
    costs = costs or Costs(ecfg["slippage_ticks"], 0.01, ecfg["commission_per_bbl"])
    lag = ecfg["lag_days"] if lag is None else lag
    return {name: simulate(inp.daily, tgt[name], lag=lag, costs=costs) for name in tgt.columns}


def summarise(bts: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for pname, bounds in cfg["periods"].items():
        bench = period(bts["benchmark"], bounds)
        out[pname] = pd.DataFrame(
            {
                name: metrics.summary(period(bt, bounds), None if name == "benchmark" else bench)
                for name, bt in bts.items()
            }
        ).T
    return out


STRESS_WINDOWS = {
    "Asian crisis": ("1997-10-01", "1998-12-31"),
    "GFC collapse": ("2008-07-01", "2008-12-31"),
    "OPEC price war": ("2014-06-01", "2016-02-29"),
    "COVID, negative front month": ("2020-01-01", "2020-04-30"),
    "Russia-Ukraine spike": ("2022-02-24", "2022-06-30"),
}


def stress(bts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for label, (a, b) in STRESS_WINDOWS.items():
        row = {"start": a, "end": b}
        for name in ("benchmark", "long_vol", MAIN):
            sub = bts[name].loc[a:b]
            row[f"{name}_ret"] = float((1 + sub["excess"]).prod() - 1)
            row[f"{name}_mdd"] = float(metrics.drawdown(sub["excess"]).min())
        rows[label] = row
    return pd.DataFrame(rows).T


def robustness(inp: Inputs, cfg: dict, vol: dict, var_res: dict) -> pd.DataFrame:
    """One-at-a-time deviations from the frozen configuration.

    Rows with strategy carry_vol vary the main strategy; rows with var_vol vary the
    VAR specification behind the competing VAR-signal strategy.
    """
    rhat, sigma = var_res["forecast"], vol["sigma_ann"]
    vcfg = cfg["var"]
    rows = []

    def record(group: str, label: str, bts: dict, strategy: str = MAIN) -> None:
        s = summarise({"benchmark": bts["benchmark"], strategy: bts[strategy]}, cfg)
        rows.append(
            {
                "strategy": strategy,
                "group": group,
                "variant": label,
                "sharpe_val": s["validation"].loc[strategy, "sharpe"],
                "sharpe_oos": s["oos"].loc[strategy, "sharpe"],
                "maxdd_oos": s["oos"].loc[strategy, "max_dd"],
                "turnover_oos": s["oos"].loc[strategy, "turnover"],
            }
        )

    def run(**kwargs):
        return run_strategies(inp, cfg, kwargs.pop("rhat", rhat), kwargs.pop("sigma", sigma), **kwargs)

    record("Baseline", "Frozen configuration", run())
    alt = build_inputs(cfg, slope_far=3)
    record("Signal", "Slope from C2/C3", run(panel=alt.panel))
    record("Signal", "Neutral zone |slope| < 5% p.a.", run(band=5.0))
    record("Signal", "Long/flat (no shorts)", run(long_only=True))
    record("Rebalancing", "Monthly", run(rebalance="M"))
    for tv in (0.20, 0.50):
        record("Target volatility", f"{tv:.0%}", run(target_vol=tv))
    for cap in (1.0, 3.0):
        record("Leverage cap", f"{cap:.0f}x", run(max_leverage=cap))
    ewma = np.sqrt(garch.ewma_variance(100.0 * inp.rolled["logret"])) / 100.0 * np.sqrt(252)
    record("Volatility model", "EWMA (0.94) instead of EGARCH-t", run(sigma=ewma))
    normal = garch.walk_forward(
        100.0 * inp.rolled["logret"], "GARCH-N", cfg["garch"]["start"], cfg["garch"]["refit_every"], horizon=1
    )
    record("Volatility model", "GARCH(1,1)-normal", run(sigma=np.sqrt(normal["h1"]) / 100.0 * np.sqrt(252)))
    for lag in (0, 2):
        record("Execution lag", f"{lag} days", run(lag=lag))
    for ticks in (0.0, 3.0, 5.0):
        record(
            "Slippage",
            f"{ticks:.0f} ticks per side",
            run(costs=Costs(ticks, 0.01, cfg["execution"]["commission_per_bbl"])),
        )

    record("VAR signal", f"Frozen VAR({vcfg['lags']})", run(), "var_vol")
    for p in (1, 8):
        alt_rhat = varmodel.walk_forward(inp.panel, p, vcfg["start"], vcfg["window"])
        record("VAR signal", f"VAR({p})", run(rhat=alt_rhat), "var_vol")
    alt_rhat = varmodel.walk_forward(inp.panel, vcfg["lags"], vcfg["start"], 520)
    record("VAR signal", "Rolling 10-year window", run(rhat=alt_rhat), "var_vol")
    for drop in ("inv", "slope"):
        alt_rhat = varmodel.walk_forward(inp.panel.drop(columns=drop), vcfg["lags"], vcfg["start"], vcfg["window"])
        record("VAR signal", f"Without {drop}", run(rhat=alt_rhat), "var_vol")
    return pd.DataFrame(rows)


def multiple_testing(bts: dict, summary: dict, robust: pd.DataFrame, cfg: dict) -> dict:
    """Deflated Sharpe for the selection on validation, Holm-adjusted p-values out of sample."""
    candidates = [n for n in NAMES if n != "benchmark"]
    trials = np.concatenate(
        [
            summary["validation"].loc[candidates, "sharpe"].to_numpy(),
            robust.loc[robust["group"] != "Baseline", "sharpe_val"].to_numpy(),
        ]
    )
    val = period(bts[MAIN], cfg["periods"]["validation"])["excess"]
    pvals = {n: 1.0 - metrics.probabilistic_sharpe(period(bts[n], cfg["periods"]["oos"])["excess"]) for n in candidates}
    order = sorted(pvals, key=pvals.get)
    holm, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[name]))
        holm[name] = running
    oos_main = period(bts[MAIN], cfg["periods"]["oos"])["excess"]
    oos_bench = period(bts["benchmark"], cfg["periods"]["oos"])["excess"]
    return {
        "dsr_validation": metrics.deflated_sharpe(val, trials),
        "pvalues_oos": pvals,
        "holm_oos": holm,
        "bootstrap_oos": metrics.bootstrap_sharpe(oos_main, oos_bench),
    }


def subperiods(bts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spans = {"2010-2014": ("2010", "2014"), "2015-2019": ("2015", "2019"), "2020-2024": ("2020", "2024")}
    return pd.DataFrame(
        {label: {n: metrics.sharpe(bt.loc[a:b, "excess"]) for n, bt in bts.items()} for label, (a, b) in spans.items()}
    )


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    inp = build_inputs(cfg)
    print(f"daily obs {len(inp.daily)}, weekly obs {len(inp.panel)}, rolls {len(inp.rolls)}")
    facts = pd.DataFrame(
        {
            "In-sample 1985-2009": stylized.facts(100 * inp.rolled["logret"].loc[: cfg["sample"]["in_sample_end"]]),
            "Out-of-sample 2010-2024": stylized.facts(100 * inp.rolled["logret"].loc[cfg["sample"]["oos_start"] :]),
        }
    )
    acfs = stylized.acf_table(100 * inp.rolled["logret"].loc[: cfg["sample"]["in_sample_end"]])
    vol = volatility(inp, cfg)
    print("volatility done")
    var_res = var_analysis(inp, cfg)
    print("VAR done")
    bts = run_strategies(inp, cfg, var_res["forecast"], vol["sigma_ann"])
    summary = summarise(bts, cfg)
    risk = value_at_risk(inp, cfg, vol, {"benchmark": bts["benchmark"]["pos"], MAIN: bts[MAIN]["pos"]})
    robust = robustness(inp, cfg, vol, var_res)
    print("robustness done")
    results = {
        "cfg": cfg,
        "inputs": vars(inp),
        "facts": facts,
        "acf": acfs,
        "vol": vol,
        "var": var_res,
        "backtests": bts,
        "summary": summary,
        "risk": risk,
        "stress": stress(bts),
        "subperiods": subperiods(bts),
        "robustness": robust,
        "multiple_testing": multiple_testing(bts, summary, robust, cfg),
    }
    with open(OUTPUT / "results.pkl", "wb") as handle:
        pickle.dump(results, handle)
    cols = ["sharpe", "ann_vol", "max_dd", "cagr"]
    headline = {name: s.loc[list(NAMES), cols].round(3).to_dict() for name, s in summary.items()}
    (OUTPUT / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(json.dumps(headline["validation"], indent=1))


if __name__ == "__main__":
    main()

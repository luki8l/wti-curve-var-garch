"""Tables, figures and number macros for the LaTeX report and factsheet.

    python -m wti.report      # reads output/results.pkl written by wti.analysis

Every number quoted in the report text is defined in reports/tables/numbers.tex,
so the prose cannot drift from the code output.
"""

from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wti import metrics  # noqa: E402
from wti.analysis import STRESS_WINDOWS, period  # noqa: E402
from wti.paths import FIGURES, OUTPUT, TABLES, ensure_dirs  # noqa: E402
from wti.strategies import MAIN, NAMES  # noqa: E402

# categorical slots validated for CVD separation (light surface); benchmark in neutral grey
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, GRID, SHADE = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
COLORS = {"benchmark": GREY, "long_vol": AQUA, "var_vol": ORANGE, MAIN: BLUE, "carry": YELLOW, "var": ORANGE}
FULL = 6.69  # text width in inches (A4, 2 cm margins)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7,
        "axes.labelcolor": INK2,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# ----------------------------------------------------------------------------- formatting
def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "\\textminus ")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%".replace("-", "\\textminus ")


def pval(x: float) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def write(name: str, body: str) -> None:
    (TABLES / name).write_text(body.strip() + "\n", encoding="utf-8")


def tabular(columns: str, header: list[str], rows: list[list[str]], midrules: tuple[int, ...] = ()) -> str:
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, row in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- tables
def table_facts(R: dict) -> None:
    f = R["facts"]
    is_, oos = f.columns
    rows = [
        (
            "Absence of autocorrelation",
            "$\\rho_1(r)$; Ljung--Box(10) $p$",
            lambda c: f"{num(f.at['acf1_r', c], 3)}; {pval(f.at['ljung_box10_p', c])}",
            "constant mean",
        ),
        (
            "Heavy tails",
            "excess kurtosis; Jarque--Bera $p$",
            lambda c: f"{num(f.at['kurt_daily', c], 1)}; {pval(f.at['jarque_bera_p', c])}",
            "Student-$t$ errors",
        ),
        ("Gain/loss asymmetry", "skewness", lambda c: num(f.at["skew", c]), "tail risk in ES"),
        (
            "Aggregational Gaussianity",
            "excess kurtosis, weekly / monthly",
            lambda c: f"{num(f.at['kurt_weekly', c], 1)} / {num(f.at['kurt_monthly', c], 1)}",
            "weekly VAR",
        ),
        (
            "Volatility clustering",
            "$\\rho_1(|r|)$; ARCH-LM(10) $p$",
            lambda c: f"{num(f.at['acf1_abs', c])}; {pval(f.at['arch_lm10_p', c])}",
            "GARCH sizing",
        ),
        (
            "Slow decay of $|r|$ autocorrelation",
            "$\\rho_{50}(|r|)$ / $\\rho_{100}(|r|)$",
            lambda c: f"{num(f.at['acf50_abs', c])} / {num(f.at['acf100_abs', c])}",
            "persistence near 1",
        ),
        (
            "Conditional heavy tails",
            "excess kurtosis of GARCH-N residuals",
            lambda c: num(f.at["kurt_std_resid", c], 1),
            "$t$ beyond GARCH",
        ),
        ("Leverage effect", "corr$(r_t, |r_{t+1}|)$", lambda c: num(f.at["leverage_corr", c], 3), "asymmetric term"),
    ]
    body = tabular(
        "@{}llllll@{}",
        ["Stylized fact (Cont 2001)", "Statistic", "1985--2009", "2010--2024", "Model consequence"],
        [[a, b, g(is_), g(oos), d] for a, b, g, d in rows],
    )
    write("tab_facts.tex", body.replace("{@{}llllll@{}}", "{@{}llrrl@{}}"))


def table_vol(R: dict) -> None:
    specs, ev = R["vol"]["specs"], R["vol"]["eval"]
    rows_a = []
    for name, row in specs.iterrows():
        gamma = "" if np.isnan(row["gamma"]) else f"{num(row['gamma'], 3)} ({num(row['gamma_t'], 1)})"
        nu = "" if np.isnan(row["nu"]) else num(row["nu"], 1)
        rows_a.append([name, f"{row['bic']:,.0f}".replace(",", "\\,"), num(row["persistence"], 3), gamma, nu])
    body_a = tabular("@{}lrrrr@{}", ["Model", "BIC", "Persistence", "$\\gamma$ ($t$)", "$\\nu$"], rows_a)
    main = R["cfg"]["garch"]["spec"]
    rows_b = []
    for name in ev["oos"].index:
        cells = [name]
        for p in ("validation", "oos"):
            e = ev[p].loc[name]
            cells.append(num(e["qlike"], 3))
            cells.append("--" if name == main else f"{num(e['dm_t'], 1)} [{pval(e['dm_pvalue'])}]")
        rows_b.append(cells)
    body_b = tabular(
        "@{}lrrrr@{}",
        ["1-day forecast", "QLIKE", "DM $t$ [$p$]", "QLIKE", "DM $t$ [$p$]"],
        rows_b,
    )
    write("tab_vol_specs.tex", body_a)
    write("tab_vol_eval.tex", body_b)


def table_var(R: dict) -> None:
    V = R["var"]
    g, pt, reg, fe = V["granger_is"], V["passthrough"], V["regimes"], V["forecast_eval"]
    rows = [
        ["Granger: inventory $\\to r$ (HC3 Wald, $p$)", pval(g.at["inv", "pvalue"]), ""],
        ["Granger: slope $\\to r$", pval(g.at["slope", "pvalue"]), ""],
        ["Granger: all lags $\\to r$", pval(g.at["all", "pvalue"]), ""],
        ["Slope pass-through $\\sum_i b_i$ (s.e.)", f"{num(pt['sum'], 4)} ({num(pt['se'], 4)})", ""],
        ["\\quad $H_0$: 0 / $H_0$: 1/52 (full carry), $p$", f"{pval(pt['p_zero'])} / {pval(pt['p_null'])}", ""],
        [
            "Next-week return, backwardation (\\% p.a., $t$)",
            f"{num(reg.at['backwardation', 'ann'], 1)} ({num(reg.at['backwardation', 't'], 1)})",
            "",
        ],
        [
            "Next-week return, contango (\\% p.a., $t$)",
            f"{num(reg.at['contango', 'ann'], 1)} ({num(reg.at['contango', 't'], 1)})",
            "",
        ],
        ["Out-of-sample $R^2$ vs prevailing mean", pct(fe.at["validation", "oos_r2"]), pct(fe.at["oos", "oos_r2"])],
        ["Clark--West $p$ (one-sided)", pval(fe.at["validation", "cw_pvalue"]), pval(fe.at["oos", "cw_pvalue"])],
        [
            "Directional hit rate; Pesaran--Timmermann $p$",
            f"{pct(fe.at['validation', 'hit_rate'])}; {pval(fe.at['validation', 'pt_pvalue'])}",
            f"{pct(fe.at['oos', 'hit_rate'])}; {pval(fe.at['oos', 'pt_pvalue'])}",
        ],
    ]
    body = tabular(
        "@{}lrr@{}",
        [f"VAR({R['cfg']['var']['lags']}) evidence", "In-sample", "Out-of-sample"],
        rows,
        midrules=(7,),
    )
    write("tab_var.tex", body)


PERF_COLS = [
    ("sharpe", "Sharpe", lambda x: num(x)),
    ("ann_vol", "Vol", lambda x: pct(x, 0)),
    ("cagr", "CAGR", lambda x: pct(x)),
    ("max_dd", "MaxDD", lambda x: pct(x, 0)),
    ("alpha_ann", "$\\alpha$ p.a.", lambda x: "" if np.isnan(x) else pct(x)),
    ("alpha_t", "$t(\\alpha)$", lambda x: "" if np.isnan(x) else num(x, 1)),
    ("beta", "$\\beta$", lambda x: "" if np.isnan(x) else num(x)),
    ("turnover", "Turnover", lambda x: f"{x:.0f}$\\times$"),
]


def table_performance(R: dict) -> None:
    rows, rules = [], []
    for p, label in (("validation", "Validation 1996--2009"), ("oos", "Out-of-sample 2010--2024")):
        rules.append(len(rows))
        rows.append([f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{label}}}}}"])
        s = R["summary"][p]
        for name in NAMES:
            cells = [NAMES[name] if name != MAIN else f"\\textbf{{{NAMES[name]}}}"]
            cells += [fmt(s.at[name, key]) for key, _, fmt in PERF_COLS]
            rows.append(cells)
    body = tabular("@{}lrrrrrrrr@{}", ["Strategy"] + [h for _, h, _ in PERF_COLS], rows, midrules=(rules[1],))
    write("tab_performance.tex", body)


def table_risk(R: dict) -> None:
    risk = R["risk"]
    rows = []
    for name in ("benchmark", MAIN):
        for alpha in (0.01, 0.05):
            r = risk[name].loc[("oos", alpha)]
            rows.append(
                [
                    NAMES[name].split(" (")[0],
                    f"{alpha:.0%}".replace("%", "\\%"),
                    f"{int(r['hits'])} / {int(r['n'])}",
                    pct(r["hit_rate"], 2),
                    pval(r["kupiec_pvalue"]),
                    pval(r["christoffersen_pvalue"]),
                ]
            )
    write(
        "tab_var_backtest.tex",
        tabular(
            "@{}llrrrr@{}",
            ["Position", "$\\alpha$", "Hits / days", "Hit rate", "Kupiec $p$", "Christoffersen $p$"],
            rows,
        ),
    )
    st = R["stress"]
    short = {
        "Asian crisis": "Asian crisis 1997--98",
        "GFC collapse": "GFC, 2008 H2",
        "OPEC price war": "OPEC price war 2014--16",
        "COVID, negative front month": "COVID, Jan--Apr 2020",
        "Russia-Ukraine spike": "Russia--Ukraine 2022",
    }
    rows = []
    for label in STRESS_WINDOWS:
        r = st.loc[label]
        rows.append(
            [
                short[label],
                pct(r["benchmark_ret"], 0),
                pct(r["long_vol_ret"], 0),
                pct(r[f"{MAIN}_ret"], 0),
                pct(r[f"{MAIN}_mdd"], 0),
            ]
        )
    write("tab_stress.tex", tabular("@{}lrrrr@{}", ["Episode", "Bench.", "Long v.t.", "Carry v.t.", "Carry DD"], rows))


def table_robustness(R: dict) -> None:
    rb = R["robustness"]
    rows, rules, last = [], [], None
    for i, r in rb.iterrows():
        if r["group"] != last and i > 0:
            rules.append(len(rows))
        last = r["group"]
        variant = r["variant"].replace("%", "\\%").replace("|slope| <", "$|s| <$")
        rows.append(
            [
                r["group"] if (i == 0 or rb.at[i - 1, "group"] != r["group"]) else "",
                variant,
                num(r["sharpe_val"]),
                num(r["sharpe_oos"]),
                pct(r["maxdd_oos"], 0),
                f"{r['turnover_oos']:.0f}",
            ]
        )
    write(
        "tab_robustness.tex",
        tabular(
            "@{}llrrrr@{}",
            ["Dimension", "Variant", "Sharpe val.", "Sharpe OOS", "MaxDD OOS", "Turnover OOS"],
            rows,
            midrules=tuple(rules),
        ),
    )


def table_params(R: dict) -> None:
    c = R["cfg"]
    rows = [
        ["Instrument", "NYMEX WTI futures (CL), 2nd nearby, rolled on front-month expiry"],
        [
            "Signal",
            f"sign of annualised log slope ln(C{c['futures']['slope_near']}/C{c['futures']['slope_far']}),"
            " +1 backwardation, $-$1 contango",
        ],
        [
            "Sizing",
            f"target volatility {pct(c['sizing']['target_vol'], 0)}, EGARCH(1,1)-$t$ 1-day forecast,"
            f" cap {c['sizing']['max_leverage']:.0f}$\\times$ capital",
        ],
        ["Rebalancing", "weekly, last trading day of the week; executed at next settlement"],
        [
            "Costs",
            f"{c['execution']['slippage_ticks']:.0f} tick (0.01 USD/bbl) slippage +"
            f" {c['execution']['commission_per_bbl']:.4f} USD/bbl fees per side; rolls charged both legs",
        ],
        [
            "Volatility model",
            f"{c['garch']['spec']}, expanding window, refit every {c['garch']['refit_every']} trading days",
        ],
        ["VAR", f"VAR({c['var']['lags']}) in (inventory, slope, return), expanding window, refit weekly"],
        [
            "Inventory input",
            f"weekly log change minus {c['inventory']['seasonal_years']}-year same-week mean;"
            f" usable {c['inventory']['release_lag_days']} days after week end",
        ],
        ["Samples", "design 1985--2009 (validation 1996--2009); out-of-sample 2010-01--2024-04"],
    ]
    write("tab_params.tex", tabular("@{}p{0.2\\textwidth}p{0.76\\textwidth}@{}", ["Item", "Setting"], rows))


def table_data(R: dict) -> None:
    man = R["inputs"]["manifest"]
    rows = []
    for name, entry in man.items():
        escaped = name.replace("_", "\\_")
        rows.append(
            [f"\\texttt{{{escaped}}}", entry["description"].replace("%", "\\%"), f"\\texttt{{{entry['sha256'][:12]}}}"]
        )
    write(
        "tab_data.tex",
        tabular("@{}p{0.27\\textwidth}p{0.5\\textwidth}l@{}", ["File", "Content", "SHA-256 (first 12)"], rows),
    )


def table_factsheet(R: dict) -> None:
    s = R["summary"]["oos"]
    m, b = s.loc[MAIN], s.loc["benchmark"]
    rows = [
        ["CAGR (total return)", pct(m["cagr"]), pct(b["cagr"])],
        ["Volatility p.a.", pct(m["ann_vol"]), pct(b["ann_vol"])],
        ["Sharpe ratio", num(m["sharpe"]), num(b["sharpe"])],
        ["Maximum drawdown", pct(m["max_dd"]), pct(b["max_dd"])],
        ["Calmar ratio", num(m["calmar"]), num(b["calmar"]) if b["cagr"] > 0 else "neg."],
        [
            "Skewness / excess kurtosis (daily)",
            f"{num(m['skew'])} / {num(m['kurt'], 1)}",
            f"{num(b['skew'])} / {num(b['kurt'], 1)}",
        ],
        ["Weekly hit rate", pct(m["hit_rate_w"]), pct(b["hit_rate_w"])],
        [
            "Avg. weekly win / loss",
            f"{pct(m['avg_win_w'])} / {pct(m['avg_loss_w'])}",
            f"{pct(b['avg_win_w'])} / {pct(b['avg_loss_w'])}",
        ],
        ["Beta / alpha p.a. vs benchmark", f"{num(m['beta'])} / {pct(m['alpha_ann'])}", "1.00 / --"],
        [
            "1-day VaR / ES 99\\% (historical)",
            f"{pct(m['var99'])} / {pct(m['es99'])}",
            f"{pct(b['var99'])} / {pct(b['es99'])}",
        ],
        ["Time long / net exposure", f"{pct(m['time_long'], 0)} / {num(m['net_exposure'])}", "100\\% / 1.00"],
        ["Turnover p.a. (incl. rolls)", f"{m['turnover']:.0f}$\\times$", f"{b['turnover']:.0f}$\\times$"],
        ["Trading-cost drag p.a.", pct(m["cost_drag"], 2), pct(b["cost_drag"], 2)],
    ]
    write("fs_metrics.tex", tabular("@{}lrr@{}", ["Out-of-sample 01/2010--04/2024", "Strategy", "Benchmark"], rows))


# ----------------------------------------------------------------------------- figures
def end_label(ax, series: pd.Series, text: str, color: str, dy: float = 0.0) -> None:
    ax.annotate(
        text,
        (series.index[-1], series.iloc[-1]),
        xytext=(4, dy),
        textcoords="offset points",
        color=INK2,
        fontsize=6.5,
        va="center",
    )
    ax.plot(series.index[-1], series.iloc[-1], "o", ms=2.5, color=color)


def fig_performance(R: dict) -> None:
    bts, cfg = R["backtests"], R["cfg"]
    start = pd.Timestamp(cfg["periods"]["validation"][0])
    oos0, oos1 = (pd.Timestamp(x) for x in cfg["periods"]["oos"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL, 2.7), sharex=True, gridspec_kw={"height_ratios": [2.6, 1]})
    for name in ["benchmark", "long_vol", "var_vol", MAIN]:
        w = metrics.wealth(bts[name].loc[start:, "total"])
        ax.plot(
            w.index,
            w,
            color=COLORS[name],
            lw=1.7 if name == MAIN else 1.0,
            zorder=3 if name == MAIN else 2,
            label=NAMES[name],
        )
    ax.set_yscale("log")
    ticks = [2.0**k for k in range(-5, 6)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"1/{int(1 / t)}" if t < 1 else f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.02, 90)
    ax.set_ylabel("Growth of 1, total return (log)")
    for a in (ax, ax2):
        a.axvspan(oos0, oos1, color=SHADE, zorder=0, lw=0)
    ax.text(start + pd.Timedelta(days=60), 55, "Validation 1996\u20132009", color=INK2, fontsize=6.5, va="center")
    ax.text(
        oos0 + pd.Timedelta(days=60),
        55,
        "Out-of-sample 2010\u20132024, frozen configuration",
        color=INK2,
        fontsize=6.5,
        va="center",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4, handlelength=1.8, columnspacing=1.4)
    for name in ("benchmark", MAIN):
        dd = metrics.drawdown(bts[name].loc[start:, "total"])
        ax2.fill_between(dd.index, dd, 0, color=COLORS[name], alpha=0.35 if name == MAIN else 0.25, lw=0)
        ax2.plot(dd.index, dd, color=COLORS[name], lw=0.7)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.set_ylim(-1, 0.02)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(start, oos1 + pd.Timedelta(days=30))
    fig.savefig(FIGURES / "fig_performance.pdf")
    plt.close(fig)


def fig_evidence(R: dict) -> None:
    """Stylized facts, volatility forecast and VAR impulse responses."""
    acf, vol, V = R["acf"], R["vol"], R["var"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.2))
    ax = axes[0, 0]
    lags = acf.index[1:]
    ax.bar(lags, acf["r"].iloc[1:], color=GREY, width=0.8, label="$r_t$")
    ax.plot(lags, acf["abs_r"].iloc[1:], color=BLUE, lw=1.1, label="$|r_t|$")
    ax.axhline(0, color=INK2, lw=0.5)
    n = R["inputs"]["rolled"].loc[: R["cfg"]["sample"]["in_sample_end"]].shape[0]
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=SHADE, lw=0)
    ax.set_title("(a) ACF of $r$ and $|r|$, 1985\u20132009")
    ax.set_xlabel("Lag (days)")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    sigma = vol["sigma_ann"].loc["2007-05-10":]
    ovx = R["inputs"]["ovx"]
    weekly = (
        pd.concat([sigma.rename("garch"), (ovx / 100).rename("ovx")], axis=1, join="inner").resample("W-FRI").last()
    )
    ax.plot(weekly.index, weekly["ovx"], color=GREY, lw=0.8, label="OVX (implied)")
    ax.plot(weekly.index, weekly["garch"], color=BLUE, lw=0.9, label=f"{R['cfg']['garch']['spec']} forecast")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("(b) GARCH vs implied volatility")
    ax.legend(loc="upper left", handlelength=1.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, 1.6)

    ax = axes[0, 2]
    pos = R["backtests"][MAIN]["pos"].loc[R["cfg"]["periods"]["validation"][0] :]
    weekly_pos = pos.resample("W-FRI").last()
    ax.fill_between(
        weekly_pos.index,
        weekly_pos,
        0,
        where=weekly_pos > 0,
        color=BLUE,
        alpha=0.55,
        lw=0,
        label="long (backwardation)",
    )
    ax.fill_between(
        weekly_pos.index, weekly_pos, 0, where=weekly_pos < 0, color=ORANGE, alpha=0.55, lw=0, label="short (contango)"
    )
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_title("(c) Exposure, multiple of capital")
    ax.legend(loc="upper left", ncols=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)
    ax.xaxis.set_major_locator(mdates.YearLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-2.3, 3.6)
    ax.set_yticks([-2, -1, 0, 1, 2])

    irf, boot = V["irf"]["cholesky"], V["irf_boot"]["cholesky"]
    h = np.arange(irf.shape[0])
    panels = [
        (axes[1, 0], 1, 0, False, "(d) Slope after an inventory shock", "pp p.a."),
        (axes[1, 1], 2, 0, True, "(e) Cum. return after an inventory shock", "%"),
        (axes[1, 2], 2, 1, True, "(f) Cum. return after a slope shock", "%"),
    ]
    for ax, i, j, cum, title, unit in panels:
        point, draws = irf[:, i, j], boot[:, :, i, j]
        if cum:
            point, draws = np.cumsum(point), np.cumsum(draws, axis=1)
        lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
        ax.fill_between(h, lo, hi, color=BLUE, alpha=0.18, lw=0, label="90% wild bootstrap")
        ax.plot(h, point, color=BLUE, lw=1.2, marker="o", ms=2.2, label="point estimate")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("Weeks after a one-s.d. shock")
        ax.set_ylabel(unit)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIGURES / "fig_evidence.pdf")
    plt.close(fig)


ROBUST_LABEL = {
    "Target volatility": "Target volatility {}",
    "Leverage cap": "Leverage cap {}",
    "Execution lag": "Execution lag {}",
    "Slippage": "Slippage {}",
    "Rebalancing": "{} rebalancing",
}


def robust_label(group: str, variant: str) -> str:
    if variant.startswith("EWMA"):
        return "EWMA volatility (0.94)"
    return ROBUST_LABEL.get(group, "{}").format(variant)


def fig_girf(R: dict) -> None:
    V = R["var"]
    irf, boot = V["irf"]["generalized"], V["irf_boot"]["generalized"]
    h = np.arange(irf.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.75))
    panels = [
        (axes[0], 1, 0, False, "Slope after an inventory shock", "pp p.a."),
        (axes[1], 2, 0, True, "Cum. return after an inventory shock", "%"),
        (axes[2], 2, 1, True, "Cum. return after a slope shock", "%"),
    ]
    for ax, i, j, cum, title, unit in panels:
        point, draws = irf[:, i, j], boot[:, :, i, j]
        if cum:
            point, draws = np.cumsum(point), np.cumsum(draws, axis=1)
        lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
        ax.fill_between(h, lo, hi, color=BLUE, alpha=0.18, lw=0)
        ax.plot(h, point, color=BLUE, lw=1.2, marker="o", ms=2.2)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("Weeks after a one-s.d. shock")
        ax.set_ylabel(unit)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "fig_girf.pdf")
    plt.close(fig)


def fig_robustness(R: dict) -> None:
    rb = R["robustness"]
    rb = rb[rb["strategy"] == MAIN].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    y = np.arange(len(rb))
    ax.scatter(rb["sharpe_val"], y, s=14, facecolors="white", edgecolors=BLUE, lw=0.9, label="Validation", zorder=3)
    ax.scatter(rb["sharpe_oos"], y, s=14, color=BLUE, label="Out-of-sample", zorder=3)
    for yi, (a, b) in enumerate(zip(rb["sharpe_val"], rb["sharpe_oos"], strict=True)):
        ax.plot([a, b], [yi, yi], color=GRID, lw=1.0, zorder=1)
    base = rb.loc[rb["group"] == "Baseline"].iloc[0]
    ax.axvline(base["sharpe_oos"], color=BLUE, lw=0.6, ls=":")
    bench = R["summary"]["oos"].at["benchmark", "sharpe"]
    ax.axvline(bench, color=GREY, lw=0.8)
    ax.text(bench + 0.015, -0.55, "benchmark, OOS", color=INK2, fontsize=6, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([robust_label(g, v) for g, v in zip(rb["group"], rb["variant"], strict=True)], fontsize=6.2)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe ratio")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.7, len(rb) - 0.1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=2, handletextpad=0.3)
    fig.savefig(FIGURES / "fig_robustness.pdf")
    plt.close(fig)


def fig_inputs(R: dict) -> None:
    panel = R["inputs"]["panel"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.8))
    ax = axes[0]
    s = panel["slope"]
    ax.fill_between(s.index, s, 0, where=s > 0, color=BLUE, alpha=0.6, lw=0, label="backwardation")
    ax.fill_between(s.index, s, 0, where=s <= 0, color=ORANGE, alpha=0.6, lw=0, label="contango")
    ax.axvline(pd.Timestamp("2010-01-01"), color=INK2, lw=0.6, ls=":")
    ax.set_title("(a) Roll yield ln(C2/C4) × 6, % p.a., weekly")
    ax.set_ylim(-60, 60)
    ax.legend(loc="lower left", ncols=2)
    ax = axes[1]
    inv = panel["inv"].rolling(13).mean()
    ax.plot(inv.index, inv, color=BLUE, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axvline(pd.Timestamp("2010-01-01"), color=INK2, lw=0.6, ls=":")
    ax.set_title("(b) Abnormal inventory change, %, 13-week mean")
    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator(6))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES / "fig_inputs.pdf")
    plt.close(fig)


def fig_rolling(R: dict) -> None:
    bts = R["backtests"]
    fig, ax = plt.subplots(figsize=(FULL, 1.7))
    for name in ("benchmark", MAIN):
        x = bts[name]["excess"].loc["1996":]
        roll = x.rolling(252).mean() / x.rolling(252).std() * np.sqrt(252)
        ax.plot(roll.index, roll, color=COLORS[name], lw=1.3 if name == MAIN else 0.9, label=NAMES[name])
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2024-04-05"), color=SHADE, zorder=0, lw=0)
    ax.set_title("Rolling 1-year Sharpe ratio")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(FIGURES / "fig_rolling_sharpe.pdf")
    plt.close(fig)


def fig_factsheet(R: dict) -> None:
    bts, cfg = R["backtests"], R["cfg"]
    oos = cfg["periods"]["oos"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.3, 2.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    for name in ("benchmark", MAIN):
        x = period(bts[name], oos)["total"]
        w = 100 * metrics.wealth(x)
        ax.plot(w.index, w, color=COLORS[name], lw=1.6 if name == MAIN else 1.0, label=NAMES[name])
        end_label(ax, w, f"{w.iloc[-1]:.0f}", COLORS[name])
        dd = metrics.drawdown(x)
        ax2.fill_between(dd.index, dd, 0, color=COLORS[name], alpha=0.35 if name == MAIN else 0.25, lw=0)
        ax2.plot(dd.index, dd, color=COLORS[name], lw=0.6)
    ax.axhline(100, color=INK2, lw=0.5)
    ax.set_ylabel("Index, 01/2010 = 100")
    ax.legend(loc="upper left")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp(oos[0]), pd.Timestamp("2025-03-01"))
    fig.savefig(FIGURES / "fs_cumulative.pdf")
    plt.close(fig)

    years = range(2010, 2025)
    annual = {
        n: [(1 + period(bts[n], oos)["total"].loc[str(y)]).prod() - 1 for y in years] for n in ("benchmark", MAIN)
    }
    fig, ax = plt.subplots(figsize=(4.3, 1.25))
    x = np.arange(len(years))
    ax.bar(x - 0.19, annual["benchmark"], width=0.36, color=GREY, label="Benchmark")
    ax.bar(x + 0.19, annual[MAIN], width=0.36, color=BLUE, label="Strategy")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}" if y != 2024 else "2024*" for y in years], fontsize=5.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower left", ncols=1, handlelength=1.0, borderaxespad=0.2)
    ax.set_ylim(-0.75, 1.0)
    fig.savefig(FIGURES / "fs_annual.pdf")
    plt.close(fig)


def fig_heatmap(R: dict) -> None:
    """Monthly total returns of the strategy, out-of-sample, as a diverging heatmap."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    x = period(R["backtests"][MAIN], R["cfg"]["periods"]["oos"])["total"]
    monthly = (1 + x).groupby([x.index.year, x.index.month]).prod() - 1
    table = monthly.unstack()
    yearly = (1 + x).groupby(x.index.year).prod() - 1
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25)
    fig, ax = plt.subplots(figsize=(4.3, 1.95))
    ax.imshow(table.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy(dtype=float)):
        if np.isfinite(v):
            label = f"{100 * v:.0f}".replace("-0", "0") if abs(100 * v) < 0.5 else f"{100 * v:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=5.2, color="white" if abs(v) > 0.15 else INK)
    for i, y in enumerate(table.index):
        ax.text(
            12.15,
            i,
            f"{100 * yearly.loc[y]:.0f}%",
            ha="left",
            va="center",
            fontsize=5.4,
            color=INK,
            fontweight="semibold",
        )
    ax.set_xticks(range(12))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=6)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=5.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(12.15, -1.0, "Year", ha="left", va="center", fontsize=5.6, color=INK2)
    ax.set_xlim(-0.5, 13.3)
    fig.savefig(FIGURES / "fs_heatmap.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- numbers
def numbers(R: dict) -> None:
    S, cfg = R["summary"], R["cfg"]
    V, vol, mt, inp = R["var"], R["vol"], R["multiple_testing"], R["inputs"]
    diag = inp["roll_diag"]
    informative = diag[diag["info"] > 0.1]
    irf, boot = V["irf"]["cholesky"], V["irf_boot"]["cholesky"]
    drift = np.cumsum(boot[:, :, 2, 1], axis=1)[:, -1] - boot[:, 0, 2, 1]
    order = V["order"]
    rb = R["robustness"]
    carry_rows = rb[(rb["strategy"] == MAIN) & (rb["group"] != "Baseline")]
    boot_oos = mt["bootstrap_oos"]
    sub = R["subperiods"]
    specs = vol["specs"]
    main_spec = cfg["garch"]["spec"]
    risk_b, risk_m = R["risk"]["benchmark"], R["risk"][MAIN]
    st = R["stress"]
    n = {
        "NDaily": f"{len(inp['daily']):,}".replace(",", "{,}"),
        "NWeekly": f"{len(inp['panel']):,}".replace(",", "{,}"),
        "NRolls": str(len(inp["rolls"])),
        "RollAgree": pct((informative["score"] < 0).mean(), 0),
        "RollAgreeN": str(len(informative)),
        "RollBefore": pct((informative["score_day_before"] < 0).mean(), 0),
        "RollAfter": pct((informative["score_day_after"] < 0).mean(), 0),
        "KurtDaily": num(R["facts"].iloc[:, 0]["kurt_daily"], 1),
        "KurtMonthly": num(R["facts"].iloc[:, 0]["kurt_monthly"], 1),
        "KurtResid": num(R["facts"].iloc[:, 0]["kurt_std_resid"], 1),
        "AcfAbsHundred": num(R["facts"].iloc[:, 0]["acf100_abs"]),
        "GarchSpec": main_spec,
        "GarchPersist": num(specs.at[main_spec, "persistence"], 3),
        "GarchNu": num(specs.at[main_spec, "nu"], 1),
        "GarchGammaT": num(specs.at[main_spec, "gamma_t"], 1),
        "GarchBicGain": f"{specs.at['GJR-t', 'bic'] - specs.at[main_spec, 'bic']:.0f}",
        "DmEwmaP": pval(vol["eval"]["oos"].at["EWMA", "dm_pvalue"]),
        "DmRollP": pval(vol["eval"]["oos"].at["Rolling 63d", "dm_pvalue"]),
        "DmNormalP": pval(vol["eval"]["oos"].at["GARCH-N", "dm_pvalue"]),
        "OvxCorr": num(vol["ovx_corr"]),
        "VarLags": str(cfg["var"]["lags"]),
        "BicLags": str(int(order["bic"].idxmin())),
        "AicLags": str(int(order["aic"].idxmin())),
        "LbSlopeOne": pval(V["order_fits"].at[1, "lb12_p_slope"]),
        "LbMinFour": num(V["order_fits"].loc[4, ["lb12_p_inv", "lb12_p_slope", "lb12_p_r"]].min()),
        "GrangerAllP": pval(V["granger_is"].at["all", "pvalue"]),
        "GrangerSlopeP": pval(V["granger_is"].at["slope", "pvalue"]),
        "PassSum": num(V["passthrough"]["sum"], 4),
        "PassZeroP": pval(V["passthrough"]["p_zero"]),
        "PassFullP": pval(V["passthrough"]["p_null"]),
        "BackAnn": num(V["regimes"].at["backwardation", "ann"], 1),
        "BackT": num(V["regimes"].at["backwardation", "t"], 1),
        "ContAnn": num(V["regimes"].at["contango", "ann"], 1),
        "ContT": num(V["regimes"].at["contango", "t"], 1),
        "IrfSlopeInvImpact": num(irf[0, 1, 0]),
        "IrfSlopeInvTwelve": num(irf[-1, 1, 0]),
        "IrfRetInvImpact": num(irf[0, 2, 0]),
        "IrfRetInvTwelve": num(np.cumsum(irf[:, 2, 0])[-1]),
        "IrfRetSlopeImpact": num(irf[0, 2, 1]),
        "IrfRetSlopeTwelve": num(np.cumsum(irf[:, 2, 1])[-1]),
        "IrfDrift": num(np.cumsum(irf[:, 2, 1])[-1] - irf[0, 2, 1]),
        "IrfDriftLo": num(np.quantile(drift, 0.05)),
        "IrfDriftHi": num(np.quantile(drift, 0.95)),
        "SlopeSd": num(np.sqrt(V["sigma"][1, 1]), 1),
        "WeeklyVolIs": num(inp["panel"].loc[: cfg["sample"]["in_sample_end"], "r"].std(), 1),
        "CarryWeekIs": num(inp["panel"].loc[: cfg["sample"]["in_sample_end"], "slope"].abs().mean() / 52, 2),
        "BenchVolIs": pct(inp["rolled"].loc[: cfg["sample"]["in_sample_end"], "ret"].std() * np.sqrt(252), 1),
        "OosRtwoVal": pct(V["forecast_eval"].at["validation", "oos_r2"]),
        "OosRtwoOos": pct(V["forecast_eval"].at["oos", "oos_r2"]),
        "CwPOos": pval(V["forecast_eval"].at["oos", "cw_pvalue"]),
        "VarShareLong": pct(V["forecast_eval"].at["oos", "share_long"], 0),
        "NTrials": str(mt["dsr_validation"]["n_trials"]),
        "DsrHurdle": num(mt["dsr_validation"]["hurdle_sharpe_ann"]),
        "Dsr": pct(mt["dsr_validation"]["dsr"], 0),
        "TargetVol": pct(cfg["sizing"]["target_vol"], 0),
        "MaxLev": f"{cfg['sizing']['max_leverage']:.0f}",
        "FreezeCommit": "9fb1e8d",
        "HolmMain": pval(mt["holm_oos"][MAIN]),
        "PsrMainP": pval(mt["pvalues_oos"][MAIN]),
        "BootLo": num(boot_oos["sharpe_ci90"][0]),
        "BootHi": num(boot_oos["sharpe_ci90"][1]),
        "BootP": pval(boot_oos["p_sharpe_le0"]),
        "BootDiff": num(boot_oos["diff"]),
        "BootDiffLo": num(boot_oos["diff_ci90"][0]),
        "BootDiffHi": num(boot_oos["diff_ci90"][1]),
        "BootDiffP": pval(boot_oos["p_diff_le0"]),
        "SubOne": num(sub.at[MAIN, "2010-2014"]),
        "SubTwo": num(sub.at[MAIN, "2015-2019"]),
        "SubThree": num(sub.at[MAIN, "2020-2024"]),
        "RobMin": num(carry_rows["sharpe_oos"].min()),
        "RobMax": num(carry_rows["sharpe_oos"].max()),
        "RobN": str(len(carry_rows)),
        "RobFiveTicks": num(rb.loc[rb["variant"] == "5 ticks per side", "sharpe_oos"].iloc[0]),
        "RobLongFlat": num(rb.loc[rb["variant"] == "Long/flat (no shorts)", "sharpe_oos"].iloc[0]),
        "RobLongFlatDd": pct(rb.loc[rb["variant"] == "Long/flat (no shorts)", "maxdd_oos"].iloc[0], 0),
        "RobMonthly": num(rb.loc[rb["variant"] == "Monthly", "sharpe_oos"].iloc[0]),
        "RobBand": num(rb.loc[rb["variant"].str.startswith("Neutral"), "sharpe_oos"].iloc[0]),
        "RobTwoThree": num(rb.loc[rb["variant"] == "Slope from C2/C3", "sharpe_oos"].iloc[0]),
        "VarHitBench": pct(risk_b.loc[("oos", 0.01), "hit_rate"], 2),
        "VarKupBench": pval(risk_b.loc[("oos", 0.01), "kupiec_pvalue"]),
        "VarHitMain": pct(risk_m.loc[("oos", 0.01), "hit_rate"], 2),
        "VarKupMain": pval(risk_m.loc[("oos", 0.01), "kupiec_pvalue"]),
        "VarHitsBench": f"{int(risk_b.loc[('oos', 0.01), 'hits'])}",
        "VarDaysBench": f"{int(risk_b.loc[('oos', 0.01), 'n']):,}".replace(",", "{,}"),
        "SkewOos": num(R["facts"].iloc[:, 1]["skew"], 1),
        "StressCovid": pct(st.at["COVID, negative front month", f"{MAIN}_ret"], 0),
        "StressCovidBench": pct(st.at["COVID, negative front month", "benchmark_ret"], 0),
        "StressOpec": pct(st.at["OPEC price war", f"{MAIN}_ret"], 0),
        "StressOpecBench": pct(st.at["OPEC price war", "benchmark_ret"], 0),
        "StressRussia": pct(st.at["Russia-Ukraine spike", f"{MAIN}_ret"], 0),
    }
    for p, tag in (("validation", "Val"), ("oos", "Oos")):
        s = S[p]
        for name, key in (
            ("benchmark", "Bench"),
            ("long_vol", "LongVol"),
            ("var", "Var"),
            ("var_vol", "VarVol"),
            ("carry", "Carry"),
            (MAIN, "Main"),
        ):
            n[f"{tag}Sharpe{key}"] = num(s.at[name, "sharpe"])
        m = s.loc[MAIN]
        n[f"{tag}CagrMain"] = pct(m["cagr"])
        n[f"{tag}VolMain"] = pct(m["ann_vol"], 0)
        n[f"{tag}MddMain"] = pct(m["max_dd"], 0)
        n[f"{tag}MddBench"] = pct(s.at["benchmark", "max_dd"], 0)
        n[f"{tag}CagrBench"] = pct(s.at["benchmark", "cagr"])
        n[f"{tag}AlphaMain"] = pct(m["alpha_ann"])
        n[f"{tag}AlphaT"] = num(m["alpha_t"], 1)
        n[f"{tag}BetaMain"] = num(m["beta"])
        n[f"{tag}TurnMain"] = f"{m['turnover']:.0f}"
        n[f"{tag}CostMain"] = pct(m["cost_drag"], 2)
        n[f"{tag}LongMain"] = pct(m["time_long"], 0)
        n[f"{tag}SharpeSe"] = num(m["sharpe_se"])
        n[f"{tag}EsMain"] = pct(m["es99"])
        n[f"{tag}EsBench"] = pct(s.at["benchmark", "es99"])
    lines = [f"\\newcommand{{\\N{k}}}{{{v}}}" for k, v in n.items()]
    write("numbers.tex", "% generated by wti.report - do not edit\n" + "\n".join(lines))


def main() -> None:
    ensure_dirs()
    with open(OUTPUT / "results.pkl", "rb") as handle:
        R = pickle.load(handle)
    for build in (
        table_facts,
        table_vol,
        table_var,
        table_performance,
        table_risk,
        table_robustness,
        table_params,
        table_data,
        table_factsheet,
        fig_performance,
        fig_evidence,
        fig_girf,
        fig_robustness,
        fig_inputs,
        fig_rolling,
        fig_factsheet,
        fig_heatmap,
        numbers,
    ):
        build(R)
    print(f"tables -> {TABLES}\nfigures -> {FIGURES}")


if __name__ == "__main__":
    main()

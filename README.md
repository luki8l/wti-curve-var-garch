# Earning the Roll

A volatility-targeted carry strategy for WTI crude oil futures.
Econometrics project, MSc Innovative Finance, University of Liechtenstein.

The strategy holds the second NYMEX WTI contract long when the futures curve is in
backwardation and short when it is in contango, and sizes the position each week to
35% annualised volatility with an EGARCH(1,1)-t forecast. A weekly VAR of inventories,
curve slope and returns (with impulse responses) motivates the signal; stylized facts
motivate the volatility model.

**Deliverables:** [`reports/report.pdf`](reports/report.pdf) (4 pages + appendix) and
[`reports/factsheet.pdf`](reports/factsheet.pdf) (1 page).

## Out-of-sample result (01/2010 to 04/2024, net of costs)

|                     | Strategy | Long WTI 1x |
|---------------------|---------:|------------:|
| Sharpe ratio        |     0.42 |        0.03 |
| CAGR (total return) |    10.1% |       −5.6% |
| Volatility          |      36% |         40% |
| Maximum drawdown    |     −54% |        −94% |

The configuration was chosen on 1985–2009 and frozen in commit `9fb1e8d` before the
out-of-sample results were inspected. The result is positive in every subperiod and
robustness variant but not statistically conclusive (90% bootstrap interval of the
Sharpe ratio 0.02 to 0.86). See the report for the full discussion.

## Method in brief

- **Data:** EIA NYMEX settlements for contracts 1–4 (daily, 1985 to 5 April 2024, when
  the EIA stopped publishing them), EIA weekly crude stocks, FRED T-bill and OVX.
- **Rolled series:** second contract, rolled at each front-month expiry using the CME
  last-trading-day rule; returns across curve labels on roll days.
- **Stylized facts → volatility model:** heavy tails, clustering and leverage effect lead
  to EGARCH(1,1) with Student-t errors (lowest BIC of five candidates).
- **VAR(4) + IRFs:** inventories, slope and returns; HC3 Granger tests, recursive-design
  wild-bootstrap bands, walk-forward forecast evaluation (OOS R², Clark–West).
- **Validation:** six candidate strategies compared on 1996–2009; deflated Sharpe ratio,
  bootstrap and Holm-adjusted inference out of sample; 15 one-at-a-time robustness variants
  of the strategy and 5 of the VAR signal.
- **Hygiene:** one-day execution lag, tick-based costs on trades and rolls, point-in-time
  inventory releases, expanding-window estimation, tests that shock future data and check
  that no earlier decision changes.

## Reproduce

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
python scripts/download_data.py   # only needed to refresh data/raw (standard library only)
python -m wti.analysis            # models and backtests, about one minute
python -m wti.report              # tables, figures, reports/tables/numbers.tex
pytest
cd reports && latexmk -pdf report.tex factsheet.tex
```

or `make all`. The raw files in `data/raw/` are the snapshot used for the report; their
SHA-256 checksums in `MANIFEST.json` are verified on every run. Every number quoted in the
report text is a macro in `reports/tables/numbers.tex`, written by `wti.report`.

## Layout

```
config/strategy.yaml     frozen parameters, with the reason for each choice
data/raw/                downloaded files and MANIFEST.json
scripts/download_data.py data download
src/wti/
  data.py                file readers, checksum verification
  futures.py             expiry calendar, roll validation, rolled returns, slope
  features.py            weekly decision dates, inventory input, release alignment
  stylized.py            stylized-facts statistics
  garch.py               GARCH family, walk-forward variance forecasts
  varmodel.py            VAR, Granger tests, impulse responses, wild bootstrap
  evaluation.py          forecast tests, VaR backtests
  backtest.py            execution and cost model
  strategies.py          candidate strategies
  metrics.py             performance statistics, Sharpe inference
  analysis.py            end-to-end run
  report.py              tables, figures, number macros
reports/                 LaTeX sources, generated tables and figures, PDFs
tests/                   unit tests, including look-ahead checks
```

## Team

Team Backwardation: Lukas Lobnig, Niklas Omann, Noemi Pirrello, Adrian Rupp

"""Readers for the raw EIA and FRED files and a manifest integrity check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from wti.paths import RAW


def read_eia_xls(path: Path) -> pd.Series:
    """Read an EIA 'hist_xls' file (sheet 'Data 1': header block, then Date | value)."""
    sheet = pd.read_excel(path, sheet_name="Data 1", header=None)
    header_row = sheet.index[sheet.iloc[:, 0].astype(str).str.strip().eq("Date")][0]
    body = sheet.iloc[header_row + 1 :, :2].dropna(how="any")
    series = pd.Series(
        pd.to_numeric(body.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.to_datetime(body.iloc[:, 0]),
        name=str(sheet.iloc[header_row, 1]),
    )
    series.index.name = "date"
    return series.dropna().sort_index()


def read_fred_csv(path: Path) -> pd.Series:
    """Read a FRED 'fredgraph.csv' download; missing values are coded as '.'."""
    frame = pd.read_csv(path, na_values=["."])
    series = pd.Series(
        pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame.iloc[:, 0]),
        name=frame.columns[1],
    )
    series.index.name = "date"
    return series.dropna().sort_index()


def verify_manifest(raw_dir: Path = RAW) -> dict:
    """Check every file listed in MANIFEST.json against its recorded SHA-256."""
    manifest = json.loads((raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        digest = hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{name}: checksum mismatch, raw snapshot was modified")
    return manifest


def load_curve(raw_dir: Path = RAW) -> pd.DataFrame:
    """NYMEX WTI settlement prices for contracts 1-4.

    The index is the exchange calendar as observed in the file. Some exchange
    holidays (mostly 2018-2019) carry the previous settlement forward; such rows,
    identical to the day before in all four contracts, are removed. Days on which
    single contracts are missing are kept with NaN, so that the index can serve
    as the business-day calendar for the expiry rule.
    """
    columns = {f"c{k}": read_eia_xls(raw_dir / f"eia_RCLC{k}d.xls") for k in range(1, 5)}
    curve = pd.DataFrame(columns)
    carried_forward = curve.eq(curve.shift()).all(axis=1)
    return curve[~carried_forward]


def load_stocks(raw_dir: Path = RAW) -> pd.Series:
    """US commercial crude oil stocks excl. SPR (kbbl), indexed by week-ending Friday."""
    return read_eia_xls(raw_dir / "eia_WCESTUS1w.xls").rename("stocks")


def load_tbill(raw_dir: Path = RAW) -> pd.Series:
    """3-month T-bill secondary market rate, % p.a. (discount basis)."""
    return read_fred_csv(raw_dir / "fred_DTB3.csv").rename("tbill")


def load_optional(name: str, raw_dir: Path = RAW) -> pd.Series | None:
    path = raw_dir / name
    if not path.exists():
        return None
    reader = read_fred_csv if path.suffix == ".csv" else read_eia_xls
    return reader(path)

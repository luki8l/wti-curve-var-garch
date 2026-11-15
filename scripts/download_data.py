"""Download the raw input data from EIA and FRED.

Run once from the repository root on a machine with internet access:

    python scripts/download_data.py

Only the Python standard library is used, so no environment setup is needed.
Files are stored unmodified in data/raw/ together with MANIFEST.json (source
URL, download time, size, SHA-256). Parsing happens later in the pipeline, so
a snapshot is never altered after download.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EIA = "https://www.eia.gov/dnav/pet/hist_xls/{}.xls"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# (file name, url, required, description)
SOURCES = [
    ("eia_RCLC1d.xls", EIA.format("RCLC1d"), True, "NYMEX WTI futures, contract 1, daily settle (USD/bbl)"),
    ("eia_RCLC2d.xls", EIA.format("RCLC2d"), True, "NYMEX WTI futures, contract 2, daily settle (USD/bbl)"),
    ("eia_RCLC3d.xls", EIA.format("RCLC3d"), True, "NYMEX WTI futures, contract 3, daily settle (USD/bbl)"),
    ("eia_RCLC4d.xls", EIA.format("RCLC4d"), True, "NYMEX WTI futures, contract 4, daily settle (USD/bbl)"),
    ("eia_WCESTUS1w.xls", EIA.format("WCESTUS1w"), True, "US crude oil ending stocks excl. SPR, weekly (kbbl)"),
    ("eia_cushing_stocks_w.xls", EIA.format("W_EPC0_SAX_YCUOK_MBBLw"), False, "Cushing crude oil ending stocks excl. SPR, weekly (kbbl)"),
    ("eia_RWTCd.xls", EIA.format("RWTCd"), False, "WTI spot price, Cushing, daily (USD/bbl)"),
    ("fred_DTB3.csv", FRED.format("DTB3"), True, "3-month T-bill, secondary market, daily (% p.a.)"),
    ("fred_OVXCLS.csv", FRED.format("OVXCLS"), False, "CBOE crude oil ETF volatility index, daily"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; wti-curve-var-garch)"}


def fetch(url: str, attempts: int = 3, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed after {attempts} attempts: {last_error}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    failed_required = []
    for name, url, required, description in SOURCES:
        try:
            payload = fetch(url)
        except RuntimeError as error:
            print(f"[{'FAIL' if required else 'skip'}] {name}: {error}")
            if required:
                failed_required.append(name)
            continue
        (RAW_DIR / name).write_bytes(payload)
        manifest[name] = {
            "url": url,
            "description": description,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        print(f"[ok]   {name} ({len(payload) / 1024:.0f} KB)")

    with open(RAW_DIR / "MANIFEST.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if failed_required:
        print(f"\nMissing required files: {', '.join(failed_required)}")
        return 1
    print(f"\nSaved {len(manifest)} files to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

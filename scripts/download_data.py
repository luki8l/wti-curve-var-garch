"""Download the raw input data from EIA and FRED.

Run once from the repository root on a machine with internet access:

    python scripts/download_data.py           # fetch files that are missing
    python scripts/download_data.py --force   # re-download everything

Only the Python standard library is used, so no environment setup is needed.
Files are stored unmodified in data/raw/ and listed in MANIFEST.json (source
URL, time stamp, size, SHA-256). Parsing happens later in the pipeline, so a
snapshot is never altered after download.

If a FRED request is refused, open the URL printed in the error message in a
browser, save the CSV under the file name shown, and run the script again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    (
        "eia_cushing_stocks_w.xls",
        EIA.format("W_EPC0_SAX_YCUOK_MBBLw"),
        False,
        "Cushing crude oil ending stocks excl. SPR, weekly (kbbl)",
    ),
    ("eia_RWTCd.xls", EIA.format("RWTCd"), False, "WTI spot price, Cushing, daily (USD/bbl)"),
    ("fred_DTB3.csv", FRED.format("DTB3"), True, "3-month T-bill, secondary market, daily (% p.a.)"),
    ("fred_OVXCLS.csv", FRED.format("OVXCLS"), False, "CBOE crude oil ETF volatility index, daily"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch(url: str, attempts: int = 3, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # network errors come in many types (URLError, RemoteDisconnected, ...)
            last_error = error
            time.sleep(3 * attempt)
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def record(path: Path, url: str, description: str, timestamp: datetime) -> dict:
    payload = path.read_bytes()
    return {
        "url": url,
        "description": description,
        "retrieved_utc": timestamp.isoformat(timespec="seconds"),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    missing_required = []

    for name, url, required, description in SOURCES:
        path = RAW_DIR / name
        if path.exists() and not args.force:
            if name not in manifest:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                manifest[name] = record(path, url, description, mtime)
            print(f"[keep] {name}")
            continue
        try:
            payload = fetch(url)
        except RuntimeError as error:
            print(f"[{'FAIL' if required else 'skip'}] {name}: {error}\n       manual download: {url}")
            if required:
                missing_required.append(name)
            continue
        path.write_bytes(payload)
        manifest[name] = record(path, url, description, datetime.now(timezone.utc))
        print(f"[ok]   {name} ({len(payload) / 1024:.0f} KB)")

    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n", encoding="utf-8")

    if missing_required:
        print(f"\nMissing required files: {', '.join(missing_required)}")
        return 1
    print(f"\n{len(manifest)} files in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

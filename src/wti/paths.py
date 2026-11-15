from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "strategy.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"


def ensure_dirs() -> None:
    for path in (PROCESSED, OUTPUT, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)

# scripts/04_add_sentiment.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fng.csv"

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    print(f"[INFO] Descargando Fear & Greed: {API_URL}")
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()

    data = payload.get("data", [])
    if not data:
        print("[ERROR] Respuesta vacía de FNG API")
        return 1

    rows = []
    for item in data:
        try:
            ts = int(item["timestamp"])  # seconds
            val = float(item["value"])
            cls = str(item.get("value_classification", ""))
        except Exception:
            continue

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rows.append(
            {
                "timestamp_s": ts,
                "datetime_utc": dt.isoformat(),
                "date_utc": dt.date().isoformat(),
                "fng_value": val,
                "fng_classification": cls,
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp_s").drop_duplicates("timestamp_s")
    df.to_csv(OUT_PATH, index=False)

    print("[INFO] Guardado:", OUT_PATH)
    print("[INFO] Rango:", df["date_utc"].min(), "->", df["date_utc"].max())
    print("[INFO] Filas:", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

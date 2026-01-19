"""
00_download.py
Descarga OHLCV (BTC/USDT 1H) desde un exchange (default Binance) usando ccxt.

- Lee config desde: config/base.yaml
- Permite override por .env / variables de entorno:
  EXCHANGE, SYMBOL, TIMEFRAME, START_DATE, END_DATE, LIMIT, RATE_LIMIT_MS

Salida:
- data/raw/{exchange}_{symbol}_{timeframe}_{start}_{end}.csv

Uso:
  python scripts/00_download.py
  python scripts/00_download.py --exchange binance --symbol BTC/USDT --timeframe 1h --start 2019-01-01 --end 2025-12-31
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import ccxt
import pandas as pd
import yaml
from dotenv import load_dotenv

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# -----------------------------
# Helpers
# -----------------------------
def utc_dt_from_ymd(s: str) -> datetime:
    # Expect YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def safe_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "-")


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else default


def parse_int(s: Optional[str], default: int) -> int:
    if s is None:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def parse_float(s: Optional[str], default: float) -> float:
    if s is None:
        return default
    try:
        return float(s)
    except ValueError:
        return default


# -----------------------------
# Settings
# -----------------------------
@dataclass
class DownloadSettings:
    exchange: str
    symbol: str
    timeframe: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD (inclusive-ish)
    limit: int = 1000
    rate_limit_ms: int = 200  # extra sleep between calls (Binance is ok with ~200-500ms)
    max_retries: int = 8


def resolve_settings(args: argparse.Namespace) -> DownloadSettings:
    cfg = load_yaml_config(CONFIG_PATH)

    # YAML defaults
    y_market = (cfg.get("market") or {})
    exchange = y_market.get("exchange", "binance")
    symbol = y_market.get("symbol", "BTC/USDT")
    timeframe = y_market.get("timeframe", "1h")
    start_date = y_market.get("start_date", "2019-01-01")
    end_date = (cfg.get("validation") or {}).get("test_end", None)  # fallback
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ENV overrides
    exchange = get_env("EXCHANGE", exchange)
    symbol = get_env("SYMBOL", symbol)
    timeframe = get_env("TIMEFRAME", timeframe)
    start_date = get_env("START_DATE", start_date)
    end_date = get_env("END_DATE", end_date)

    limit = parse_int(get_env("LIMIT", None), 1000)
    rate_limit_ms = parse_int(get_env("RATE_LIMIT_MS", None), 200)

    # CLI overrides (highest priority)
    if args.exchange: exchange = args.exchange
    if args.symbol: symbol = args.symbol
    if args.timeframe: timeframe = args.timeframe
    if args.start: start_date = args.start
    if args.end: end_date = args.end
    if args.limit is not None: limit = args.limit
    if args.rate_limit_ms is not None: rate_limit_ms = args.rate_limit_ms

    return DownloadSettings(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        rate_limit_ms=rate_limit_ms,
    )


# -----------------------------
# Exchange init
# -----------------------------
def init_exchange(exchange_id: str) -> ccxt.Exchange:
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange no soportado por ccxt o mal escrito: {exchange_id}")

    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })

    # Si el exchange necesita credentials para endpoints, podrías setear acá:
    # ex.apiKey = os.getenv("API_KEY")
    # ex.secret = os.getenv("API_SECRET")

    return ex


# -----------------------------
# Download core
# -----------------------------
def fetch_ohlcv_paged(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    rate_limit_ms: int,
    max_retries: int,
) -> pd.DataFrame:
    """
    Descarga OHLCV desde since_ms (incl) hasta until_ms (excl) paginando.
    Devuelve DataFrame con columnas: timestamp, open, high, low, close, volume
    """
    all_rows: List[List[float]] = []
    tf_ms = ex.parse_timeframe(timeframe) * 1000

    cur = since_ms
    n_calls = 0

    while cur < until_ms:
        # Reintentos con backoff
        attempt = 0
        while True:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
                n_calls += 1
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Fallo fetch_ohlcv tras {max_retries} reintentos. Ultimo error: {e}") from e
                sleep_s = min(60, (2 ** attempt) * 0.5)
                print(f"[WARN] Error fetch_ohlcv ({type(e).__name__}): {e}. Reintento {attempt}/{max_retries} en {sleep_s:.1f}s")
                time.sleep(sleep_s)

        if not rows:
            # Si no devuelve nada, avanzamos una vela para evitar loop infinito.
            cur += tf_ms
            time.sleep(rate_limit_ms / 1000)
            continue

        # Guardar y avanzar
        all_rows.extend(rows)

        last_ts = rows[-1][0]
        # Avanzar a la próxima vela para evitar duplicados en el borde
        next_cur = last_ts + tf_ms

        # Protección: si el exchange devolvió algo raro y no avanzamos, forzamos avance
        if next_cur <= cur:
            next_cur = cur + tf_ms

        cur = next_cur

        # Extra rate limit
        time.sleep(rate_limit_ms / 1000)

        # Pequeño feedback
        if n_calls % 10 == 0:
            last_dt = ms_to_dt(last_ts).strftime("%Y-%m-%d %H:%M")
            print(f"[INFO] Llamadas: {n_calls} | ultimo: {last_dt} UTC | filas acumuladas: {len(all_rows)}")

        # Corte duro por until
        if last_ts >= until_ms:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def clean_and_validate(df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, dict]:
    """
    - Dedup por timestamp
    - Orden
    - Detecta gaps
    """
    if df.empty:
        return df, {"rows": 0, "duplicates_removed": 0, "gaps": 0}

    df = df.copy()
    before = len(df)

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    after = len(df)

    # Expected delta
    # NOTE: no dependemos de exchange acá, usamos ccxt.parse_timeframe? pero no tenemos ex.
    # Para 1h es 3600s, para general: parse manual simple
    # (soporta: m,h,d,w)
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    seconds = {
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800
    }.get(unit)
    if seconds is None:
        raise ValueError(f"timeframe no soportado para validación simple: {timeframe}")
    expected_ms = n * seconds * 1000

    ts = df["timestamp"].astype("int64")
    diffs = ts.diff().dropna()
    gaps = int((diffs > expected_ms).sum())

    stats = {
        "rows": int(after),
        "duplicates_removed": int(before - after),
        "gaps": gaps,
        "start": ms_to_dt(int(df["timestamp"].iloc[0])).isoformat(),
        "end": ms_to_dt(int(df["timestamp"].iloc[-1])).isoformat(),
    }
    return df, stats


def build_output_path(exchange: str, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sym = safe_symbol_for_filename(symbol)
    fname = f"{exchange}_{sym}_{timeframe}_{start_date}_to_{end_date}.csv"
    return RAW_DIR / fname


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", type=str, default=None)
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--timeframe", type=str, default=None)
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=None, help="OHLCV limit por llamada (binance suele 1000)")
    p.add_argument("--rate-limit-ms", type=int, default=None, help="sleep extra entre llamadas")
    return p.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    args = parse_args()
    s = resolve_settings(args)

    # Parse fechas
    start_dt = utc_dt_from_ymd(s.start_date)
    end_dt = utc_dt_from_ymd(s.end_date)

    # until: ponemos end + 1 día para cubrir inclusive y luego recortamos
    until_dt = end_dt.replace(hour=23, minute=59, second=59)
    since_ms = dt_to_ms(start_dt)
    until_ms = dt_to_ms(until_dt)

    print("[INFO] Settings:")
    print(f"  exchange   : {s.exchange}")
    print(f"  symbol     : {s.symbol}")
    print(f"  timeframe  : {s.timeframe}")
    print(f"  start_date : {s.start_date}")
    print(f"  end_date   : {s.end_date}")
    print(f"  limit      : {s.limit}")
    print(f"  rate_limit : {s.rate_limit_ms} ms")

    ex = init_exchange(s.exchange)

    # Validar que el mercado exista y el timeframe sea soportado
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[ERROR] No pude load_markets(): {e}")
        return 1

    if s.symbol not in ex.markets:
        # algunos exchanges requieren símbolo sin slash o mayúsculas; informamos mercados similares
        print(f"[ERROR] Símbolo no encontrado en mercados del exchange: {s.symbol}")
        suggestions = [k for k in ex.markets.keys() if k.replace("/", "").upper() == s.symbol.replace("/", "").upper()]
        if suggestions:
            print(f"[HINT] Quizás quisiste: {suggestions[:10]}")
        return 1

    if not ex.has.get("fetchOHLCV", False):
        print("[ERROR] Exchange no soporta fetchOHLCV en ccxt")
        return 1

    # Descarga
    t0 = time.time()
    df = fetch_ohlcv_paged(
        ex=ex,
        symbol=s.symbol,
        timeframe=s.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=s.limit,
        rate_limit_ms=s.rate_limit_ms,
        max_retries=s.max_retries,
    )

    if df.empty:
        print("[ERROR] No se descargaron datos.")
        return 1

    # Recortar a end_date (por si vino un poco más)
    df = df[df["timestamp"] <= until_ms].copy()

    # Clean + validate
    df, stats = clean_and_validate(df, s.timeframe)

    # Agregar columna datetime legible
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Reordenar columnas
    df = df[["timestamp", "datetime_utc", "open", "high", "low", "close", "volume"]]

    out_path = build_output_path(s.exchange, s.symbol, s.timeframe, s.start_date, s.end_date)
    df.to_csv(out_path, index=False)

    dt_s = time.time() - t0
    print("[INFO] Descarga OK")
    print(f"  output: {out_path}")
    print(f"  rows  : {stats.get('rows')}")
    print(f"  dup_rm: {stats.get('duplicates_removed')}")
    print(f"  gaps  : {stats.get('gaps')}")
    print(f"  range : {stats.get('start')} -> {stats.get('end')}")
    print(f"  time  : {dt_s:.1f}s")

    # Si hay gaps, avisar fuerte (no fallamos porque a veces el exchange tiene huecos históricos)
    if stats.get("gaps", 0) > 0:
        print("[WARN] Se detectaron gaps. Siguiente paso: correr un validador más estricto y/o reintentar relleno.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
01_build_features.py
Construye features mínimos para el MVP:
- EMA fast / EMA slow
- ATR (Wilder RMA)
- Retornos (opcional) y algunas columnas auxiliares

Lee:
  data/raw/{exchange}_{symbol}_{timeframe}_{start}_to_{end}.csv

Escribe:
  data/processed/{exchange}_{symbol}_{timeframe}_{start}_to_{end}_features.csv

Config:
  config/base.yaml (strategy.ema_fast, strategy.ema_slow, strategy.atr_period)
y overrides por env: EXCHANGE, SYMBOL, TIMEFRAME, START_DATE, END_DATE
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import pandas as pd
import yaml
from dotenv import load_dotenv


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


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


@dataclass
class FeatureSettings:
    exchange: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    ema_fast: int
    ema_slow: int
    atr_period: int


def resolve_settings() -> FeatureSettings:
    cfg = load_yaml_config(CONFIG_PATH)

    # YAML
    market = (cfg.get("market") or {})
    strat = (cfg.get("strategy") or {})

    exchange = market.get("exchange", "binance")
    symbol = market.get("symbol", "BTC/USDT")
    timeframe = market.get("timeframe", "1h")
    start_date = market.get("start_date", "2019-01-01")

    # end_date: tomamos test_end si está, sino hoy
    end_date = (cfg.get("validation") or {}).get("test_end", None)
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ema_fast = int(strat.get("ema_fast", 20))
    ema_slow = int(strat.get("ema_slow", 50))
    atr_period = int(strat.get("atr_period", 14))

    # ENV overrides
    exchange = get_env("EXCHANGE", exchange)
    symbol = get_env("SYMBOL", symbol)
    timeframe = get_env("TIMEFRAME", timeframe)
    start_date = get_env("START_DATE", start_date)
    end_date = get_env("END_DATE", end_date)

    ema_fast = parse_int(get_env("EMA_FAST", None), ema_fast)
    ema_slow = parse_int(get_env("EMA_SLOW", None), ema_slow)
    atr_period = parse_int(get_env("ATR_PERIOD", None), atr_period)

    if ema_fast >= ema_slow:
        raise ValueError("Para esta estrategia, ema_fast debe ser < ema_slow (ej. 20 y 50).")

    return FeatureSettings(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr_period=atr_period,
    )


def build_raw_path(s: FeatureSettings) -> Path:
    sym = safe_symbol_for_filename(s.symbol)
    fname = f"{s.exchange}_{sym}_{s.timeframe}_{s.start_date}_to_{s.end_date}.csv"
    return RAW_DIR / fname


def build_out_path(s: FeatureSettings) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sym = safe_symbol_for_filename(s.symbol)
    fname = f"{s.exchange}_{sym}_{s.timeframe}_{s.start_date}_to_{s.end_date}_features.csv"
    return PROCESSED_DIR / fname


# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    # Pandas EWM uses alpha=2/(span+1) which matches common EMA definition
    return series.ewm(span=span, adjust=False).mean()


def rma_wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RMA (a.k.a. smoothed moving average):
    RMA_t = (RMA_{t-1}*(period-1) + x_t)/period
    Implemented via ewm(alpha=1/period, adjust=False)
    """
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return rma_wilder(tr, period)

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    # Wilder smoothing = EMA con alpha = 1/period
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX clásico (Welles Wilder) usando Wilder smoothing.
    high/low/close deben estar indexados por datetime y sin huecos grandes.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_sm = _wilder_ema(tr, period)
    plus_dm_sm = _wilder_ema(pd.Series(plus_dm, index=high.index), period)
    minus_dm_sm = _wilder_ema(pd.Series(minus_dm, index=high.index), period)

    plus_di = 100.0 * (plus_dm_sm / tr_sm.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_sm / tr_sm.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = _wilder_ema(dx, period)

    return adx


# -----------------------------
# Validation
# -----------------------------
def validate_1h(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    - Ordena por timestamp
    - Dedup
    - Chequea frecuencia constante (1h)
    """
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    after = len(df)

    ts = df["timestamp"].astype("int64")
    diffs = ts.diff().dropna()
    expected_ms = 3600 * 1000
    gaps = int((diffs > expected_ms).sum())
    backwards = int((diffs < expected_ms).sum())  # si hay desorden o duplicados raros

    stats = {
        "rows": int(after),
        "duplicates_removed": int(before - after),
        "gaps": gaps,
        "backwards_or_weird_steps": backwards,
    }
    return df, stats


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = resolve_settings()

    raw_path = build_raw_path(s)
    if not raw_path.exists():
        print(f"[ERROR] No encuentro el raw CSV esperado: {raw_path}")
        print("        Verificá EXCHANGE/SYMBOL/TIMEFRAME/START_DATE/END_DATE o el nombre del archivo.")
        return 1

    print("[INFO] Leyendo:", raw_path)
    df = pd.read_csv(raw_path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Faltan columnas requeridas: {missing}")
        return 1

    df, vstats = validate_1h(df)
    if vstats["backwards_or_weird_steps"] > 0:
        print("[WARN] Se detectaron pasos raros en timestamps (posible desorden). Se ordenó igualmente.")

    # Asegurar datetime_utc consistente
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Cast a float para seguridad
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Features
    print(f"[INFO] Calculando EMA({s.ema_fast}), EMA({s.ema_slow}), ATR({s.atr_period}), Donchian, HTF")
    df[f"ema_{s.ema_fast}"] = ema(df["close"], s.ema_fast)
    df[f"ema_{s.ema_slow}"] = ema(df["close"], s.ema_slow)
    df[f"atr_{s.atr_period}"] = atr_wilder(df, s.atr_period)

    # Donchian Channels (excluye la vela actual)
    donchian_n = int(os.getenv("DONCHIAN_N", "20"))
    df[f"donchian_high_{donchian_n}"] = df["high"].rolling(donchian_n).max().shift(1)
    df[f"donchian_low_{donchian_n}"] = df["low"].rolling(donchian_n).min().shift(1)


    df[f"atr_sma_{s.atr_period}"] = df[f"atr_{s.atr_period}"].rolling(s.atr_period).mean()

    # Aux (útil para backtest/reporting)
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    #df["logret_1"] = (df["close"] / df["close"].shift(1)).apply(lambda x: 0.0 if pd.isna(x) else float(pd.np.log(x)))  # compat
    ratio = (df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    df["logret_1"] = np.log(ratio).fillna(0.0)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

   # ---------- HTF Regime (4H) + ADX gate ----------
    htf = os.getenv("HTF", "4H").upper()     # "4H" (recomendado para el MVP)
    htf_ema = int(os.getenv("HTF_EMA", "200"))
    adx_period = int(os.getenv("ADX_PERIOD", "14"))

    tmp = df[["datetime_utc", "open", "high", "low", "close", "volume"]].copy()
    tmp = tmp.set_index("datetime_utc")

    ohlc = tmp.resample(htf, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # EMA en HTF
    ohlc[f"ema_{htf_ema}"] = ohlc["close"].ewm(span=htf_ema, adjust=False).mean()

    # Regime: +1 bull, -1 bear
    ohlc["regime"] = np.where(ohlc["close"] >= ohlc[f"ema_{htf_ema}"], 1, -1)

    # ADX en HTF
    ohlc[f"adx_{htf.lower()}_{adx_period}"] = compute_adx(
        ohlc["high"], ohlc["low"], ohlc["close"], period=adx_period
    )

    # Alinear a 1H sin lookahead: cada 1H hereda el último valor CERRADO del HTF
    df = df.set_index("datetime_utc")
    df["regime_htf"] = ohlc["regime"].reindex(df.index, method="ffill").fillna(0).astype(int)
    df[f"adx_{htf.lower()}_{adx_period}"] = ohlc[f"adx_{htf.lower()}_{adx_period}"].reindex(df.index, method="ffill")

    df = df.reset_index()



    # Limpiar np deprec warning (pandas np)
    # Nota: dejamos el cálculo como está por simplicidad MVP. Si querés, lo cambiamos por numpy.

    out_path = build_out_path(s)
    df.to_csv(out_path, index=False)

    print("[INFO] OK features")
    print(f"  output: {out_path}")
    print(f"  rows  : {vstats['rows']}")
    print(f"  dup_rm: {vstats['duplicates_removed']}")
    print(f"  gaps  : {vstats['gaps']}")
    if vstats["gaps"] > 0:
        print("[WARN] Hay gaps. El backtest va a seguir, pero conviene revisarlos luego.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

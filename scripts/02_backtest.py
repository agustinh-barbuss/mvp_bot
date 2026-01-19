"""
02_backtest.py (v3 + size modulator)
Donchian breakout + ATR trailing stop (both) + HTF regime + ATR expansion + buffer
+ modulador de tamaño por Fear & Greed (Alternative.me).

- Entry signal computed on CLOSE of bar i using Donchian high/low (N) excluding current bar.
- Entry executed at OPEN of bar i+1 (no lookahead).
- Exit via ATR trailing stop evaluated using bar range (high/low) on bar i.
- Costs: fee_bps + slippage_bps applied on fills.
- Size modulator: ajusta qty según FNG (solo afecta tamaño, no señales).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FNG_PATH = EXTERNAL_DIR / "fng.csv"


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


def parse_float(s: Optional[str], default: float) -> float:
    if s is None:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_int(s: Optional[str], default: int) -> int:
    if s is None:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def size_multiplier_from_fng(fng: float, side: str, floor: float) -> float:
    """
    side: 'LONG' o 'SHORT'
    floor: multiplicador mínimo (ej 0.5)
    Regla suave:
      - LONG se reduce cuando fng > 50
      - SHORT se reduce cuando fng < 50
    """
    if not np.isfinite(fng):
        return 1.0

    floor = float(max(0.0, min(1.0, floor)))

    if side == "LONG":
        if fng <= 50:
            return 1.0
        x = (fng - 50.0) / 50.0  # 0..1
        return max(floor, 1.0 - (1.0 - floor) * x)

    if side == "SHORT":
        if fng >= 50:
            return 1.0
        x = (50.0 - fng) / 50.0  # 0..1
        return max(floor, 1.0 - (1.0 - floor) * x)

    return 1.0


@dataclass
class BacktestSettings:
    exchange: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str

    atr_period: int
    donchian_n: int
    atr_trail_mult: float
    min_atr_ratio: float

    fee_bps: float
    slippage_bps: float

    initial_equity: float
    risk_per_trade: float

    direction: str  # both / long_only

    atr_exp_mult: float
    breakout_buffer: float

    # Sentiment
    sent_enabled: bool
    fng_floor_mult: float


def resolve_settings() -> BacktestSettings:
    cfg = load_yaml_config(CONFIG_PATH)
    market = (cfg.get("market") or {})
    strat = (cfg.get("strategy") or {})
    costs = (cfg.get("costs") or {})
    bt = (cfg.get("backtest") or {})
    sent = (cfg.get("sentiment") or {})

    exchange = get_env("EXCHANGE", market.get("exchange", "binance"))
    symbol = get_env("SYMBOL", market.get("symbol", "BTC/USDT"))
    timeframe = get_env("TIMEFRAME", market.get("timeframe", "1h"))
    start_date = get_env("START_DATE", market.get("start_date", "2019-01-01"))

    end_date = (cfg.get("validation") or {}).get("test_end", None)
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_date = get_env("END_DATE", end_date)

    atr_period = int(strat.get("atr_period", 14))
    donchian_n = parse_int(get_env("DONCHIAN_N", None), int(strat.get("donchian_n", 20)))
    atr_trail_mult = parse_float(get_env("ATR_TRAIL_MULT", None), float(strat.get("atr_trail_mult", 3.0)))
    min_atr_ratio = parse_float(get_env("MIN_ATR_RATIO", None), float(strat.get("min_atr_ratio", 0.0015)))

    fee_bps = parse_float(get_env("FEE_BPS", None), float(costs.get("fee_bps", 7.5)))
    slippage_bps = parse_float(get_env("SLIPPAGE_BPS", None), float(costs.get("slippage_bps", 2.0)))

    breakout_buffer = parse_float(get_env("BREAKOUT_BUFFER", None), float(strat.get("breakout_buffer", 0.0010)))
    atr_exp_mult = parse_float(get_env("ATR_EXP_MULT", None), float(strat.get("atr_exp_mult", 1.05)))

    initial_equity = parse_float(get_env("INITIAL_EQUITY", None), float(bt.get("initial_equity", 10_000.0)))
    risk_per_trade = parse_float(get_env("RISK_PER_TRADE", None), float(bt.get("risk_per_trade", 0.01)))
    direction = get_env("DIRECTION", str(bt.get("direction", "both")))

    sent_enabled = bool(sent.get("enabled", True))
    fng_floor_mult = parse_float(get_env("FNG_FLOOR_MULT", None), float(sent.get("fng_floor_mult", 0.50)))

    return BacktestSettings(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        atr_period=atr_period,
        donchian_n=donchian_n,
        atr_trail_mult=atr_trail_mult,
        min_atr_ratio=min_atr_ratio,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        direction=direction,
        breakout_buffer=breakout_buffer,
        atr_exp_mult=atr_exp_mult,
        sent_enabled=sent_enabled,
        fng_floor_mult=fng_floor_mult,
    )


def build_features_path(s: BacktestSettings) -> Path:
    sym = safe_symbol_for_filename(s.symbol)
    return PROCESSED_DIR / f"{s.exchange}_{sym}_{s.timeframe}_{s.start_date}_to_{s.end_date}_features.csv"


def make_run_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_slippage(price: float, bps: float, side: str) -> float:
    frac = bps / 10_000.0
    if side == "buy":
        return price * (1.0 + frac)
    if side == "sell":
        return price * (1.0 - frac)
    raise ValueError("side debe ser 'buy' o 'sell'")


def fee_cost(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10_000.0)


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict[str, Any]:
    if equity.empty:
        return {"error": "equity vacío"}
    eq = equity["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(8760)) if rets.std() > 0 else 0.0

    out: Dict[str, Any] = {
        "final_equity": float(eq.iloc[-1]),
        "max_drawdown": max_dd,
        "sharpe_approx": sharpe,
        "trades": int(len(trades)),
    }
    if trades.empty:
        out.update({"win_rate": 0.0, "profit_factor": 0.0})
        return out

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    out["win_rate"] = float(len(wins) / len(trades))

    gp = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gl = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    out["profit_factor"] = float(gp / gl) if gl > 0 else float("inf")

    out["avg_pnl"] = float(trades["pnl"].mean())
    out["median_pnl"] = float(trades["pnl"].median())
    out["avg_hold_bars"] = float(trades["bars_held"].mean())
    out["long_trades"] = int((trades["side"] == "LONG").sum())
    out["short_trades"] = int((trades["side"] == "SHORT").sum())
    return out


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = resolve_settings()

    path = build_features_path(s)
    if not path.exists():
        print(f"[ERROR] No encuentro el features CSV: {path}")
        return 1

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # --- Sentiment merge (FNG) ---
    df["timestamp_s"] = (df["timestamp"] // 1000).astype(int)

    if s.sent_enabled:
        if not FNG_PATH.exists():
            print(f"[ERROR] No existe {FNG_PATH}. Corré: python scripts\\04_add_sentiment.py")
            return 1

        fng = pd.read_csv(FNG_PATH)
        if "timestamp_s" not in fng.columns or "fng_value" not in fng.columns:
            print("[ERROR] fng.csv inválido: faltan columnas timestamp_s y/o fng_value")
            return 1

        fng["timestamp_s"] = fng["timestamp_s"].astype(int)
        fng["fng_value"] = fng["fng_value"].astype(float)
        fng = fng.sort_values("timestamp_s").reset_index(drop=True)

        df = df.sort_values("timestamp_s").reset_index(drop=True)
        df = pd.merge_asof(
            df,
            fng[["timestamp_s", "fng_value"]],
            on="timestamp_s",
            direction="backward",
            allow_exact_matches=True,
        )
        df["fng_value"] = df["fng_value"].fillna(50.0)
    else:
        df["fng_value"] = 50.0

    atr_col = f"atr_{s.atr_period}"
    dh_col = f"donchian_high_{s.donchian_n}"
    dl_col = f"donchian_low_{s.donchian_n}"
    atr_sma_col = f"atr_sma_{s.atr_period}"

    required = {"timestamp", "open", "high", "low", "close", atr_col, dh_col, dl_col, atr_sma_col, "regime_htf"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Faltan columnas requeridas: {missing}")
        return 1

    for c in ["open", "high", "low", "close", atr_col, atr_sma_col, dh_col, dl_col]:
        df[c] = df[c].astype(float)

    atr = df[atr_col]
    atr_sma = df[atr_sma_col]
    close = df["close"]
    ok_vol = (atr / close) >= s.min_atr_ratio

    regime = df["regime_htf"].astype(int)
    buffer = s.breakout_buffer
    exp_ok = atr > (atr_sma * s.atr_exp_mult)

    long_signal = ok_vol & (regime == 1) & exp_ok & (close > df[dh_col] * (1 + buffer))
    short_signal = ok_vol & (regime == -1) & exp_ok & (close < df[dl_col] * (1 - buffer))

    equity = s.initial_equity

    position = 0
    qty = 0.0
    entry_price = 0.0
    entry_time = None
    trail_stop = None
    entry_equity = None
    entry_bar_i = None
    entry_mult = 1.0

    trades_rows: List[dict] = []
    equity_rows: List[dict] = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        ts = int(row["timestamp"])
        dt = row["datetime_utc"]
        equity_rows.append({"timestamp": ts, "datetime_utc": dt, "equity": equity})

        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # EXIT
        if position == 1:
            new_stop = float(row["close"]) - s.atr_trail_mult * atr_i
            trail_stop = max(trail_stop, new_stop)

            if float(row["low"]) <= trail_stop:
                exit_raw = float(trail_stop)
                exit_px = apply_slippage(exit_raw, s.slippage_bps, side="sell")

                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (exit_px - entry_price)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl

                trades_rows.append({
                    "side": "LONG",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": dt.isoformat(),
                    "exit_price": exit_px,
                    "exit_reason": "TRAIL",
                    "qty": qty,
                    "size_mult": entry_mult,
                    "pnl": pnl,
                    "equity_before": entry_equity,
                    "equity_after": equity,
                    "bars_held": int(i - entry_bar_i),
                })

                position = 0
                qty = 0.0
                trail_stop = None

        elif position == -1:
            new_stop = float(row["close"]) + s.atr_trail_mult * atr_i
            trail_stop = min(trail_stop, new_stop)

            if float(row["high"]) >= trail_stop:
                exit_raw = float(trail_stop)
                exit_px = apply_slippage(exit_raw, s.slippage_bps, side="buy")

                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (entry_price - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl

                trades_rows.append({
                    "side": "SHORT",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": dt.isoformat(),
                    "exit_price": exit_px,
                    "exit_reason": "TRAIL",
                    "qty": qty,
                    "size_mult": entry_mult,
                    "pnl": pnl,
                    "equity_before": entry_equity,
                    "equity_after": equity,
                    "bars_held": int(i - entry_bar_i),
                })

                position = 0
                qty = 0.0
                trail_stop = None

        # ENTRY (flat only)
        if position == 0:
            if not bool(ok_vol.iloc[i]):
                continue

            stop_dist = s.atr_trail_mult * atr_i
            if stop_dist <= 0:
                continue

            risk_amount = equity * s.risk_per_trade
            qty_calc = risk_amount / stop_dist
            if qty_calc <= 0:
                continue

            fng_val = float(df.iloc[i]["fng_value"])

            if s.direction in ("both", "long_only") and bool(long_signal.iloc[i]):
                entry_raw = float(next_row["open"])
                entry_px = apply_slippage(entry_raw, s.slippage_bps, side="buy")

                mult = size_multiplier_from_fng(fng_val, "LONG", s.fng_floor_mult)
                qty = qty_calc * mult

                if qty <= 0:
                    continue

                position = 1
                entry_price = entry_px
                entry_time = next_row["datetime_utc"].isoformat()
                entry_equity = equity
                entry_bar_i = i + 1
                trail_stop = entry_raw - stop_dist
                entry_mult = mult

            elif s.direction == "both" and bool(short_signal.iloc[i]):
                entry_raw = float(next_row["open"])
                entry_px = apply_slippage(entry_raw, s.slippage_bps, side="sell")

                mult = size_multiplier_from_fng(fng_val, "SHORT", s.fng_floor_mult)
                qty = qty_calc * mult

                if qty <= 0:
                    continue

                position = -1
                entry_price = entry_px
                entry_time = next_row["datetime_utc"].isoformat()
                entry_equity = equity
                entry_bar_i = i + 1
                trail_stop = entry_raw + stop_dist
                entry_mult = mult

    # Final equity point
    if len(df) > 0:
        last = df.iloc[-1]
        equity_rows.append({"timestamp": int(last["timestamp"]), "datetime_utc": last["datetime_utc"], "equity": equity})

    trades = pd.DataFrame(trades_rows)
    eq = pd.DataFrame(equity_rows)

    run_dir = make_run_dir()
    used_cfg = s.__dict__
    (run_dir / "config_used.json").write_text(json.dumps(used_cfg, indent=2), encoding="utf-8")

    trades.to_csv(run_dir / "trades.csv", index=False)
    eq.to_csv(run_dir / "equity.csv", index=False)

    metrics = compute_metrics(trades, eq)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[INFO] Backtest OK (v3 + sentiment size modulator)")
    print(f"  run_dir : {run_dir}")
    print(f"  trades  : {len(trades)}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

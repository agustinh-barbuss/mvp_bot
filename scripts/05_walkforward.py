"""
05_walkforward.py (v4.9 parity con 02)
Walk-forward rolling:
- train_days (solo para mover la ventana)
- test_days (se backtestea solo el período TEST)
- equity encadenada entre ventanas
- fuerza cierre al final de cada ventana (EOD del slice)

Outputs:
- outputs/walkforward/<run_id>/walkforward_windows.csv
- outputs/walkforward/<run_id>/summary.json
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = PROJECT_ROOT / "config" / "base.yaml"
CONFIG_YML  = PROJECT_ROOT / "config" / "base.yml"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_WF_DIR = PROJECT_ROOT / "outputs" / "walkforward"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FNG_PATH = EXTERNAL_DIR / "fng.csv"


# ----------------------------
# Helpers
# ----------------------------
def safe_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "-")


def load_yaml_config() -> dict:
    path = CONFIG_YAML if CONFIG_YAML.exists() else CONFIG_YML
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


def apply_slippage(price: float, bps: float, side: str) -> float:
    frac = bps / 10_000.0
    if side == "buy":
        return price * (1.0 + frac)
    if side == "sell":
        return price * (1.0 - frac)
    raise ValueError("side debe ser 'buy' o 'sell'")


def fee_cost(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10_000.0)


def size_multiplier_from_fng(fng: float, side: str, floor: float) -> float:
    if not np.isfinite(fng):
        return 1.0
    floor = float(max(0.0, min(1.0, floor)))

    if side == "LONG":
        if fng <= 50:
            return 1.0
        x = (fng - 50.0) / 50.0
        return max(floor, 1.0 - (1.0 - floor) * x)

    if side == "SHORT":
        if fng >= 50:
            return 1.0
        x = (50.0 - fng) / 50.0
        return max(floor, 1.0 - (1.0 - floor) * x)

    return 1.0


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


def make_run_dir(base_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ----------------------------
# Settings
# ----------------------------
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
    atr_exp_mult: float
    breakout_buffer: float

    adx_period: int
    adx_min_4h: float

    fee_bps: float
    slippage_bps: float

    initial_equity: float
    risk_per_trade: float
    direction: str

    sent_enabled: bool
    fng_floor_mult: float

    use_pullback_entry: bool
    pullback_max_wait_bars: int
    pullback_level: str
    pullback_fallback_market: bool
    pullback_fallback_after_bars: int

    confirm_buffer: float

    use_time_stop: bool
    time_stop_bars: int
    time_stop_only_if_pnl_leq_zero: bool

    use_take_profit: bool
    tp_r_mult: float


@dataclass
class WalkForwardSettings:
    train_days: int
    test_days: int
    min_test_bars: int


def resolve_settings() -> Tuple[BacktestSettings, WalkForwardSettings, dict]:
    cfg = load_yaml_config()
    market = (cfg.get("market") or {})
    strat = (cfg.get("strategy") or {})
    costs = (cfg.get("costs") or {})
    bt = (cfg.get("backtest") or {})
    sent = (cfg.get("sentiment") or {})
    wf = (cfg.get("walkforward") or {})

    exchange = get_env("EXCHANGE", market.get("exchange", "binance"))
    symbol = get_env("SYMBOL", market.get("symbol", "BTC/USDT"))
    timeframe = get_env("TIMEFRAME", market.get("timeframe", "1h"))
    start_date = get_env("START_DATE", market.get("start_date", "2019-01-01"))

    end_date = (cfg.get("validation") or {}).get("test_end", None)
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_date = get_env("END_DATE", end_date)

    atr_period = parse_int(get_env("ATR_PERIOD", None), int(strat.get("atr_period", 14)))
    donchian_n = parse_int(get_env("DONCHIAN_N", None), int(strat.get("donchian_n", 20)))
    atr_trail_mult = parse_float(get_env("ATR_TRAIL_MULT", None), float(strat.get("atr_trail_mult", 3.0)))
    min_atr_ratio = parse_float(get_env("MIN_ATR_RATIO", None), float(strat.get("min_atr_ratio", 0.0015)))
    atr_exp_mult = parse_float(get_env("ATR_EXP_MULT", None), float(strat.get("atr_exp_mult", 1.05)))
    breakout_buffer = parse_float(get_env("BREAKOUT_BUFFER", None), float(strat.get("breakout_buffer", 0.0010)))

    adx_period = parse_int(get_env("ADX_PERIOD", None), int(strat.get("adx_period", 14)))
    adx_min_4h = parse_float(get_env("ADX_MIN_4H", None), float(strat.get("adx_min_4h", 30.0)))

    fee_bps = parse_float(get_env("FEE_BPS", None), float(costs.get("fee_bps", 7.5)))
    slippage_bps = parse_float(get_env("SLIPPAGE_BPS", None), float(costs.get("slippage_bps", 2.0)))

    initial_equity = parse_float(get_env("INITIAL_EQUITY", None), float(bt.get("initial_equity", 10_000.0)))
    risk_per_trade = parse_float(get_env("RISK_PER_TRADE", None), float(bt.get("risk_per_trade", 0.01)))
    direction = get_env("DIRECTION", str(bt.get("direction", "both")))

    sent_enabled = bool(sent.get("enabled", True))
    fng_floor_mult = parse_float(get_env("FNG_FLOOR_MULT", None), float(sent.get("fng_floor_mult", 0.10)))

    use_pullback_entry = bool(strat.get("use_pullback_entry", True))
    pullback_max_wait_bars = parse_int(get_env("PULLBACK_MAX_WAIT_BARS", None), int(strat.get("pullback_max_wait_bars", 12)))
    pullback_level = str(strat.get("pullback_level", "channel")).strip().lower()
    if pullback_level not in ("channel", "mid"):
        pullback_level = "channel"

    pullback_fallback_market = bool(strat.get("pullback_fallback_market", False))
    pullback_fallback_after_bars = parse_int(get_env("PULLBACK_FALLBACK_AFTER_BARS", None), int(strat.get("pullback_fallback_after_bars", 6)))

    confirm_buffer = parse_float(get_env("CONFIRM_BUFFER", None), float(strat.get("confirm_buffer", 0.0005)))

    use_time_stop = bool(strat.get("use_time_stop", False))
    time_stop_bars = parse_int(get_env("TIME_STOP_BARS", None), int(strat.get("time_stop_bars", 72)))
    time_stop_only_if_pnl_leq_zero = bool(strat.get("time_stop_only_if_pnl_leq_zero", True))

    use_take_profit = bool(strat.get("use_take_profit", True))
    tp_r_mult = parse_float(get_env("TP_R_MULT", None), float(strat.get("tp_r_mult", 2.0)))

    bt_s = BacktestSettings(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        atr_period=atr_period,
        donchian_n=donchian_n,
        atr_trail_mult=atr_trail_mult,
        min_atr_ratio=min_atr_ratio,
        atr_exp_mult=atr_exp_mult,
        breakout_buffer=breakout_buffer,
        adx_period=adx_period,
        adx_min_4h=adx_min_4h,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        direction=direction,
        sent_enabled=sent_enabled,
        fng_floor_mult=fng_floor_mult,
        use_pullback_entry=use_pullback_entry,
        pullback_max_wait_bars=pullback_max_wait_bars,
        pullback_level=pullback_level,
        pullback_fallback_market=pullback_fallback_market,
        pullback_fallback_after_bars=pullback_fallback_after_bars,
        confirm_buffer=confirm_buffer,
        use_time_stop=use_time_stop,
        time_stop_bars=time_stop_bars,
        time_stop_only_if_pnl_leq_zero=time_stop_only_if_pnl_leq_zero,
        use_take_profit=use_take_profit,
        tp_r_mult=tp_r_mult,
    )

    wf_s = WalkForwardSettings(
        train_days=parse_int(get_env("WF_TRAIN_DAYS", None), int(wf.get("train_days", 180))),
        test_days=parse_int(get_env("WF_TEST_DAYS", None), int(wf.get("test_days", 90))),
        min_test_bars=parse_int(get_env("WF_MIN_TEST_BARS", None), int(wf.get("min_test_bars", 300))),
    )

    return bt_s, wf_s, cfg


def build_features_path(s: BacktestSettings) -> Path:
    sym = safe_symbol_for_filename(s.symbol)
    return PROCESSED_DIR / f"{s.exchange}_{sym}_{s.timeframe}_{s.start_date}_to_{s.end_date}_features.csv"


def load_features_df(s: BacktestSettings) -> pd.DataFrame:
    path = build_features_path(s)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el features CSV: {path}")

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp_s"] = (df["timestamp"] // 1000).astype(int)

    if s.sent_enabled:
        if not FNG_PATH.exists():
            raise FileNotFoundError(f"Falta {FNG_PATH}. Corré: python scripts\\04_add_sentiment.py")
        fng = pd.read_csv(FNG_PATH)
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

    return df


# ----------------------------
# Core engine (paridad con 02_backtest.py v4.9)
# ----------------------------
def run_engine(
    df: pd.DataFrame,
    s: BacktestSettings,
    equity_start: float,
    force_close_end: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    atr_col = f"atr_{s.atr_period}"
    atr_sma_col = f"atr_sma_{s.atr_period}"
    dh_col = f"donchian_high_{s.donchian_n}"
    dl_col = f"donchian_low_{s.donchian_n}"
    adx_col = f"adx_4h_{s.adx_period}"

    required = {"timestamp", "open", "high", "low", "close", "regime_htf", atr_col, atr_sma_col, dh_col, dl_col, adx_col, "fng_value", "datetime_utc"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas requeridas: {missing}")

    for c in ["open", "high", "low", "close", atr_col, atr_sma_col, dh_col, dl_col, adx_col, "fng_value"]:
        df[c] = df[c].astype(float)
    df["regime_htf"] = df["regime_htf"].astype(int)

    atr = df[atr_col].astype(float)
    atr_sma = df[atr_sma_col].astype(float)
    close = df["close"].astype(float)

    ok_vol = (atr / close) >= float(s.min_atr_ratio)
    exp_ok = atr > (atr_sma * float(s.atr_exp_mult))
    regime = df["regime_htf"].astype(int)
    adx_ok = df[adx_col].astype(float) >= float(s.adx_min_4h)

    buffer = float(s.breakout_buffer)
    long_signal  = ok_vol & exp_ok & adx_ok & (regime == 1)  & (close > df[dh_col] * (1.0 + buffer))
    short_signal = ok_vol & exp_ok & adx_ok & (regime == -1) & (close < df[dl_col] * (1.0 - buffer))

    equity = float(equity_start)

    position = 0
    qty = 0.0
    entry_px = 0.0
    entry_time = None
    entry_equity = None
    entry_bar_i = None
    entry_mult = 1.0
    entry_type = "MKT"
    trail_stop = None
    tp_price = None

    pending: Optional[Dict[str, Any]] = None

    st = {
        "pending_created": 0,
        "pending_filled": 0,
        "pending_expired": 0,
        "pending_touch_rejected_by_close": 0,
        "pending_canceled_invalid": 0,
        "entry_limit_fills": 0,
        "entry_fallback_market": 0,
        "exit_time_stop": 0,
        "exit_take_profit": 0,
    }

    trades_rows: List[dict] = []
    equity_rows: List[dict] = []

    n = len(df)
    if n < 5:
        eq = pd.DataFrame([{"timestamp": int(df.iloc[-1]["timestamp"]), "datetime_utc": df.iloc[-1]["datetime_utc"], "equity": equity}])
        tr = pd.DataFrame([])
        return tr, eq, compute_metrics(tr, eq), st

    for i in range(1, n - 1):
        row = df.iloc[i]
        ts = int(row["timestamp"])
        dt = row["datetime_utc"]
        equity_rows.append({"timestamp": ts, "datetime_utc": dt, "equity": equity})

        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # EXIT
        if position != 0:
            bars_held = int(i - entry_bar_i)

            if position == 1:
                new_stop = float(row["close"]) - float(s.atr_trail_mult) * atr_i
                trail_stop = max(trail_stop, new_stop)
            else:
                new_stop = float(row["close"]) + float(s.atr_trail_mult) * atr_i
                trail_stop = min(trail_stop, new_stop)

            # TIME STOP (next open)
            if s.use_time_stop and bars_held >= int(s.time_stop_bars):
                if (i + 1) < n:
                    if s.time_stop_only_if_pnl_leq_zero:
                        if position == 1:
                            unreal = (float(row["close"]) - float(entry_px)) * qty
                        else:
                            unreal = (float(entry_px) - float(row["close"])) * qty
                        if unreal <= 0:
                            exit_raw = float(df.iloc[i + 1]["open"])
                            exit_px = apply_slippage(exit_raw, s.slippage_bps, side=("sell" if position == 1 else "buy"))
                            not_entry = qty * float(entry_px)
                            not_exit = qty * float(exit_px)
                            if position == 1:
                                pnl = (qty * (exit_px - entry_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                            else:
                                pnl = (qty * (entry_px - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                            equity += pnl
                            trades_rows.append({
                                "side": "LONG" if position == 1 else "SHORT",
                                "entry_time": entry_time,
                                "entry_price": float(entry_px),
                                "entry_type": entry_type,
                                "exit_time": df.iloc[i + 1]["datetime_utc"].isoformat(),
                                "exit_price": float(exit_px),
                                "exit_reason": "TIME",
                                "qty": float(qty),
                                "size_mult": float(entry_mult),
                                "pnl": float(pnl),
                                "equity_before": float(entry_equity),
                                "equity_after": float(equity),
                                "bars_held": int(bars_held),
                            })
                            st["exit_time_stop"] += 1
                            position = 0
                            qty = 0.0
                            trail_stop = None
                            tp_price = None
                            entry_mult = 1.0
                            entry_type = "MKT"
                            pending = None
                            continue
                    else:
                        exit_raw = float(df.iloc[i + 1]["open"])
                        exit_px = apply_slippage(exit_raw, s.slippage_bps, side=("sell" if position == 1 else "buy"))
                        not_entry = qty * float(entry_px)
                        not_exit = qty * float(exit_px)
                        if position == 1:
                            pnl = (qty * (exit_px - entry_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                        else:
                            pnl = (qty * (entry_px - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                        equity += pnl
                        trades_rows.append({
                            "side": "LONG" if position == 1 else "SHORT",
                            "entry_time": entry_time,
                            "entry_price": float(entry_px),
                            "entry_type": entry_type,
                            "exit_time": df.iloc[i + 1]["datetime_utc"].isoformat(),
                            "exit_price": float(exit_px),
                            "exit_reason": "TIME",
                            "qty": float(qty),
                            "size_mult": float(entry_mult),
                            "pnl": float(pnl),
                            "equity_before": float(entry_equity),
                            "equity_after": float(equity),
                            "bars_held": int(bars_held),
                        })
                        st["exit_time_stop"] += 1
                        position = 0
                        qty = 0.0
                        trail_stop = None
                        tp_price = None
                        entry_mult = 1.0
                        entry_type = "MKT"
                        pending = None
                        continue

            # STOP vs TP (stop first if both)
            if position == 1:
                stop_hit = float(row["low"]) <= float(trail_stop)
                tp_hit = False
                if s.use_take_profit and tp_price is not None:
                    tp_hit = float(row["high"]) >= float(tp_price)

                if stop_hit:
                    exit_raw = float(trail_stop)
                    exit_px = apply_slippage(exit_raw, s.slippage_bps, side="sell")
                    not_entry = qty * float(entry_px)
                    not_exit = qty * float(exit_px)
                    pnl = (qty * (exit_px - entry_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                    equity += pnl
                    trades_rows.append({
                        "side": "LONG",
                        "entry_time": entry_time,
                        "entry_price": float(entry_px),
                        "entry_type": entry_type,
                        "exit_time": dt.isoformat(),
                        "exit_price": float(exit_px),
                        "exit_reason": "TRAIL",
                        "qty": float(qty),
                        "size_mult": float(entry_mult),
                        "pnl": float(pnl),
                        "equity_before": float(entry_equity),
                        "equity_after": float(equity),
                        "bars_held": int(bars_held),
                    })
                    position = 0
                    qty = 0.0
                    trail_stop = None
                    tp_price = None
                    entry_mult = 1.0
                    entry_type = "MKT"
                    pending = None
                    continue

                if tp_hit:
                    exit_raw = float(tp_price)
                    exit_px = apply_slippage(exit_raw, s.slippage_bps, side="sell")
                    not_entry = qty * float(entry_px)
                    not_exit = qty * float(exit_px)
                    pnl = (qty * (exit_px - entry_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                    equity += pnl
                    trades_rows.append({
                        "side": "LONG",
                        "entry_time": entry_time,
                        "entry_price": float(entry_px),
                        "entry_type": entry_type,
                        "exit_time": dt.isoformat(),
                        "exit_price": float(exit_px),
                        "exit_reason": "TP",
                        "qty": float(qty),
                        "size_mult": float(entry_mult),
                        "pnl": float(pnl),
                        "equity_before": float(entry_equity),
                        "equity_after": float(equity),
                        "bars_held": int(bars_held),
                    })
                    st["exit_take_profit"] += 1
                    position = 0
                    qty = 0.0
                    trail_stop = None
                    tp_price = None
                    entry_mult = 1.0
                    entry_type = "MKT"
                    pending = None
                    continue
            else:
                stop_hit = float(row["high"]) >= float(trail_stop)
                tp_hit = False
                if s.use_take_profit and tp_price is not None:
                    tp_hit = float(row["low"]) <= float(tp_price)

                if stop_hit:
                    exit_raw = float(trail_stop)
                    exit_px = apply_slippage(exit_raw, s.slippage_bps, side="buy")
                    not_entry = qty * float(entry_px)
                    not_exit = qty * float(exit_px)
                    pnl = (qty * (entry_px - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                    equity += pnl
                    trades_rows.append({
                        "side": "SHORT",
                        "entry_time": entry_time,
                        "entry_price": float(entry_px),
                        "entry_type": entry_type,
                        "exit_time": dt.isoformat(),
                        "exit_price": float(exit_px),
                        "exit_reason": "TRAIL",
                        "qty": float(qty),
                        "size_mult": float(entry_mult),
                        "pnl": float(pnl),
                        "equity_before": float(entry_equity),
                        "equity_after": float(equity),
                        "bars_held": int(bars_held),
                    })
                    position = 0
                    qty = 0.0
                    trail_stop = None
                    tp_price = None
                    entry_mult = 1.0
                    entry_type = "MKT"
                    pending = None
                    continue

                if tp_hit:
                    exit_raw = float(tp_price)
                    exit_px = apply_slippage(exit_raw, s.slippage_bps, side="buy")
                    not_entry = qty * float(entry_px)
                    not_exit = qty * float(exit_px)
                    pnl = (qty * (entry_px - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                    equity += pnl
                    trades_rows.append({
                        "side": "SHORT",
                        "entry_time": entry_time,
                        "entry_price": float(entry_px),
                        "entry_type": entry_type,
                        "exit_time": dt.isoformat(),
                        "exit_price": float(exit_px),
                        "exit_reason": "TP",
                        "qty": float(qty),
                        "size_mult": float(entry_mult),
                        "pnl": float(pnl),
                        "equity_before": float(entry_equity),
                        "equity_after": float(equity),
                        "bars_held": int(bars_held),
                    })
                    st["exit_take_profit"] += 1
                    position = 0
                    qty = 0.0
                    trail_stop = None
                    tp_price = None
                    entry_mult = 1.0
                    entry_type = "MKT"
                    pending = None
                    continue

            continue

        # FLAT: pending
        if pending is not None:
            if i >= int(pending["start_i"]):
                if i > int(pending["expire_i"]):
                    st["pending_expired"] += 1
                    pending = None
                else:
                    side = str(pending["side"])
                    limit_px = float(pending["limit"])
                    touched = (float(row["low"]) <= limit_px <= float(row["high"]))
                    if touched:
                        cb = float(s.confirm_buffer)
                        if side == "LONG":
                            confirm_ok = float(row["close"]) > (limit_px * (1.0 + float(s.breakout_buffer) + cb))
                        else:
                            confirm_ok = float(row["close"]) < (limit_px * (1.0 - float(s.breakout_buffer) - cb))

                        if confirm_ok:
                            stop_dist = float(s.atr_trail_mult) * atr_i
                            if stop_dist > 0:
                                risk_amount = equity * float(s.risk_per_trade)
                                qty_calc = risk_amount / stop_dist
                                fng_val = float(row["fng_value"])

                                if side == "LONG" and s.direction in ("both", "long_only"):
                                    entry_raw = limit_px
                                    entry_px = apply_slippage(entry_raw, s.slippage_bps, side="buy")
                                    mult = size_multiplier_from_fng(fng_val, "LONG", s.fng_floor_mult) if s.sent_enabled else 1.0
                                    qty = qty_calc * mult
                                    if qty > 0:
                                        position = 1
                                        entry_time = dt.isoformat()
                                        entry_equity = equity
                                        entry_bar_i = i
                                        entry_mult = mult
                                        entry_type = "LIMIT"
                                        trail_stop = entry_raw - stop_dist
                                        tp_price = (entry_raw + float(s.tp_r_mult) * stop_dist) if s.use_take_profit else None
                                        st["pending_filled"] += 1
                                        st["entry_limit_fills"] += 1
                                        pending = None
                                        continue

                                if side == "SHORT" and s.direction in ("both", "short_only"):
                                    entry_raw = limit_px
                                    entry_px = apply_slippage(entry_raw, s.slippage_bps, side="sell")
                                    mult = size_multiplier_from_fng(fng_val, "SHORT", s.fng_floor_mult) if s.sent_enabled else 1.0
                                    qty = qty_calc * mult
                                    if qty > 0:
                                        position = -1
                                        entry_time = dt.isoformat()
                                        entry_equity = equity
                                        entry_bar_i = i
                                        entry_mult = mult
                                        entry_type = "LIMIT"
                                        trail_stop = entry_raw + stop_dist
                                        tp_price = (entry_raw - float(s.tp_r_mult) * stop_dist) if s.use_take_profit else None
                                        st["pending_filled"] += 1
                                        st["entry_limit_fills"] += 1
                                        pending = None
                                        continue
                        else:
                            st["pending_touch_rejected_by_close"] += 1

        # Place pending
        if pending is None and s.use_pullback_entry:
            if s.direction in ("both", "long_only") and bool(long_signal.iloc[i]):
                if s.pullback_level == "mid":
                    limit_px = float((df.iloc[i][dh_col] + df.iloc[i][dl_col]) / 2.0)
                else:
                    limit_px = float(df.iloc[i][dh_col])

                pending = {
                    "side": "LONG",
                    "limit": limit_px,
                    "start_i": i + 1,
                    "expire_i": i + int(s.pullback_max_wait_bars),
                    "fallback_i": i + int(s.pullback_fallback_after_bars) + 1,
                }
                st["pending_created"] += 1

            elif s.direction in ("both", "short_only") and bool(short_signal.iloc[i]):
                if s.pullback_level == "mid":
                    limit_px = float((df.iloc[i][dh_col] + df.iloc[i][dl_col]) / 2.0)
                else:
                    limit_px = float(df.iloc[i][dl_col])

                pending = {
                    "side": "SHORT",
                    "limit": limit_px,
                    "start_i": i + 1,
                    "expire_i": i + int(s.pullback_max_wait_bars),
                    "fallback_i": i + int(s.pullback_fallback_after_bars) + 1,
                }
                st["pending_created"] += 1

    # force close slice end
    if force_close_end and position != 0:
        last = df.iloc[-1]
        dt = last["datetime_utc"]
        exit_raw = float(last["close"])
        exit_px = apply_slippage(exit_raw, s.slippage_bps, side=("sell" if position == 1 else "buy"))
        not_entry = qty * float(entry_px)
        not_exit = qty * float(exit_px)
        if position == 1:
            pnl = (qty * (exit_px - entry_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
            side_str = "LONG"
        else:
            pnl = (qty * (entry_px - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
            side_str = "SHORT"
        equity += pnl
        trades_rows.append({
            "side": side_str,
            "entry_time": entry_time,
            "entry_price": float(entry_px),
            "entry_type": entry_type,
            "exit_time": dt.isoformat(),
            "exit_price": float(exit_px),
            "exit_reason": "EOD",
            "qty": float(qty),
            "size_mult": float(entry_mult),
            "pnl": float(pnl),
            "equity_before": float(entry_equity),
            "equity_after": float(equity),
            "bars_held": int((n - 1) - entry_bar_i),
        })

    last = df.iloc[-1]
    equity_rows.append({"timestamp": int(last["timestamp"]), "datetime_utc": last["datetime_utc"], "equity": equity})

    trades_df = pd.DataFrame(trades_rows)
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(trades_df, equity_df)
    return trades_df, equity_df, metrics, st


# ----------------------------
# Walk-forward
# ----------------------------
def build_windows(df: pd.DataFrame, wf: WalkForwardSettings) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dt_min = to_utc_ts(df["datetime_utc"].min())
    dt_max = to_utc_ts(df["datetime_utc"].max())

    start = dt_min + pd.Timedelta(days=int(wf.train_days))
    windows = []
    while start < dt_max:
        end = start + pd.Timedelta(days=int(wf.test_days))
        windows.append((start, end))
        start = end  # rolling forward by test_days
    return windows


def window_mask(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = to_utc_ts(start)
    e = to_utc_ts(end)
    return (df["datetime_utc"] >= s) & (df["datetime_utc"] < e)


def profit_factor_from_trades(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    wins = trades[trades["pnl"] > 0]["pnl"].sum()
    losses = -trades[trades["pnl"] < 0]["pnl"].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else np.nan
    return float(wins / losses)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    s, wf, cfg = resolve_settings()
    df = load_features_df(s)

    run_dir = make_run_dir(OUT_WF_DIR)
    windows = build_windows(df, wf)

    equity = float(s.initial_equity)

    rows = []
    all_trades = []

    active_windows = 0
    win_windows = 0
    dd_list = []
    sharpe_list = []
    winrate_list = []
    ret_list_active = []
    pf_list = []

    for wi, (test_start, test_end) in enumerate(windows):
        mask = window_mask(df, test_start, test_end)
        df_w = df.loc[mask].copy()
        bars = int(len(df_w))
        if bars < int(wf.min_test_bars):
            break

        eq_start = float(equity)
        trades_w, equity_w, m_w, st = run_engine(df_w, s, equity_start=equity, force_close_end=True)
        eq_end = float(m_w["final_equity"])
        equity = eq_end

        tr_count = int(len(trades_w))
        if tr_count > 0:
            active_windows += 1
            ret_w = (eq_end / eq_start) - 1.0
            ret_list_active.append(float(ret_w))
            if eq_end > eq_start:
                win_windows += 1

        pf_w = profit_factor_from_trades(trades_w)
        pf_list.append(pf_w)

        all_trades.append(trades_w)

        rows.append({
            "window": wi,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "bars": bars,
            "trades": tr_count,
            "profit_factor": float(pf_w) if np.isfinite(pf_w) else (np.nan if np.isnan(pf_w) else float("inf")),
            "sharpe": float(m_w.get("sharpe_approx", 0.0)),
            "max_drawdown": float(m_w.get("max_drawdown", 0.0)),
            "win_rate": float(m_w.get("win_rate", 0.0)),
            "equity_start": eq_start,
            "equity_end": eq_end,
        })

        # store for robust stats (only where meaningful)
        if tr_count > 0:
            dd_list.append(float(m_w.get("max_drawdown", 0.0)))
            sharpe_list.append(float(m_w.get("sharpe_approx", 0.0)))
            winrate_list.append(float(m_w.get("win_rate", 0.0)))

    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(run_dir / "walkforward_windows.csv", index=False)

    trades_all = pd.concat(all_trades, ignore_index=True) if len(all_trades) else pd.DataFrame([])
    pf_total = profit_factor_from_trades(trades_all)

    # PF median variants
    pf_finite = [x for x in pf_list if (x is not None and np.isfinite(x))]
    pf_nonzero = [x for x in pf_finite if x > 0]

    summary = {
        "run_id": run_dir.name,
        "train_days": int(wf.train_days),
        "test_days": int(wf.test_days),
        "min_test_bars": int(wf.min_test_bars),
        "windows": int(len(wf_df)),
        "active_windows": int(active_windows),
        "active_windows_pct": float(active_windows / max(1, int(len(wf_df)))),
        "win_windows": int(win_windows),
        "trades_total": int(len(trades_all)) if not trades_all.empty else 0,
        "equity_start": float(s.initial_equity),
        "equity_end": float(equity),
        "return_total_pct": float((equity / float(s.initial_equity)) - 1.0),
        "profit_factor_total": float(pf_total) if np.isfinite(pf_total) else (np.nan if np.isnan(pf_total) else float("inf")),
        "profit_factor_median_defined": float(np.median(pf_finite)) if len(pf_finite) else np.nan,
        "profit_factor_median_nonzero": float(np.median(pf_nonzero)) if len(pf_nonzero) else np.nan,
        "median_return_active": float(np.median(ret_list_active)) if len(ret_list_active) else np.nan,
        "sharpe_mean_active": float(np.mean(sharpe_list)) if len(sharpe_list) else np.nan,
        "dd_mean_active": float(np.mean(dd_list)) if len(dd_list) else np.nan,
        "winrate_mean_active": float(np.mean(winrate_list)) if len(winrate_list) else np.nan,
        "config_used": {
            "adx_min_4h": float(s.adx_min_4h),
            "adx_period": int(s.adx_period),
            "donchian_n": int(s.donchian_n),
            "atr_period": int(s.atr_period),
            "atr_trail_mult": float(s.atr_trail_mult),
            "min_atr_ratio": float(s.min_atr_ratio),
            "atr_exp_mult": float(s.atr_exp_mult),
            "breakout_buffer": float(s.breakout_buffer),
            "pullback_max_wait_bars": int(s.pullback_max_wait_bars),
            "pullback_level": str(s.pullback_level),
            "pullback_fallback_market": bool(s.pullback_fallback_market),
            "pullback_fallback_after_bars": int(s.pullback_fallback_after_bars),
            "confirm_buffer": float(s.confirm_buffer),
            "sent_enabled": bool(s.sent_enabled),
            "fng_floor_mult": float(s.fng_floor_mult),
            "fee_bps": float(s.fee_bps),
            "slippage_bps": float(s.slippage_bps),
            "direction": str(s.direction),
            "use_time_stop": bool(s.use_time_stop),
            "time_stop_bars": int(s.time_stop_bars),
            "time_stop_only_if_pnl_leq_zero": bool(s.time_stop_only_if_pnl_leq_zero),
            "use_take_profit": bool(s.use_take_profit),
            "tp_r_mult": float(s.tp_r_mult),
        },
        "note": "Walk-forward corre solo sobre ventanas TEST y encadena equity entre ventanas. Motor igual a 02_backtest.py v4.9 (pullback channel + close-confirm fill)."
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[INFO] Walk-forward OK (paridad con 02 v4.9 pullback channel + close-confirm fill)")
    print(f"  out_dir: {run_dir}")
    print(f"  windows: {summary['windows']}")
    print(f"  active_windows: {summary['active_windows']} ({summary['active_windows_pct']*100:.1f}%)")
    print(f"  win_windows: {summary['win_windows']} ({summary['win_windows']/max(1,summary['windows'])*100:.1f}%)")
    print(f"  trades_total: {summary['trades_total']}")
    print(f"  equity_start: {summary['equity_start']}")
    print(f"  equity_end  : {summary['equity_end']}")
    print(f"  return_total_pct: {summary['return_total_pct']}")
    print(f"  profit_factor_total: {summary['profit_factor_total']}")
    print(f"  profit_factor_median_defined: {summary['profit_factor_median_defined']}")
    print(f"  profit_factor_median_nonzero: {summary['profit_factor_median_nonzero']}")
    print(f"  median_return_active: {summary['median_return_active']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

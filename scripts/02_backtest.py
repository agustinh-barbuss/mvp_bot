"""
02_backtest.py (v4.9)
Pullback-to-Channel LIMIT entry + CLOSE-CONFIRM fill (no lookahead)
+ ATR trailing stop + TP (R-multiple) + TIME STOP
+ HTF regime + ATR expansion + ADX 4H gate + FNG size modulator
+ optional fallback MARKET on expiration (disabled by default)

Logic summary:
- Signal on CLOSE of bar i (breakout)
- Create pending LIMIT at channel level (donchian high/low from signal bar)
- Pending active from i+1 ... i+pullback_max_wait_bars
- Fill happens only if:
    1) price touches LIMIT within bar range AND
    2) bar CLOSE confirms breakout beyond (breakout_buffer + confirm_buffer)
- Exits:
    - ATR trailing stop (bar range)
    - Take-profit at entry_raw +/- tp_r_mult * (atr_trail_mult*ATR_at_entry_bar)
    - Time-stop at next bar OPEN after N bars held (optional, and optional only if pnl<=0)
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
OUT_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
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
    """
    fng in [0..100], neutral at 50.
    - For LONG: reduce size when fng>50 (greed), down to floor at 100
    - For SHORT: reduce size when fng<50 (fear), down to floor at 0
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

    if "size_mult" in trades.columns and not trades.empty:
        out["avg_size_mult"] = float(trades["size_mult"].mean())

    # entry/exit breakdown (optional)
    if "entry_type" in trades.columns:
        out["entry_limit_fills"] = int((trades["entry_type"] == "LIMIT").sum())
        out["entry_fallback_market"] = int((trades["entry_type"] == "FALLBACK_MKT").sum())

    if "exit_reason" in trades.columns:
        out["exit_time_stop"] = int((trades["exit_reason"] == "TIME").sum())
        out["exit_take_profit"] = int((trades["exit_reason"] == "TP").sum())

    return out


def make_run_dir(base_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    direction: str  # both / long_only / short_only

    # sentiment
    sent_enabled: bool
    fng_floor_mult: float

    # pullback
    use_pullback_entry: bool
    pullback_max_wait_bars: int
    pullback_level: str  # "channel" or "mid"
    pullback_fallback_market: bool
    pullback_fallback_after_bars: int

    # close-confirm
    confirm_buffer: float

    # exits
    use_time_stop: bool
    time_stop_bars: int
    time_stop_only_if_pnl_leq_zero: bool

    use_take_profit: bool
    tp_r_mult: float


def resolve_settings() -> BacktestSettings:
    cfg = load_yaml_config()
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

    # Merge FNG
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
# Core engine (shared by WF too)
# ----------------------------
def run_engine(
    df: pd.DataFrame,
    s: BacktestSettings,
    equity_start: float,
    force_close_end: bool = True,
    debug: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Runs the exact execution logic over df slice.
    Returns: trades_df, equity_df, metrics, debug_stats
    """

    atr_col = f"atr_{s.atr_period}"
    atr_sma_col = f"atr_sma_{s.atr_period}"
    dh_col = f"donchian_high_{s.donchian_n}"
    dl_col = f"donchian_low_{s.donchian_n}"
    adx_col = f"adx_4h_{s.adx_period}"

    required = {"timestamp", "open", "high", "low", "close", "regime_htf", atr_col, atr_sma_col, dh_col, dl_col, adx_col, "fng_value", "datetime_utc"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas requeridas: {missing}")

    # Cast
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

    # State
    equity = float(equity_start)

    position = 0        # 1 long, -1 short
    qty = 0.0
    entry_px = 0.0      # executed px (with slippage)
    entry_raw = 0.0     # raw level (for stops/TP)
    entry_time = None
    entry_equity = None
    entry_bar_i = None
    entry_mult = 1.0
    entry_type = "MKT"

    trail_stop = None   # raw stop
    tp_price = None     # raw tp

    pending: Optional[Dict[str, Any]] = None
    # schema:
    # {
    #   "side": "LONG"/"SHORT",
    #   "limit": float,       # fixed at signal time
    #   "start_i": int,
    #   "expire_i": int,
    #   "fallback_i": int
    # }

    # Stats
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

    # iterate up to n-2 to allow next_open for time-stop
    for i in range(1, n - 1):
        row = df.iloc[i]
        ts = int(row["timestamp"])
        dt = row["datetime_utc"]
        equity_rows.append({"timestamp": ts, "datetime_utc": dt, "equity": equity})

        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # ---------------- EXIT (if in position) ----------------
        if position != 0:
            bars_held = int(i - entry_bar_i)

            # update trailing stop based on CLOSE
            if position == 1:
                new_stop = float(row["close"]) - float(s.atr_trail_mult) * atr_i
                trail_stop = max(trail_stop, new_stop)
            else:
                new_stop = float(row["close"]) + float(s.atr_trail_mult) * atr_i
                trail_stop = min(trail_stop, new_stop)

            # TIME STOP (exit at next bar OPEN)
            if s.use_time_stop and bars_held >= int(s.time_stop_bars):
                if (i + 1) < n:
                    # apply only if pnl <= 0, optional
                    if s.time_stop_only_if_pnl_leq_zero:
                        if position == 1:
                            unreal = (float(row["close"]) - float(entry_px)) * qty
                        else:
                            unreal = (float(entry_px) - float(row["close"])) * qty
                        if unreal > 0:
                            # skip time-stop if in profit
                            pass
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

                            # flat
                            position = 0
                            qty = 0.0
                            trail_stop = None
                            tp_price = None
                            entry_mult = 1.0
                            entry_type = "MKT"
                            pending = None
                            continue
                    else:
                        # unconditional time-stop
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

            # STOP vs TP (conservador: stop primero si ambos posibles)
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

            else:  # short
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

            # still in position
            continue

        # ---------------- Flat: handle pending ----------------
        if pending is not None:
            # Not active yet
            if i < int(pending["start_i"]):
                pass
            else:
                # expire
                if i > int(pending["expire_i"]):
                    st["pending_expired"] += 1
                    pending = None
                else:
                    side = str(pending["side"])
                    limit_px = float(pending["limit"])

                    # touch
                    touched = (float(row["low"]) <= limit_px <= float(row["high"]))
                    if touched:
                        # close-confirm against breakout level (signal-level limit)
                        cb = float(s.confirm_buffer)
                        if side == "LONG":
                            confirm_ok = float(row["close"]) > (limit_px * (1.0 + float(s.breakout_buffer) + cb))
                        else:
                            confirm_ok = float(row["close"]) < (limit_px * (1.0 - float(s.breakout_buffer) - cb))

                        if confirm_ok:
                            # size
                            stop_dist = float(s.atr_trail_mult) * atr_i
                            if stop_dist <= 0:
                                pending = None
                            else:
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

                    # optional fallback market after N bars (if enabled)
                    if pending is not None and s.pullback_fallback_market and i >= int(pending["fallback_i"]):
                        # Only if gates still pass
                        if bool(ok_vol.iloc[i]) and bool(exp_ok.iloc[i]) and bool(adx_ok.iloc[i]):
                            side = str(pending["side"])
                            # still valid breakout at current OPEN
                            dh_now = float(df.iloc[i][dh_col])
                            dl_now = float(df.iloc[i][dl_col])
                            if side == "LONG":
                                still_valid = float(row["open"]) > dh_now * (1.0 + float(s.breakout_buffer))
                            else:
                                still_valid = float(row["open"]) < dl_now * (1.0 - float(s.breakout_buffer))
                            if still_valid:
                                stop_dist = float(s.atr_trail_mult) * atr_i
                                if stop_dist > 0:
                                    risk_amount = equity * float(s.risk_per_trade)
                                    qty_calc = risk_amount / stop_dist
                                    fng_val = float(row["fng_value"])

                                    if side == "LONG" and s.direction in ("both", "long_only"):
                                        entry_raw = float(row["open"])
                                        entry_px = apply_slippage(entry_raw, s.slippage_bps, side="buy")
                                        mult = size_multiplier_from_fng(fng_val, "LONG", s.fng_floor_mult) if s.sent_enabled else 1.0
                                        qty = qty_calc * mult
                                        if qty > 0:
                                            position = 1
                                            entry_time = dt.isoformat()
                                            entry_equity = equity
                                            entry_bar_i = i
                                            entry_mult = mult
                                            entry_type = "FALLBACK_MKT"
                                            trail_stop = entry_raw - stop_dist
                                            tp_price = (entry_raw + float(s.tp_r_mult) * stop_dist) if s.use_take_profit else None
                                            st["entry_fallback_market"] += 1
                                            pending = None
                                            continue

                                    if side == "SHORT" and s.direction in ("both", "short_only"):
                                        entry_raw = float(row["open"])
                                        entry_px = apply_slippage(entry_raw, s.slippage_bps, side="sell")
                                        mult = size_multiplier_from_fng(fng_val, "SHORT", s.fng_floor_mult) if s.sent_enabled else 1.0
                                        qty = qty_calc * mult
                                        if qty > 0:
                                            position = -1
                                            entry_time = dt.isoformat()
                                            entry_equity = equity
                                            entry_bar_i = i
                                            entry_mult = mult
                                            entry_type = "FALLBACK_MKT"
                                            trail_stop = entry_raw + stop_dist
                                            tp_price = (entry_raw - float(s.tp_r_mult) * stop_dist) if s.use_take_profit else None
                                            st["entry_fallback_market"] += 1
                                            pending = None
                                            continue

                        # if fallback can't enter, cancel to avoid stale
                        st["pending_canceled_invalid"] += 1
                        pending = None

        # ---------------- Place pending on signal ----------------
        if pending is None and s.use_pullback_entry:
            if s.direction in ("both", "long_only") and bool(long_signal.iloc[i]):
                # limit fixed at signal time
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

    # force close at end (window or full backtest)
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

    # finalize equity row
    last = df.iloc[-1]
    equity_rows.append({"timestamp": int(last["timestamp"]), "datetime_utc": last["datetime_utc"], "equity": equity})

    trades_df = pd.DataFrame(trades_rows)
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(trades_df, equity_df)

    # attach some useful stats to metrics
    metrics.update({
        "avg_size_mult": float(trades_df["size_mult"].mean()) if (not trades_df.empty and "size_mult" in trades_df.columns) else None,
        "entry_limit_fills": st["entry_limit_fills"],
        "entry_fallback_market": st["entry_fallback_market"],
        "exit_time_stop": st["exit_time_stop"],
        "exit_take_profit": st["exit_take_profit"],
        "pending_created": st["pending_created"],
        "pending_filled": st["pending_filled"],
        "pending_expired": st["pending_expired"],
        "pending_touch_rejected_by_close": st["pending_touch_rejected_by_close"],
    })

    # Debug prints (optional)
    dbg = {}
    if debug:
        dbg = {
            "gates_pass_rate": {
                "ok_vol": float(ok_vol.mean()),
                "exp_ok": float(exp_ok.mean()),
                "adx_ok": float(adx_ok.mean()),
            },
            "signal_counts": {
                "long_signal_bars": int(long_signal.sum()),
                "short_signal_bars": int(short_signal.sum()),
            },
            "pending_stats": dict(st),
        }

    return trades_df, equity_df, metrics, dbg


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = resolve_settings()

    df = load_features_df(s)

    trades, eq, metrics, dbg = run_engine(
        df=df.copy(),
        s=s,
        equity_start=float(s.initial_equity),
        force_close_end=True,
        debug=True,
    )

    run_dir = make_run_dir(OUT_RUNS_DIR)
    (run_dir / "config_used.json").write_text(json.dumps(s.__dict__, indent=2), encoding="utf-8")
    trades.to_csv(run_dir / "trades.csv", index=False)
    eq.to_csv(run_dir / "equity.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "debug.json").write_text(json.dumps(dbg, indent=2), encoding="utf-8")

    # console
    print("[INFO] Backtest OK (v4.9 pullback channel + close-confirm fill)")
    print(f"  run_dir : {run_dir}")
    print(f"  trades  : {len(trades)}")

    # show some debug
    if dbg:
        gp = dbg["gates_pass_rate"]
        print("[DEBUG] Gates pass-rate:")
        print(f"  ok_vol      : {gp['ok_vol']*100:.2f}%")
        print(f"  exp_ok      : {gp['exp_ok']*100:.2f}%")
        print(f"  adx_ok      : {gp['adx_ok']*100:.2f}% (min={s.adx_min_4h})")

        sc = dbg["signal_counts"]
        print("[DEBUG] Signal counts:")
        print(f"  long_signal bars : {sc['long_signal_bars']}")
        print(f"  short_signal bars: {sc['short_signal_bars']}")

        ps = {
            "pending_created": metrics.get("pending_created"),
            "pending_filled": metrics.get("pending_filled"),
            "pending_expired": metrics.get("pending_expired"),
            "pending_touch_rejected_by_close": metrics.get("pending_touch_rejected_by_close"),
        }
        print("[DEBUG] Pending stats:")
        for k, v in ps.items():
            print(f"  {k:28s}: {v}")

    for k, v in metrics.items():
        if k in ("pending_created", "pending_filled", "pending_expired", "pending_touch_rejected_by_close"):
            continue
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

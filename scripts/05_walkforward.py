"""
05_walkforward.py (paridad con 02_backtest.py v4.4.1)

Walk-forward rolling:
- train_days: NO entrena nada, solo define el inicio de la primera ventana TEST
- test_days: corre backtest SOLO en ventana TEST y encadena equity entre ventanas
- min_test_bars: si una ventana tiene menos barras, se corta

Motor idéntico a 02 v4.4.1:
- Pullback-to-Channel LIMIT entry
- ATR trailing stop
- TIME STOP (loss-only) a OPEN usando prev CLOSE
- TAKE PROFIT (R-multiple) intrabar usando high/low
- Pending se cancela solo por invalidación real (gates/regime/opposite break)

Outputs:
- outputs/walkforward/<run_id>/summary.json
- outputs/walkforward/<run_id>/windows.csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_WF_DIR = PROJECT_ROOT / "outputs" / "walkforward"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FNG_PATH = EXTERNAL_DIR / "fng.csv"


# ---------------------------- helpers ----------------------------

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

    if "size_mult" in trades.columns:
        out["avg_size_mult"] = float(trades["size_mult"].mean())

    if "entry_type" in trades.columns:
        out["entry_limit_fills"] = int((trades["entry_type"] == "LIMIT").sum())
        out["entry_fallback_market"] = int((trades["entry_type"] == "FALLBACK_MKT").sum())

    if "exit_reason" in trades.columns:
        out["exit_time_stop"] = int((trades["exit_reason"] == "TIME_STOP").sum())
        out["exit_take_profit"] = int((trades["exit_reason"] == "TAKE_PROFIT").sum())

    return out


def build_features_path(exchange: str, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
    sym = safe_symbol_for_filename(symbol)
    return PROCESSED_DIR / f"{exchange}_{sym}_{timeframe}_{start_date}_to_{end_date}_features.csv"


def make_out_dir() -> Tuple[Path, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_WF_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out, run_id


# ---------------------------- settings ----------------------------

@dataclass
class WalkForwardSettings:
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
    pullback_fallback_market: bool
    pullback_fallback_after_bars: int

    use_time_stop: bool
    time_stop_bars: int
    time_stop_only_if_pnl_leq_zero: bool

    use_take_profit: bool
    tp_r_mult: float

    train_days: int
    test_days: int
    min_test_bars: int


def resolve_settings() -> WalkForwardSettings:
    cfg = load_yaml_config(CONFIG_PATH)
    market = (cfg.get("market") or {})
    strat = (cfg.get("strategy") or {})
    costs = (cfg.get("costs") or {})
    bt = (cfg.get("backtest") or {})
    sent = (cfg.get("sentiment") or {})
    wf = (cfg.get("walkforward") or {})
    validation = (cfg.get("validation") or {})

    exchange = get_env("EXCHANGE", market.get("exchange", "binance"))
    symbol = get_env("SYMBOL", market.get("symbol", "BTC/USDT"))
    timeframe = get_env("TIMEFRAME", market.get("timeframe", "1h"))
    start_date = get_env("START_DATE", market.get("start_date", "2019-01-01"))

    end_date = validation.get("test_end", None)
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_date = get_env("END_DATE", end_date)

    atr_period = int(strat.get("atr_period", 14))
    donchian_n = parse_int(get_env("DONCHIAN_N", None), int(strat.get("donchian_n", 20)))
    atr_trail_mult = parse_float(get_env("ATR_TRAIL_MULT", None), float(strat.get("atr_trail_mult", 3.0)))
    min_atr_ratio = parse_float(get_env("MIN_ATR_RATIO", None), float(strat.get("min_atr_ratio", 0.0015)))
    atr_exp_mult = parse_float(get_env("ATR_EXP_MULT", None), float(strat.get("atr_exp_mult", 1.05)))
    breakout_buffer = parse_float(get_env("BREAKOUT_BUFFER", None), float(strat.get("breakout_buffer", 0.0010)))

    adx_period = int(strat.get("adx_period", 14))
    adx_min_4h = parse_float(get_env("ADX_MIN_4H", None), float(strat.get("adx_min_4h", 30)))

    fee_bps = parse_float(get_env("FEE_BPS", None), float(costs.get("fee_bps", 7.5)))
    slippage_bps = parse_float(get_env("SLIPPAGE_BPS", None), float(costs.get("slippage_bps", 2.0)))

    initial_equity = parse_float(get_env("INITIAL_EQUITY", None), float(bt.get("initial_equity", 10_000.0)))
    risk_per_trade = parse_float(get_env("RISK_PER_TRADE", None), float(bt.get("risk_per_trade", 0.01)))
    direction = get_env("DIRECTION", str(bt.get("direction", "both")))

    sent_enabled = bool(sent.get("enabled", True))
    fng_floor_mult = parse_float(get_env("FNG_FLOOR_MULT", None), float(sent.get("fng_floor_mult", 0.10)))

    use_pullback_entry = bool(strat.get("use_pullback_entry", True))
    pullback_max_wait_bars = parse_int(get_env("PULLBACK_MAX_WAIT_BARS", None), int(strat.get("pullback_max_wait_bars", 12)))
    pullback_fallback_market = bool(strat.get("pullback_fallback_market", False))
    pullback_fallback_after_bars = parse_int(get_env("PULLBACK_FALLBACK_AFTER_BARS", None), int(strat.get("pullback_fallback_after_bars", 6)))

    use_time_stop = bool(strat.get("use_time_stop", True))
    time_stop_bars = parse_int(get_env("TIME_STOP_BARS", None), int(strat.get("time_stop_bars", 72)))
    time_stop_only_if_pnl_leq_zero = bool(strat.get("time_stop_only_if_pnl_leq_zero", True))

    use_take_profit = bool(strat.get("use_take_profit", True))
    tp_r_mult = parse_float(get_env("TP_R_MULT", None), float(strat.get("tp_r_mult", 2.0)))

    train_days = parse_int(get_env("WF_TRAIN_DAYS", None), int(wf.get("train_days", 180)))
    test_days = parse_int(get_env("WF_TEST_DAYS", None), int(wf.get("test_days", 90)))
    min_test_bars = parse_int(get_env("WF_MIN_TEST_BARS", None), int(wf.get("min_test_bars", 300)))

    return WalkForwardSettings(
        exchange=exchange, symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
        atr_period=atr_period, donchian_n=donchian_n, atr_trail_mult=atr_trail_mult, min_atr_ratio=min_atr_ratio,
        atr_exp_mult=atr_exp_mult, breakout_buffer=breakout_buffer,
        adx_period=adx_period, adx_min_4h=adx_min_4h,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        initial_equity=initial_equity, risk_per_trade=risk_per_trade, direction=direction,
        sent_enabled=sent_enabled, fng_floor_mult=fng_floor_mult,
        use_pullback_entry=use_pullback_entry, pullback_max_wait_bars=pullback_max_wait_bars,
        pullback_fallback_market=pullback_fallback_market, pullback_fallback_after_bars=pullback_fallback_after_bars,
        use_time_stop=use_time_stop, time_stop_bars=time_stop_bars, time_stop_only_if_pnl_leq_zero=time_stop_only_if_pnl_leq_zero,
        use_take_profit=use_take_profit, tp_r_mult=tp_r_mult,
        train_days=train_days, test_days=test_days, min_test_bars=min_test_bars,
    )


# ---------------------------- core engine (same as 02) ----------------------------

def run_backtest_slice(
    df: pd.DataFrame,
    s: WalkForwardSettings,
    atr_col: str,
    atr_sma_col: str,
    dh_col: str,
    dl_col: str,
    adx_col: str,
    equity_start: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    atr = df[atr_col]
    atr_sma = df[atr_sma_col]
    close = df["close"]

    ok_vol = (atr / close) >= s.min_atr_ratio
    exp_ok = atr > (atr_sma * s.atr_exp_mult)
    regime = df["regime_htf"].astype(int)
    adx_ok = df[adx_col] >= float(s.adx_min_4h)

    buffer = s.breakout_buffer
    long_signal = ok_vol & exp_ok & adx_ok & (regime == 1) & (close > df[dh_col] * (1.0 + buffer))
    short_signal = ok_vol & exp_ok & adx_ok & (regime == -1) & (close < df[dl_col] * (1.0 - buffer))

    equity = float(equity_start)

    position = 0
    qty = 0.0
    entry_price = 0.0
    entry_time: Optional[str] = None
    trail_stop: Optional[float] = None
    entry_equity: Optional[float] = None
    entry_bar_i: Optional[int] = None
    entry_mult = 1.0
    entry_type = "MKT"
    tp_price: Optional[float] = None

    pending: Optional[Dict[str, Any]] = None

    trades_rows: List[dict] = []
    equity_rows: List[dict] = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        ts = int(row["timestamp"])
        dt = row["datetime_utc"]
        equity_rows.append({"timestamp": ts, "datetime_utc": dt, "equity": equity})

        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        prev_close = float(df.iloc[i - 1]["close"])
        open_i = float(row["open"])
        high_i = float(row["high"])
        low_i = float(row["low"])

        # EXIT priority: TIME_STOP, TAKE_PROFIT, TRAIL
        if position == 1:
            if s.use_time_stop and entry_bar_i is not None:
                bars_held = i - int(entry_bar_i)
                if bars_held >= int(s.time_stop_bars):
                    cond = True
                    if s.time_stop_only_if_pnl_leq_zero:
                        cond = (prev_close <= float(entry_price))
                    if cond:
                        exit_raw = open_i
                        exit_px = apply_slippage(exit_raw, s.slippage_bps, side="sell")
                        not_entry = qty * entry_price
                        not_exit = qty * exit_px
                        pnl = (qty * (exit_px - entry_price)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                        equity += pnl
                        trades_rows.append({
                            "side": "LONG","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                            "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TIME_STOP",
                            "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                            "bars_held": int(bars_held),
                        })
                        position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None
                        continue

            if s.use_take_profit and tp_price is not None and high_i >= float(tp_price):
                fill_raw = max(float(tp_price), open_i)
                exit_px = apply_slippage(fill_raw, s.slippage_bps, side="sell")
                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (exit_px - entry_price)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl
                trades_rows.append({
                    "side": "LONG","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                    "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TAKE_PROFIT",
                    "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                    "bars_held": int(i - int(entry_bar_i)),
                })
                position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None
                continue

            new_stop = float(row["close"]) - s.atr_trail_mult * atr_i
            trail_stop = max(float(trail_stop), new_stop) if trail_stop is not None else new_stop
            if low_i <= float(trail_stop):
                exit_raw = float(trail_stop)
                exit_px = apply_slippage(exit_raw, s.slippage_bps, side="sell")
                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (exit_px - entry_price)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl
                trades_rows.append({
                    "side": "LONG","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                    "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TRAIL",
                    "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                    "bars_held": int(i - int(entry_bar_i)),
                })
                position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None

        elif position == -1:
            if s.use_time_stop and entry_bar_i is not None:
                bars_held = i - int(entry_bar_i)
                if bars_held >= int(s.time_stop_bars):
                    cond = True
                    if s.time_stop_only_if_pnl_leq_zero:
                        cond = (prev_close >= float(entry_price))
                    if cond:
                        exit_raw = open_i
                        exit_px = apply_slippage(exit_raw, s.slippage_bps, side="buy")
                        not_entry = qty * entry_price
                        not_exit = qty * exit_px
                        pnl = (qty * (entry_price - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                        equity += pnl
                        trades_rows.append({
                            "side": "SHORT","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                            "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TIME_STOP",
                            "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                            "bars_held": int(bars_held),
                        })
                        position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None
                        continue

            if s.use_take_profit and tp_price is not None and low_i <= float(tp_price):
                fill_raw = min(float(tp_price), open_i)
                exit_px = apply_slippage(fill_raw, s.slippage_bps, side="buy")
                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (entry_price - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl
                trades_rows.append({
                    "side": "SHORT","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                    "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TAKE_PROFIT",
                    "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                    "bars_held": int(i - int(entry_bar_i)),
                })
                position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None
                continue

            new_stop = float(row["close"]) + s.atr_trail_mult * atr_i
            trail_stop = min(float(trail_stop), new_stop) if trail_stop is not None else new_stop
            if high_i >= float(trail_stop):
                exit_raw = float(trail_stop)
                exit_px = apply_slippage(exit_raw, s.slippage_bps, side="buy")
                not_entry = qty * entry_price
                not_exit = qty * exit_px
                pnl = (qty * (entry_price - exit_px)) - fee_cost(not_entry, s.fee_bps) - fee_cost(not_exit, s.fee_bps)
                equity += pnl
                trades_rows.append({
                    "side": "SHORT","entry_time": entry_time,"entry_price": entry_price,"entry_type": entry_type,
                    "exit_time": dt.isoformat(),"exit_price": exit_px,"exit_reason": "TRAIL",
                    "qty": qty,"size_mult": entry_mult,"pnl": pnl,"equity_before": entry_equity,"equity_after": equity,
                    "bars_held": int(i - int(entry_bar_i)),
                })
                position = 0; qty = 0.0; trail_stop = None; entry_mult = 1.0; entry_type = "MKT"; tp_price = None

        if position != 0:
            continue

        # Pending handling (invalidación real)
        if pending is not None:
            side_p = str(pending["side"])
            dh_now = float(df.iloc[i][dh_col])
            dl_now = float(df.iloc[i][dl_col])

            gates_ok = bool(ok_vol.iloc[i]) and bool(exp_ok.iloc[i]) and bool(adx_ok.iloc[i])
            reg_ok = (int(regime.iloc[i]) == (1 if side_p == "LONG" else -1))

            if side_p == "LONG":
                opp_break = (float(row["close"]) < dl_now * (1.0 - s.breakout_buffer))
            else:
                opp_break = (float(row["close"]) > dh_now * (1.0 + s.breakout_buffer))

            if (not gates_ok) or (not reg_ok) or opp_break:
                pending = None
            else:
                if i >= int(pending["start_i"]):
                    limit_px = float(pending["limit"])
                    stop_dist = s.atr_trail_mult * atr_i
                    if stop_dist > 0:
                        risk_amount = equity * s.risk_per_trade
                        qty_calc = risk_amount / stop_dist
                        fng_val = float(row["fng_value"])

                        if side_p == "LONG" and s.direction in ("both", "long_only") and low_i <= limit_px:
                            fill_raw = min(limit_px, open_i)
                            entry_px = apply_slippage(fill_raw, s.slippage_bps, side="buy")
                            mult = size_multiplier_from_fng(fng_val, "LONG", s.fng_floor_mult) if s.sent_enabled else 1.0
                            qty = qty_calc * mult
                            if qty > 0:
                                position = 1
                                entry_price = entry_px
                                entry_time = dt.isoformat()
                                entry_equity = equity
                                entry_bar_i = i
                                entry_mult = mult
                                entry_type = "LIMIT"
                                trail_stop = fill_raw - stop_dist
                                tp_price = (fill_raw + s.tp_r_mult * stop_dist) if s.use_take_profit else None
                                pending = None
                                continue

                        if side_p == "SHORT" and s.direction == "both" and high_i >= limit_px:
                            fill_raw = max(limit_px, open_i)
                            entry_px = apply_slippage(fill_raw, s.slippage_bps, side="sell")
                            mult = size_multiplier_from_fng(fng_val, "SHORT", s.fng_floor_mult) if s.sent_enabled else 1.0
                            qty = qty_calc * mult
                            if qty > 0:
                                position = -1
                                entry_price = entry_px
                                entry_time = dt.isoformat()
                                entry_equity = equity
                                entry_bar_i = i
                                entry_mult = mult
                                entry_type = "LIMIT"
                                trail_stop = fill_raw + stop_dist
                                tp_price = (fill_raw - s.tp_r_mult * stop_dist) if s.use_take_profit else None
                                pending = None
                                continue

                if i > int(pending["expire_i"]):
                    pending = None

        if position != 0:
            continue

        # Create pending on signal
        if bool(s.use_pullback_entry):
            if s.direction in ("both", "long_only") and bool(long_signal.iloc[i]):
                pending = {
                    "side": "LONG",
                    "limit": float(df.iloc[i][dh_col]),
                    "start_i": i + 1,
                    "expire_i": i + int(s.pullback_max_wait_bars),
                    "fallback_i": i + int(s.pullback_fallback_after_bars) + 1,
                }
            elif s.direction == "both" and bool(short_signal.iloc[i]):
                pending = {
                    "side": "SHORT",
                    "limit": float(df.iloc[i][dl_col]),
                    "start_i": i + 1,
                    "expire_i": i + int(s.pullback_max_wait_bars),
                    "fallback_i": i + int(s.pullback_fallback_after_bars) + 1,
                }

    if len(df) > 0:
        last = df.iloc[-1]
        equity_rows.append({"timestamp": int(last["timestamp"]), "datetime_utc": last["datetime_utc"], "equity": equity})

    trades = pd.DataFrame(trades_rows)
    eq = pd.DataFrame(equity_rows)
    metrics = compute_metrics(trades, eq)
    return trades, eq, metrics


# ---------------------------- main WF ----------------------------

def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = resolve_settings()

    path = build_features_path(s.exchange, s.symbol, s.timeframe, s.start_date, s.end_date)
    if not path.exists():
        print(f"[ERROR] No encuentro el features CSV: {path}")
        return 1

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp_s"] = (df["timestamp"] // 1000).astype(int)

    # Merge FNG
    if s.sent_enabled:
        if not FNG_PATH.exists():
            print(f"[ERROR] Falta {FNG_PATH}. Corré: python scripts\\04_add_sentiment.py")
            return 1

        fng = pd.read_csv(FNG_PATH)
        if "timestamp_s" not in fng.columns or "fng_value" not in fng.columns:
            print("[ERROR] fng.csv inválido (faltan timestamp_s y/o fng_value)")
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
    atr_sma_col = f"atr_sma_{s.atr_period}"
    dh_col = f"donchian_high_{s.donchian_n}"
    dl_col = f"donchian_low_{s.donchian_n}"
    adx_col = f"adx_4h_{s.adx_period}"

    required = {
        "timestamp", "open", "high", "low", "close",
        "regime_htf",
        atr_col, atr_sma_col, dh_col, dl_col,
        adx_col,
        "fng_value",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Faltan columnas requeridas: {missing}")
        return 1

    for c in ["open", "high", "low", "close", atr_col, atr_sma_col, dh_col, dl_col, adx_col, "fng_value"]:
        df[c] = df[c].astype(float)
    df["regime_htf"] = df["regime_htf"].astype(int)

    # Build rolling windows over TEST slices
    start_dt = df["datetime_utc"].iloc[0]
    end_dt = df["datetime_utc"].iloc[-1]

    test_delta = pd.Timedelta(days=int(s.test_days))
    step_delta = test_delta  # rolling non-overlap in test
    warmup_delta = pd.Timedelta(days=int(s.train_days))

    # first test start:
    test_start = start_dt + warmup_delta
    if test_start >= end_dt:
        print("[ERROR] train_days deja sin datos para test.")
        return 1

    out_dir, run_id = make_out_dir()
    windows_rows: List[dict] = []

    equity_chain = float(s.initial_equity)

    pf_list: List[float] = []
    sharpe_list: List[float] = []
    dd_list: List[float] = []
    winrate_list: List[float] = []

    entry_limit_total = 0
    entry_fallback_total = 0

    w = 0
    while True:
        test_end = test_start + test_delta
        if test_start >= end_dt:
            break

        df_test = df[(df["datetime_utc"] >= test_start) & (df["datetime_utc"] < test_end)].copy()
        if len(df_test) < int(s.min_test_bars):
            break

        trades, eq, metrics = run_backtest_slice(
            df_test,
            s,
            atr_col, atr_sma_col, dh_col, dl_col, adx_col,
            equity_chain,
        )

        equity_start = equity_chain
        equity_end = float(metrics.get("final_equity", equity_chain))
        equity_chain = equity_end

        pf = float(metrics.get("profit_factor", 0.0)) if np.isfinite(float(metrics.get("profit_factor", 0.0))) else 0.0
        pf_list.append(pf)
        sharpe_list.append(float(metrics.get("sharpe_approx", 0.0)))
        dd_list.append(float(metrics.get("max_drawdown", 0.0)))
        winrate_list.append(float(metrics.get("win_rate", 0.0)))

        entry_limit_total += int(metrics.get("entry_limit_fills", 0))
        entry_fallback_total += int(metrics.get("entry_fallback_market", 0))

        windows_rows.append({
            "window": w,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "bars": int(len(df_test)),
            "trades": int(metrics.get("trades", 0)),
            "profit_factor": pf,
            "sharpe": float(metrics.get("sharpe_approx", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "equity_start": equity_start,
            "equity_end": equity_end,
        })

        w += 1
        test_start = test_start + step_delta

    windows_df = pd.DataFrame(windows_rows)
    windows_df.to_csv(out_dir / "windows.csv", index=False)

    summary = {
        "run_id": run_id,
        "train_days": int(s.train_days),
        "test_days": int(s.test_days),
        "min_test_bars": int(s.min_test_bars),
        "windows": int(len(windows_rows)),
        "equity_start": float(s.initial_equity),
        "equity_end": float(equity_chain),
        "return_total_pct": float((equity_chain / float(s.initial_equity)) - 1.0),
        "profit_factor_mean": float(np.mean(pf_list)) if pf_list else 0.0,
        "profit_factor_median": float(np.median(pf_list)) if pf_list else 0.0,
        "sharpe_mean": float(np.mean(sharpe_list)) if sharpe_list else 0.0,
        "dd_mean": float(np.mean(dd_list)) if dd_list else 0.0,
        "winrate_mean": float(np.mean(winrate_list)) if winrate_list else 0.0,
        "entry_limit_fills_total": int(entry_limit_total),
        "entry_fallback_market_total": int(entry_fallback_total),
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
            "pullback_fallback_market": bool(s.pullback_fallback_market),
            "pullback_fallback_after_bars": int(s.pullback_fallback_after_bars),
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
        "note": "Walk-forward corre solo sobre ventanas TEST y encadena equity entre ventanas. Motor de ejecución igual a 02_backtest.py v4.4.1.",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[INFO] Walk-forward OK (paridad con 02 v4.4.1)")
    print(f"  out_dir: {out_dir}")
    print(f"  windows: {summary['windows']}")
    print(f"  equity_start: {summary['equity_start']}")
    print(f"  equity_end  : {summary['equity_end']}")
    print(f"  return_total_pct: {summary['return_total_pct']}")
    print(f"  profit_factor_median: {summary['profit_factor_median']}")
    print(f"  entry_limit_fills_total: {summary['entry_limit_fills_total']}")
    print(f"  entry_fallback_market_total: {summary['entry_fallback_market_total']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

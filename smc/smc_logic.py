# smc/smc_logic.py
# =========================
# SMC INTRADAY REVERSAL (LIQUIDITY SWEEP)
# =========================

import requests
import pandas as pd
import numpy as np
from config import BINANCE_REST_URL


# ================== DATA FETCHING & UTIL ==================

def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Ambil data candlestick Binance (REST)."""
    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI sederhana."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) untuk volatilitas.
    Dipakai untuk buffer SL supaya lebih adaptif dengan market.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


# ============================================================
#               LOGIC SMC INTRADAY (GENERIC)
# ============================================================

def detect_bias_generic(df: pd.DataFrame) -> bool:
    """
    Bias generik intraday:
    - close > EMA20 > EMA50
    - EMA20 & EMA50 naik (slope positif)
    """
    close = df["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    if len(close) < 30:
        return False

    last = close.iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]

    bias_stack = last > e20 > e50

    e20_prev = ema20.iloc[-8]
    e50_prev = ema50.iloc[-8]
    ema_slope_ok = (e20 > e20_prev) and (e50 > e50_prev)

    return bool(bias_stack and ema_slope_ok)


def detect_bias_5m(df_5m: pd.DataFrame) -> bool:
    """Alias khusus 5m, pakai rule generik."""
    return detect_bias_generic(df_5m)


def _find_fractal_swings(highs: np.ndarray, lows: np.ndarray):
    """Helper kecil: cari index swing high & swing low sederhana."""
    n = len(highs)
    swing_high_idx = []
    swing_low_idx = []
    for i in range(2, n - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_high_idx.append(i)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_low_idx.append(i)
    return swing_high_idx, swing_low_idx


def detect_micro_choch(df_5m: pd.DataFrame):
    """
    Micro CHoCH intraday:
    - swing high & swing low terakhir lebih tinggi dari swing kecil sebelumnya (bullish shift).
    Premium:
    - candle terakhir bullish
    - body > 1.3x rata-rata body 8 candle sebelumnya
    - upper wick <= 25% dari total range
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    opens = df_5m["open"].values
    closes = df_5m["close"].values

    n = len(highs)
    if n < 12:
        return False, False

    swing_high_idx, swing_low_idx = _find_fractal_swings(highs, lows)

    micro_choch = False
    if swing_high_idx and swing_low_idx:
        last_sw_high = swing_high_idx[-1]
        last_sw_low = swing_low_idx[-1]

        prev_sw_high = swing_high_idx[-2] if len(swing_high_idx) >= 2 else None
        prev_sw_low = swing_low_idx[-2] if len(swing_low_idx) >= 2 else None

        if prev_sw_high is not None and prev_sw_low is not None:
            last_high = highs[last_sw_high]
            last_low = lows[last_sw_low]
            prev_high = highs[prev_sw_high]
            prev_low = lows[prev_sw_low]
            micro_choch = bool(last_high > prev_high and last_low > prev_low)

    # Fallback ke cara lama kalau fractal tidak jelas / noise
    if not micro_choch:
        micro_choch = bool(highs[-1] > highs[-3] and lows[-1] > lows[-3])

    last_open = opens[-1]
    last_close = closes[-1]
    last_high = highs[-1]
    last_low = lows[-1]

    if last_close <= last_open:
        return micro_choch, False

    body = abs(last_close - last_open)
    past_bodies = np.abs(closes[-9:-1] - opens[-9:-1])
    avg_body = past_bodies.mean() if past_bodies.size > 0 else 0.0

    total_range = last_high - last_low
    if total_range <= 0 or avg_body <= 0:
        return micro_choch, False

    upper_wick = last_high - max(last_close, last_open)

    body_big_enough = body >= avg_body * 1.3
    wick_small_enough = (upper_wick / total_range) <= 0.25

    micro_choch_premium = bool(
        micro_choch and body_big_enough and wick_small_enough
    )

    return micro_choch, micro_choch_premium


def detect_micro_fvg(df_5m: pd.DataFrame):
    """
    Micro FVG bullish (imbalance kecil):
    - low candle n > high candle n-1 di beberapa candle terakhir.
    Scan beberapa candle dan ambil FVG yang paling dekat dengan harga sekarang.
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    n = len(highs)
    if n < 4:
        return False, 0.0, 0.0

    last_close = closes[-1]
    start = max(0, n - 15)
    best_diff = None
    best_low = 0.0
    best_high = 0.0

    for i in range(start, n - 1):
        if lows[i + 1] > highs[i]:
            fvg_low = highs[i]
            fvg_high = lows[i + 1]

            mid = (fvg_low + fvg_high) / 2.0
            diff = abs(last_close - mid)
            if (best_diff is None) or (diff < best_diff):
                best_diff = diff
                best_low = fvg_low
                best_high = fvg_high

    if best_diff is None:
        return False, 0.0, 0.0

    return True, float(best_low), float(best_high)


def detect_momentum(df_5m: pd.DataFrame):
    """
    Momentum reversal:
    - RSI sempat turun (oversold ringan) lalu recovery.
    - OK: min RSI 10 candle terakhir < 32 DAN RSI sekarang > 35
    - Premium: RSI 38–55 (awal pembalikan sehat, belum overbought)
    """
    closes = df_5m["close"]
    if len(closes) < 40:
        return False, False

    rsi_series = rsi(closes, 14)
    rsi_series = rsi_series.fillna(method="bfill").fillna(method="ffill")

    last_rsi = float(rsi_series.iloc[-1])
    recent = rsi_series.iloc[-10:-1]
    if recent.empty:
        return False, False

    min_recent = float(recent.min())

    momentum_ok = bool(min_recent < 32 and last_rsi > 35)
    momentum_premium = bool(38 <= last_rsi <= 55)

    return momentum_ok, momentum_premium


def detect_not_choppy(df_5m: pd.DataFrame, window: int = 25) -> bool:
    """
    Filter choppy intraday:
    - range total > ~1.6–1.8x rata-rata range candle.
    - candle terlalu kecil semua (super pelan) → choppy.
    - minimal ~55–60% candle bergerak searah (trending).
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    if len(highs) < window + 5:
        return False

    seg_high = highs[-window:]
    seg_low = lows[-window:]
    seg_close = closes[-window:]

    ranges = seg_high - seg_low
    full_range = seg_high.max() - seg_low.min()
    avg_range = ranges.mean()

    if avg_range <= 0:
        return False

    trendiness_ok = full_range > avg_range * 1.6

    tiny_range = avg_range <= (seg_low.mean() * 0.001)
    if tiny_range:
        return False

    up_moves = (seg_close[1:] > seg_close[:-1]).sum()
    down_moves = (seg_close[1:] < seg_close[:-1]).sum()
    total_moves = up_moves + down_moves

    if total_moves == 0:
        return False

    trend_ratio = max(up_moves, down_moves) / float(total_moves)
    direction_ok = trend_ratio >= 0.55

    return bool(trendiness_ok and direction_ok)


def detect_not_overextended(df_5m: pd.DataFrame,
                            ema_period: int = 20,
                            max_distance_pct: float = 0.015) -> bool:
    """
    TRUE kalau harga TIDAK terlalu jauh dari EMA (tidak over-extended).
    Untuk long:
    - close tidak lebih dari max_distance_pct di atas EMA20.
    (Di reversal: kita masih hindari entry setelah candle lari terlalu jauh.)
    """
    close = df_5m["close"]
    ema20 = ema(close, ema_period)

    last_close = close.iloc[-1]
    last_ema = ema20.iloc[-1]

    if last_ema <= 0:
        return True

    dist_pct = (last_close - last_ema) / last_ema
    if dist_pct > max_distance_pct:
        return False

    return True


def detect_volatility_regime(df_5m: pd.DataFrame,
                             period: int = 14,
                             lookback: int = 60,
                             low_q: float = 0.25,
                             high_q: float = 0.80) -> bool:
    """
    Volatility regime check:
    - ATR terakhir tidak terlalu rendah dan tidak terlalu tinggi
      dibanding distribusi ATR lookback.
    """
    atr_series = atr(df_5m, period=period).dropna()
    if len(atr_series) < lookback:
        return True

    tail = atr_series.iloc[-lookback:]
    last_atr = tail.iloc[-1]
    q_low = tail.quantile(low_q)
    q_high = tail.quantile(high_q)

    return bool(q_low <= last_atr <= q_high)


def compute_trend_strength(df: pd.DataFrame, lookback: int = 40) -> float:
    """
    Trend strength:
    - ratio bar dalam lookback yang punya close > EMA20 > EMA50
    (dipakai sebagai info tambahan kalau mau nanti).
    """
    close = df["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    if len(close) < lookback + 5:
        return 0.0

    c_seg = close.iloc[-lookback:]
    e20_seg = ema20.iloc[-lookback:]
    e50_seg = ema50.iloc[-lookback:]

    mask = (c_seg > e20_seg) & (e20_seg > e50_seg)
    strength = mask.sum() / float(len(mask))
    return float(strength)


# ============================================================
#            LIQUIDITY SWEEP (KHUSUS REVERSAL LONG)
# ============================================================

def detect_liquidity_sweep_long(df_5m: pd.DataFrame,
                                lookback: int = 40):
    """
    Liquidity sweep untuk setup long (reversal):
    - Cari low penting (lowest dalam window lookback sebelum 5 candle terakhir)
    - Dalam 5 candle terakhir, ada low yang MENEMBUS sedikit di bawah low itu
      (ambil stop) lalu CLOSE kembali di atas low tersebut (rejection kuat).
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    n = len(highs)
    if n < lookback + 5:
        return False, 0.0

    prior_start = max(0, n - (lookback + 5))
    prior_end = n - 5
    prior_segment = lows[prior_start:prior_end]
    recent_segment = lows[n - 5:n]

    if prior_segment.size == 0 or recent_segment.size == 0:
        return False, 0.0

    prev_low = float(prior_segment.min())
    sweep_low = float(recent_segment.min())

    sweep_idx_relative = int(recent_segment.argmin())  # 0..4
    sweep_idx = (n - 5) + sweep_idx_relative

    sweep_bar_low = lows[sweep_idx]
    sweep_bar_high = highs[sweep_idx]
    close_sweep = closes[sweep_idx]

    if prev_low <= 0:
        return False, 0.0

    broke_low = sweep_low < prev_low * 0.999  # minimal ~0.1% di bawah
    reclaimed = close_sweep > prev_low

    sweep_range = sweep_bar_high - sweep_bar_low
    if sweep_range <= 0:
        return False, 0.0

    pos_in_range = (close_sweep - sweep_bar_low) / sweep_range
    rejection_ok = pos_in_range >= 0.6

    ok = bool(broke_low and reclaimed and rejection_ok)
    return ok, prev_low


# ============================================================
#                  ENTRY / SL / TP GENERATION
# ============================================================

def build_entry_sl_tp_aggressive(df_5m: pd.DataFrame,
                                 fvg_low: float,
                                 fvg_high: float) -> dict:
    """
    Entry intraday (dipakai Apex & Reversal):
    - kalau ada micro FVG → pakai mid FVG
    - kalau tidak → pakai close terakhir
    SL:
    - sedikit di bawah swing low pendek
    - ada buffer berbasis ATR (bukan hanya % fixed)
    TP:
    - kelipatan jarak entry–SL (RR 1:1.5 / 1:2 / 1:3)
    """
    closes = df_5m["close"].values
    lows = df_5m["low"].values

    last_close = closes[-1]

    if fvg_low and fvg_high and fvg_high > fvg_low:
        raw_entry = (fvg_low + fvg_high) / 2.0
    else:
        raw_entry = last_close

    entry = min(raw_entry, last_close)

    recent_low = lows[-5:].min()

    atr_series = atr(df_5m, period=14)
    atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

    if atr_val > 0:
        buffer = atr_val * 0.3
    else:
        buffer = abs(last_close) * 0.002

    sl = recent_low - buffer

    risk = abs(entry - sl)
    if risk <= 0:
        risk = max(abs(entry) * 0.003, 1e-8)

    tp1 = entry + risk * 1.5
    tp2 = entry + risk * 2.0
    tp3 = entry + risk * 3.0

    return {
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "risk_per_unit": float(risk),
    }


# ============================================================
#                    ANALYZE SYMBOL (REVERSAL)
# ============================================================

def analyse_symbol(symbol: str):
    """
    Versi SMC INTRADAY REVERSAL (LONG only):
    - Timeframe entry: 5m
    - Konteks trend: 15m & 1H tetap harus bullish (bias generik OK)
    - Fokus:
        * Liquidity sweep di low penting (ambil stop)
        * Rejection kuat (close kembali di atas low penting)
        * Micro CHoCH (shift naik)
        * Optional confluence: micro FVG
        * Momentum reversal (RSI sempat oversold, lalu pulih)
        * Market tidak choppy, tidak over-extended, volatility OK

    Return:
    - None, None  → tidak ada setup
    - conditions, levels → setup valid
    """
    try:
        df_5m = get_klines(symbol, "5m", 240)
        df_15m = get_klines(symbol, "15m", 240)
        df_1h = get_klines(symbol, "1h", 240)
    except Exception as e:
        print(f"[{symbol}] ERROR fetching data:", e)
        return None, None

    if any(df is None or df.empty for df in (df_5m, df_15m, df_1h)):
        print(f"[{symbol}] Empty dataframe on one of TF (5m/15m/1h)")
        return None, None

    bias_5m_up = detect_bias_5m(df_5m)
    bias_15m_up = detect_bias_generic(df_15m)
    bias_1h_up = detect_bias_generic(df_1h)

    liq_sweep, swept_level = detect_liquidity_sweep_long(df_5m)
    micro_choch, micro_choch_premium = detect_micro_choch(df_5m)
    micro_fvg, fvg_low, fvg_high = detect_micro_fvg(df_5m)
    momentum_ok, momentum_premium = detect_momentum(df_5m)
    not_choppy = detect_not_choppy(df_5m)
    not_overextended = detect_not_overextended(df_5m)
    volatility_ok = detect_volatility_regime(df_5m)

    # Trend strength info (optional)
    ts_5m = compute_trend_strength(df_5m, lookback=40)
    ts_15m = compute_trend_strength(df_15m, lookback=40)
    ts_1h = compute_trend_strength(df_1h, lookback=40)

    # Syarat inti Reversal:
    # - Bias 15m & 1H harus up (reversal di dalam uptrend besar)
    # - Liquidity sweep Wajib
    # - Micro CHoCH Wajib
    # - Momentum reversal OK
    # - Market tidak choppy
    # - Tidak over-extended (jangan kejar candle lari)
    # - Volatility regime OK
    if not (
        bias_15m_up
        and bias_1h_up
        and liq_sweep
        and micro_choch
        and momentum_ok
        and not_choppy
        and not_overextended
        and volatility_ok
    ):
        return None, None

    htf_15m_trend_ok = bias_15m_up
    htf_1h_trend_ok = bias_1h_up

    last_high = df_5m["high"].iloc[-1]
    last_low = df_5m["low"].iloc[-1]
    last_range = last_high - last_low

    levels = build_entry_sl_tp_aggressive(df_5m, fvg_low, fvg_high)
    entry = levels["entry"]

    # Hindari entry terlalu dekat pucuk candle terakhir
    if last_range > 0 and (last_high - entry) < (0.25 * last_range):
        return None, None

    setup_score = 0
    if micro_choch_premium:
        setup_score += 1
    if micro_fvg:
        setup_score += 1
    if momentum_premium:
        setup_score += 1

    conditions = {
        "symbol": symbol.upper(),
        "timeframe": "5m",
        "bias_ok": bias_5m_up,               # tetap ada untuk kompatibilitas scoring
        "htf_15m_trend_ok": htf_15m_trend_ok,
        "htf_1h_trend_ok": htf_1h_trend_ok,
        "micro_choch": micro_choch,
        "micro_choch_premium": micro_choch_premium,
        "micro_fvg": micro_fvg,
        "momentum_ok": momentum_ok,
        "momentum_premium": momentum_premium,
        "not_choppy": not_choppy,
        "not_overextended": not_overextended,
        "setup_score": setup_score,  # 0–3
        "volatility_ok": volatility_ok,
        "trend_strength_5m": ts_5m,
        "trend_strength_15m": ts_15m,
        "trend_strength_1h": ts_1h,
        "liquidity_sweep": liq_sweep,
        "swept_level": swept_level,
        "mode": "INTRADAY_REVERSAL",
    }

    return conditions, levels

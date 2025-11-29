# telegram/telegram_broadcast.py
# =========================
# broadcast_signal + build_signal_message (INTRADAY REVERSAL)
# =========================

import time

from config import TELEGRAM_ADMIN_ID
from core.bot_state import state, is_vip, cleanup_expired_vip
from telegram.telegram_common import send_telegram


def broadcast_signal(text: str):
    """Kirim sinyal:
    - SELALU ke admin (unlimited)
    - Juga ke semua subscribers (FREE:max 2 sinyal per hari / VIP: unlimited)
    """
    today = time.strftime("%Y-%m-%d")
    if state.daily_date != today:
        state.daily_date = today
        state.daily_counts = {}
        cleanup_expired_vip()
        print("Reset daily_counts & cleanup VIP untuk hari baru:", today)

    # admin
    if TELEGRAM_ADMIN_ID:
        try:
            send_telegram(text, chat_id=int(TELEGRAM_ADMIN_ID))
        except Exception as e:
            print("Gagal kirim ke admin:", e)
    else:
        print("⚠️ TELEGRAM_ADMIN_ID belum di-set. Admin tidak menerima sinyal.")

    # user
    if not state.subscribers:
        print("Belum ada subscriber. Hanya admin yang menerima sinyal.")
        return

    for cid in list(state.subscribers):
        if TELEGRAM_ADMIN_ID and str(cid) == str(TELEGRAM_ADMIN_ID):
            continue

        if is_vip(cid):
            send_telegram(text, chat_id=cid)
            continue

        count = state.daily_counts.get(cid, 0)
        if count >= 2:
            continue

        send_telegram(text, chat_id=cid)
        state.daily_counts[cid] = count + 1


def build_signal_message(
    symbol: str,
    levels: dict,
    conditions: dict,
    score: int,
    tier: str,
    side: str = "long"
) -> str:
    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]

    def mark(flag: bool) -> str:
        return "✅" if flag else "❌"

    side_label = "LONG" if side == "long" else "SHORT"

    bias_ok             = conditions.get("bias_ok")
    htf_15m_trend_ok    = conditions.get("htf_15m_trend_ok")
    htf_1h_trend_ok     = conditions.get("htf_1h_trend_ok")
    micro_choch         = conditions.get("micro_choch")
    micro_choch_premium = conditions.get("micro_choch_premium")
    micro_fvg           = conditions.get("micro_fvg")
    momentum_ok         = conditions.get("momentum_ok")
    momentum_premium    = conditions.get("momentum_premium")
    not_choppy          = conditions.get("not_choppy")
    not_overextended    = conditions.get("not_overextended")
    setup_score         = conditions.get("setup_score", 0)
    liquidity_sweep     = conditions.get("liquidity_sweep", False)

    text = f"""🟦 SMC INTRADAY REVERSAL — {symbol}

TF: Entry 5m | Context 15m & 1H
Mode: Liquidity Sweep Reversal
Score: {score}/125 — Tier {tier} — {side_label}
Setup internal (5m): {setup_score}/3

💰 Harga
• Entry : `{entry:.6f}`
• SL    : `{sl:.6f}`
• TP1   : `{tp1:.6f}`
• TP2   : `{tp2:.6f}`
• TP3   : `{tp3:.6f}`

📌 Checklist Multi-Timeframe
• Bias 5m (Close > EMA20 > EMA50) : {mark(bias_ok)}
• Bias 15m uptrend                 : {mark(htf_15m_trend_ok)}
• Bias 1H uptrend                  : {mark(htf_1h_trend_ok)}

📌 Checklist Reversal (5m)
• Liquidity sweep low penting      : {mark(liquidity_sweep)}
• Micro CHoCH (shift naik)         : {mark(micro_choch)}
• Micro CHoCH premium candle       : {mark(micro_choch_premium)}
• Micro FVG (demand kecil)         : {mark(micro_fvg)}
• Momentum reversal OK             : {mark(momentum_ok)}
• Momentum premium (RSI 38–55)     : {mark(momentum_premium)}
• Market tidak choppy              : {mark(not_choppy)}
• Tidak over-extended dari EMA     : {mark(not_overextended)}

📝 Catatan
• Strategi intraday reversal: masuk setelah liquidity sweep & rejection kuat.
• Reversal terjadi di dalam uptrend 15m & 1H (bukan counter-trend ngawur).
• Fokus entry di dekat low setelah sweep + CHoCH + FVG + momentum pulih.
• Tier A+ = confluence paling lengkap & kondisi market paling bersih.

Free: maksimal 2 sinyal/hari. VIP: Unlimited sinyal.
"""
    return text

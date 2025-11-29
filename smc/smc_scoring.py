# smc/smc_scoring.py
# =========================
# SMC INTRADAY SCORING (APEX / REVERSAL)
# =========================

from typing import Dict


def score_smc_signal(c: Dict) -> int:
    """
    Scoring untuk setup SMC Intraday (0–125).
    Dipakai baik untuk mode Apex maupun Reversal,
    selama field 'conditions' memakai key yang sama.

    Menggunakan:
    - 5m bias + struktur
    - 15m trend
    - 1H trend
    - Kualitas micro (CHoCH premium, FVG, momentum premium)
    - Market quality (tidak choppy, tidak overextended)
    """
    score = 0

    bias_ok = bool(c.get("bias_ok"))
    micro_choch = bool(c.get("micro_choch"))
    micro_choch_premium = bool(c.get("micro_choch_premium"))
    micro_fvg = bool(c.get("micro_fvg"))
    momentum_ok = bool(c.get("momentum_ok"))
    momentum_premium = bool(c.get("momentum_premium"))
    not_choppy = bool(c.get("not_choppy"))
    not_overextended = bool(c.get("not_overextended"))
    htf_15m_trend_ok = bool(c.get("htf_15m_trend_ok"))
    htf_1h_trend_ok = bool(c.get("htf_1h_trend_ok"))

    setup_score_internal = c.get("setup_score") or 0
    setup_score_internal = max(0, min(int(setup_score_internal), 3))

    # Bias 5m
    if bias_ok:
        score += 15

    # Trend HTF 15m
    if htf_15m_trend_ok:
        score += 10

    # Trend HTF 1H
    if htf_1h_trend_ok:
        score += 10

    # Micro CHoCH
    if micro_choch:
        score += 25
    if micro_choch_premium:
        score += 10

    # Micro FVG
    if micro_fvg:
        score += 15

    # Momentum
    if momentum_ok:
        score += 15
    if momentum_premium:
        score += 10

    # Market quality
    if not_choppy:
        score += 10
    if not_overextended:
        score += 10

    # Synergy bonus:
    # - Bias 5m + HTF 15m + HTF 1H + micro CHoCH + momentum OK
    if bias_ok and htf_15m_trend_ok and htf_1h_trend_ok and micro_choch and momentum_ok:
        synergy = 4 + setup_score_internal * 2  # max 10
        score += synergy

    return int(score)


def tier_from_score(score: int) -> str:
    """
    Tier:
    - A+ : >= 110
    - A  : 90–109
    - B  : 70–89
    - NONE : < 70
    """
    if score >= 110:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 70:
        return "B"
    else:
        return "NONE"


def should_send_tier(tier: str, min_tier: str) -> bool:
    """
    Urutan: NONE < B < A < A+
    """
    order = {"NONE": 0, "B": 1, "A": 2, "A+": 3}
    return order.get(tier, 0) >= order.get(min_tier, 2)


def evaluate_smc_signal(conditions: Dict, min_tier: str = "A") -> Dict:
    """
    Helper kecil supaya enak dipakai di kode lain.

    Return dict:
    {
        "score": int,
        "tier": str,
        "should_send": bool
    }
    """
    score = score_smc_signal(conditions)
    tier = tier_from_score(score)
    send = should_send_tier(tier, min_tier)
    return {
        "score": score,
        "tier": tier,
        "should_send": send,
    }

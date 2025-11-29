# core/bot_state.py
# Menangani state global, VIP, subscribers, dan load/save konfigurasi bot.

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Set

from config import (
    TELEGRAM_ADMIN_ID,
    MIN_VOLUME_USDT,
    MAX_USDT_PAIRS,
    MIN_TIER_TO_SEND,
    SIGNAL_COOLDOWN_SECONDS,
)

# ===== FILE DATA PERSISTENT =====
SUBSCRIBERS_FILE = "subscribers.json"
VIP_FILE = "vip_users.json"
STATE_FILE = "bot_state.json"


@dataclass
class BotState:
    scanning: bool = False            # apakah scan market ON
    running: bool = True              # kontrol main loop
    last_update_id: Optional[int] = None

    last_signal_time: Dict[str, float] = field(default_factory=dict)
    min_tier: str = MIN_TIER_TO_SEND
    cooldown_seconds: int = SIGNAL_COOLDOWN_SECONDS
    debug: bool = False

    subscribers: Set[int] = field(default_factory=set)

    vip_users: Dict[int, float] = field(default_factory=dict)
    daily_counts: Dict[int, int] = field(default_factory=dict)
    daily_date: str = ""

    # restart & pairs filter
    request_soft_restart: bool = False
    force_pairs_refresh: bool = False

    # parameter scan market yang bisa diatur admin
    min_volume_usdt: float = MIN_VOLUME_USDT   # filter volume minimum
    max_pairs: int = MAX_USDT_PAIRS            # jumlah pair yang discan


state = BotState()


# ================== UTIL & STORAGE ==================

def is_admin(chat_id: int) -> bool:
    return TELEGRAM_ADMIN_ID and str(chat_id) == str(TELEGRAM_ADMIN_ID)


def load_subscribers() -> Set[int]:
    if not os.path.exists(SUBSCRIBERS_FILE):
        return set()
    try:
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(int(x) for x in data)
    except Exception as e:
        print("Gagal load subscribers:", e)
        return set()


def save_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(state.subscribers), f)
    except Exception as e:
        print("Gagal simpan subscribers:", e)


def load_vip_users() -> Dict[int, float]:
    if not os.path.exists(VIP_FILE):
        return {}
    try:
        with open(VIP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): float(v) for k, v in data.items()}
    except Exception as e:
        print("Gagal load VIP:", e)
        return {}


def save_vip_users():
    try:
        with open(VIP_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in state.vip_users.items()}, f)
    except Exception as e:
        print("Gagal simpan VIP:", e)


def is_vip(user_id: int) -> bool:
    """VIP jika expiry_ts > sekarang, atau jika dia admin."""
    now = time.time()
    if TELEGRAM_ADMIN_ID and str(user_id) == str(TELEGRAM_ADMIN_ID):
        return True
    exp = state.vip_users.get(user_id)
    return bool(exp and exp > now)


def cleanup_expired_vip():
    """Hapus VIP yang sudah kedaluwarsa dari memori + file."""
    now = time.time()
    expired_ids = [uid for uid, exp in state.vip_users.items() if exp <= now]
    if not expired_ids:
        return
    for uid in expired_ids:
        del state.vip_users[uid]
    save_vip_users()
    print("VIP expired dihapus otomatis:", expired_ids)


def load_bot_state():
    """Load scanning/min_tier/cooldown dari file (jika ada)."""
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.scanning = bool(data.get("scanning", False))
        state.min_tier = data.get("min_tier", state.min_tier)
        state.cooldown_seconds = int(data.get("cooldown_seconds", state.cooldown_seconds))
        state.min_volume_usdt = float(data.get("min_volume_usdt", state.min_volume_usdt))
        state.max_pairs = int(data.get("max_pairs", state.max_pairs))

        print(
            f"Bot state loaded: scanning={state.scanning}, "
            f"min_tier={state.min_tier}, cooldown={state.cooldown_seconds}, "
            f"min_volume_usdt={state.min_volume_usdt}, max_pairs={state.max_pairs}"
        )
    except Exception as e:
        print("Gagal load bot_state:", e)


def save_bot_state():
    """Simpan scanning/min_tier/cooldown ke file."""
    try:
        data = {
            "scanning": state.scanning,
            "min_tier": state.min_tier,
            "cooldown_seconds": state.cooldown_seconds,
            "min_volume_usdt": state.min_volume_usdt,
            "max_pairs": state.max_pairs,
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print("Gagal simpan bot_state:", e)

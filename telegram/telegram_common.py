# telegram/telegram_common.py
# Common util Telegram: send_telegram & hard_restart.

import json
import os
import sys

import requests

from config import TELEGRAM_TOKEN, TELEGRAM_ADMIN_ID
from core.bot_state import state


def send_telegram(
    text: str,
    chat_id: int | None = None,
    reply_markup: dict | None = None,
) -> None:
    if not TELEGRAM_TOKEN:
        print("Telegram token belum di-set.")
        return

    if chat_id is None:
        if not TELEGRAM_ADMIN_ID:
            print("Tidak ada TELEGRAM_ADMIN_ID.")
            return
        chat_id = int(TELEGRAM_ADMIN_ID)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if reply_markup is not None:
        data["reply_markup"] = json.dumps(reply_markup)

    try:
        r = requests.post(url, data=data, timeout=10)
        if not r.ok:
            print("Gagal kirim Telegram:", r.text)
    except Exception as e:
        print("Error kirim Telegram:", e)


def hard_restart():
    """Restart penuh proses Python (hard restart)."""
    print("Hard restart dimulai...")
    state.running = False
    sys.stdout.flush()
    os.execl(sys.executable, sys.executable, *sys.argv)

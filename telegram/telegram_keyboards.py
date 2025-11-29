# telegram/telegram_keyboards.py
# Hanya untuk reply keyboard user & admin.

def get_user_reply_keyboard() -> dict:
    return {
        "keyboard": [
            [
                {"text": "🏠 Home"},
                {"text": "🔔 Aktifkan Sinyal"},
                {"text": "🔕 Nonaktifkan Sinyal"},
            ],
            [
                {"text": "📊 Status Saya"},
                {"text": "⭐ Upgrade VIP"},
                {"text": "❓ Bantuan"},
            ],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
    }


def get_admin_reply_keyboard() -> dict:
    return {
        "keyboard": [
            [
                {"text": "🏠 Home"},
                {"text": "▶️ Start Scan"},
                {"text": "⏸️ Pause Scan"},
            ],
            [
                {"text": "⛔ Stop Scan"},
                {"text": "📊 Status Bot"},
                {"text": "⚙️ Mode Tier"},
            ],
            [
                {"text": "⏲️ Cooldown"},
                {"text": "📈 Min Volume"},
                {"text": "📌 Max Pair"},
            ],
            [
                {"text": "⭐ VIP Control"},
                {"text": "🔄 Restart Bot"},
            ],
            [
                {"text": "❓ Help Admin"},
            ],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
    }

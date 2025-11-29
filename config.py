# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === TELEGRAM ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
# ID admin utama (chat_id Telegram kamu)
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", "")
# username admin utama
TELEGRAM_ADMIN_USERNAME = os.getenv("TELEGRAM_ADMIN_USERNAME", "")

# === BINANCE ===
BINANCE_REST_URL = "https://fapi.binance.com"
BINANCE_STREAM_URL = "wss://fstream.binance.com/stream"

# Filtering volume minimum (dalam USDT)
MIN_VOLUME_USDT = 1_000_000.0

# Berapa banyak pair USDT yang discan
MAX_USDT_PAIRS = 1000

# Tier minimum sinyal yang dikirim: "A+", "A", "B"
MIN_TIER_TO_SEND = "A"  # balanced default

# Cooldown default antar sinyal per pair (detik)
SIGNAL_COOLDOWN_SECONDS = 1800  # 30 menit

# Refresh interval untuk daftar pair (jam)
REFRESH_PAIR_INTERVAL_HOURS = 24  # satuan jam

# binance/binance_scan.py
# Fokus ke WebSocket Binance: listen 5m close, analyse_symbol, kirim sinyal.

import asyncio
import json
import time
from typing import List

import websockets

from config import BINANCE_STREAM_URL, REFRESH_PAIR_INTERVAL_HOURS
from core.bot_state import (
    state,
    load_subscribers,
    load_vip_users,
    cleanup_expired_vip,
    load_bot_state,
)
from binance.binance_pairs import get_usdt_pairs
from smc.smc_logic import analyse_symbol
from smc.smc_scoring import evaluate_smc_signal
from telegram.telegram_broadcast import build_signal_message, broadcast_signal


async def run_bot():
    # load data persistent
    state.subscribers = load_subscribers()
    state.vip_users = load_vip_users()
    state.daily_date = time.strftime("%Y-%m-%d")
    cleanup_expired_vip()
    load_bot_state()

    print(f"Loaded {len(state.subscribers)} subscribers, {len(state.vip_users)} VIP users.")

    symbols: List[str] = []
    last_pairs_refresh: float = 0.0
    refresh_interval = REFRESH_PAIR_INTERVAL_HOURS * 3600

    while state.running:
        try:
            now = time.time()
            if (
                not symbols
                or (now - last_pairs_refresh) > refresh_interval
                or state.force_pairs_refresh
            ):
                print("Refresh daftar pair USDT berdasarkan volume...")
                symbols = get_usdt_pairs(state.max_pairs, state.min_volume_usdt)
                last_pairs_refresh = now
                state.force_pairs_refresh = False
                print(f"Scan {len(symbols)} pair:", ", ".join(s.upper() for s in symbols))

            streams = "/".join([f"{s}@kline_5m" for s in symbols])
            ws_url = f"{BINANCE_STREAM_URL}?streams={streams}"

            print("Menghubungkan ke WebSocket...")
            async with websockets.connect(ws_url) as ws:
                print("WebSocket terhubung.")
                if state.scanning:
                    print("Scan sebelumnya AKTIF → melanjutkan scan otomatis.")
                else:
                    print("Bot dalam mode STANDBY. Gunakan /startscan untuk mulai scan.\n")

                while state.running:
                    if state.request_soft_restart:
                        print("Soft restart diminta → memutus WS & refresh engine...")
                        state.request_soft_restart = False
                        break

                    if time.time() - last_pairs_refresh > refresh_interval:
                        print("Interval refresh pair tercapai → refresh daftar pair & reconnect WebSocket...")
                        break

                    msg = await ws.recv()
                    data = json.loads(msg)

                    kline = data.get("data", {}).get("k", {})
                    if not kline:
                        continue

                    is_closed = kline.get("x", False)
                    symbol = kline.get("s", "").upper()

                    if not is_closed or not symbol:
                        continue

                    if not state.scanning:
                        continue

                    now = time.time()
                    if state.cooldown_seconds > 0:
                        last_ts = state.last_signal_time.get(symbol)
                        if last_ts and now - last_ts < state.cooldown_seconds:
                            if state.debug:
                                print(
                                    f"[{symbol}] Skip cooldown "
                                    f"({int(now - last_ts)}s/{state.cooldown_seconds}s)"
                                )
                            continue

                    if state.debug:
                        print(f"[{time.strftime('%H:%M:%S')}] 5m close: {symbol}")

                    conditions, levels = analyse_symbol(symbol)
                    if not conditions or not levels:
                        continue

                    eval_res = evaluate_smc_signal(conditions, min_tier=state.min_tier)
                    score = eval_res["score"]
                    tier = eval_res["tier"]

                    if not eval_res["should_send"]:
                        if state.debug:
                            print(f"[{symbol}] Tier {tier} < {state.min_tier}, skip.")
                        continue

                    text = build_signal_message(symbol, levels, conditions, score, tier)
                    broadcast_signal(text)

                    state.last_signal_time[symbol] = now
                    print(f"[{symbol}] Sinyal dikirim: Score {score}, Tier {tier}")

        except websockets.ConnectionClosed:
            print("WebSocket terputus. Reconnect dalam 5 detik...")
            await asyncio.sleep(5)
        except Exception as e:
            print("Error di run_bot (luar):", e)
            print("Coba reconnect dalam 5 detik...")
            await asyncio.sleep(5)

    print("run_bot selesai karena state.running = False")

# telegram/telegram_commands.py
# /start, /help, /mode, /cooldown, VIP, dll + callback.

import time

from config import TELEGRAM_ADMIN_USERNAME
from core.bot_state import (
    state,
    is_admin,
    is_vip,
    save_bot_state,
    save_subscribers,
    save_vip_users,
)
from telegram.telegram_common import send_telegram, hard_restart
from telegram.telegram_keyboards import get_user_reply_keyboard, get_admin_reply_keyboard


def handle_user_start(chat_id: int):
    pkg = "VIP" if is_vip(chat_id) else "FREE"
    limit = "Unlimited" if is_vip(chat_id) else "2 sinyal per hari"
    active = "AKTIF" if chat_id in state.subscribers else "Tidak aktif"

    send_telegram(
        f"🟦 SMC INTRADAY REVERSAL BOT\n\n"
        f"Status Kamu:\n"
        f"• Paket : *{pkg}*\n"
        f"• Limit : *{limit}*\n"
        f"• Sinyal : *{active}*\n\n"
        f"Gunakan menu di bawah untuk mengatur sinyal.",
        chat_id,
        reply_markup=get_user_reply_keyboard(),
    )


def handle_admin_start(chat_id: int):
    send_telegram(
        "👑 SMC INTRADAY REVERSAL — ADMIN PANEL\n\n"
        "Bot siap. Gunakan menu di bawah untuk kontrol penuh.",
        chat_id,
        reply_markup=get_admin_reply_keyboard(),
    )


def handle_command(cmd: str, args: list, chat_id: int):
    cmd = cmd.lower()

    # /start
    if cmd == "/start":
        if is_admin(chat_id):
            handle_admin_start(chat_id)
        else:
            handle_user_start(chat_id)
        return

    # /help
    if cmd == "/help":
        if is_admin(chat_id):
            send_telegram(
                "📖 Bantuan admin tersedia lewat tombol *❓ Help Admin* pada menu bawah.",
                chat_id,
                reply_markup=get_admin_reply_keyboard(),
            )
        else:
            send_telegram(
                "📖 Bantuan user tersedia lewat tombol *❓ Bantuan* pada menu bawah.",
                chat_id,
                reply_markup=get_user_reply_keyboard(),
            )
        return

    # ---------- USER COMMANDS ----------
    if not is_admin(chat_id):
        if cmd == "/activate":
            if chat_id in state.subscribers:
                send_telegram("ℹ️ Pencarian sinyal sudah *AKTIF*.", chat_id)
            else:
                state.subscribers.add(chat_id)
                save_subscribers()
                send_telegram("🔔 Pencarian sinyal *diaktifkan!*", chat_id)
            return

        if cmd == "/deactivate":
            if chat_id in state.subscribers:
                state.subscribers.remove(chat_id)
                save_subscribers()
                send_telegram("🔕 Pencarian sinyal *dinonaktifkan.*", chat_id)
            else:
                send_telegram("ℹ️ Pencarian sinyal sudah *tidak aktif*.", chat_id)
            return

        if cmd == "/mystatus":
            now = time.time()
            exp = state.vip_users.get(chat_id)
            if exp and exp > now:
                days_left = int((exp - now) / 86400)
                pkg = f"VIP (sisa ~{days_left} hari)"
                limit = "Unlimited"
            else:
                pkg = "FREE"
                limit = "2 sinyal per hari"

            active = "AKTIF ✅" if chat_id in state.subscribers else "TIDAK AKTIF ❌"
            send_telegram(
                "📊 *STATUS KAMU*\n\n"
                f"Paket  : *{pkg}*\n"
                f"Limit  : *{limit}*\n"
                f"Sinyal : *{active}*\n"
                f"User ID: `{chat_id}`",
                chat_id,
            )
            return

        send_telegram("Perintah tidak dikenali. Gunakan menu bawah atau /start.", chat_id)
        return

    # ---------- ADMIN COMMANDS ----------
    if cmd == "/startscan":
        if state.scanning:
            send_telegram("ℹ️ Scan sudah *AKTIF*.", chat_id)
        else:
            state.scanning = True
            save_bot_state()
            send_telegram("▶️ Scan market *dimulai*.", chat_id)
        return

    if cmd == "/pausescan":
        if not state.scanning:
            send_telegram("ℹ️ Scan sudah *PAUSE*.", chat_id)
        else:
            state.scanning = False
            save_bot_state()
            send_telegram("⏸️ Scan market *dijeda* (sementara).", chat_id)
        return

    if cmd == "/stopscan":
        if not state.scanning and not state.last_signal_time:
            send_telegram("ℹ️ Scan sudah *NON-AKTIF* total.", chat_id)
        else:
            state.scanning = False
            state.last_signal_time.clear()
            save_bot_state()
            send_telegram(
                "⛔ Scan market *dihentikan total.*\n"
                "Gunakan /startscan untuk mulai lagi dari awal.",
                chat_id,
            )
        return

    if cmd == "/status":
        send_telegram(
            "📊 *STATUS BOT*\n\n"
            f"Scan       : {'AKTIF' if state.scanning else 'STANDBY'}\n"
            f"Min Tier   : {state.min_tier}\n"
            f"Cooldown   : {state.cooldown_seconds} detik\n"
            f"Min Volume : {state.min_volume_usdt:,.0f} USDT\n"
            f"Max Pairs  : {state.max_pairs} pair\n"
            f"Subscribers: {len(state.subscribers)} user\n"
            f"VIP Users  : {len(state.vip_users)} user\n",
            chat_id,
        )
        return

    if cmd == "/mode":
        if not args:
            send_telegram(
                "Mode sekarang:\n"
                f"- Min Tier: {state.min_tier}\n"
                "Gunakan: /mode aplus | a | b",
                chat_id,
            )
            return
        mode = args[0].lower()
        if mode == "aplus":
            state.min_tier = "A+"
        elif mode == "a":
            state.min_tier = "A"
        elif mode == "b":
            state.min_tier = "B"
        else:
            send_telegram("Mode tidak dikenali. Gunakan: aplus | a | b", chat_id)
            return
        save_bot_state()
        send_telegram(f"⚙️ Mode tier di-set ke: *{state.min_tier}*.", chat_id)
        return

    if cmd == "/cooldown":
        if not args:
            send_telegram(
                f"Cooldown sekarang: {state.cooldown_seconds} detik.\n"
                "Contoh: /cooldown 300  (5 menit)",
                chat_id,
            )
            return
        try:
            cd = int(args[0])
            if cd < 0:
                raise ValueError
            state.cooldown_seconds = cd
            save_bot_state()
            send_telegram(f"⏲️ Cooldown di-set ke {cd} detik.", chat_id)
        except ValueError:
            send_telegram("Format salah. Gunakan: /cooldown 300", chat_id)
        return

    if cmd == "/minvol":
        if not args:
            send_telegram(
                "📈 *SET MINIMUM VOLUME USDT*\n\n"
                f"Sekarang: `{state.min_volume_usdt:,.0f}` USDT\n\n"
                "Contoh:\n"
                "`/minvol 50000000`  (50 juta USDT)\n"
                "`/minvol 100000000` (100 juta USDT)",
                chat_id,
            )
            return
        try:
            val = float(args[0])
            if val < 0:
                raise ValueError
            state.min_volume_usdt = val
            state.force_pairs_refresh = True
            save_bot_state()
            send_telegram(
                f"📈 Min volume di-set ke `{val:,.0f}` USDT.\n"
                "Daftar pair akan di-refresh pada scan berikutnya.",
                chat_id,
            )
        except ValueError:
            send_telegram("Format salah. Contoh: `/minvol 100000000`", chat_id)
        return

    if cmd == "/maxpairs":
        if not args:
            send_telegram(
                "📌 *SET MAXIMUM PAIR YANG DI-SCAN*\n\n"
                f"Sekarang: `{state.max_pairs}` pair\n\n"
                "Contoh:\n"
                "`/maxpairs 20`\n"
                "`/maxpairs 40`",
                chat_id,
            )
            return
        try:
            val = int(args[0])
            if val < 1:
                raise ValueError
            state.max_pairs = val
            state.force_pairs_refresh = True
            save_bot_state()
            send_telegram(
                f"📌 Max pairs di-set ke *{val}*.\n"
                "Daftar pair akan di-refresh pada scan berikutnya.",
                chat_id,
            )
        except ValueError:
            send_telegram("Format salah. Contoh: `/maxpairs 30`", chat_id)
        return

    if cmd == "/addvip":
        if not args:
            send_telegram("Gunakan: /addvip <user_id> [hari]", chat_id)
            return
        try:
            target_id = int(args[0])
            days = int(args[1]) if len(args) > 1 else 30
        except ValueError:
            send_telegram("Format salah. Contoh: /addvip 123456789 30", chat_id)
            return
        now = time.time()
        new_exp = now + days * 86400
        state.vip_users[target_id] = new_exp
        save_vip_users()
        send_telegram(f"⭐ VIP aktif untuk `{target_id}` selama {days} hari.", chat_id)
        send_telegram(
            f"🎉 VIP kamu diaktifkan selama {days} hari.\n"
            "Sinyal kamu sekarang *unlimited* per hari.",
            target_id,
        )
        return

    if cmd == "/removevip":
        if not args:
            send_telegram("Gunakan: /removevip <user_id>", chat_id)
            return
        try:
            target_id = int(args[0])
        except ValueError:
            send_telegram("Format salah. Contoh: /removevip 123456789", chat_id)
            return
        if target_id in state.vip_users:
            del state.vip_users[target_id]
            save_vip_users()
            send_telegram(f"VIP user `{target_id}` dihapus.", chat_id)
            send_telegram("VIP kamu telah dinonaktifkan. Kembali ke paket FREE.", target_id)
        else:
            send_telegram("User tersebut tidak terdaftar sebagai VIP.", chat_id)
        return

    if cmd == "/debug":
        if not args:
            send_telegram(f"Debug: {'ON' if state.debug else 'OFF'}", chat_id)
            return
        val = args[0].lower()
        if val == "on":
            state.debug = True
            send_telegram("Debug *ON*.", chat_id)
        elif val == "off":
            state.debug = False
            send_telegram("Debug *OFF*.", chat_id)
        else:
            send_telegram("Gunakan: /debug on | off", chat_id)
        return

    if cmd == "/softrestart":
        state.request_soft_restart = True
        state.force_pairs_refresh = True
        state.last_signal_time.clear()
        send_telegram("♻ Soft restart diminta. Bot akan refresh koneksi & engine.", chat_id)
        return

    if cmd == "/hardrestart":
        send_telegram("🔄 Hard restart dimulai. Bot akan hidup kembali sebentar lagi...", chat_id)
        hard_restart()
        return

    if cmd == "/stopbot":
        state.running = False
        send_telegram("⛔ Bot akan berhenti. Jalankan ulang main.py untuk start lagi.", chat_id)
        return

    send_telegram("Perintah admin tidak dikenali.", chat_id)


def handle_callback(data_cb: str, from_id: int, chat_id_cq: int):
    if data_cb == "user_soft_restart":
        return

    if data_cb in ("admin_soft_restart", "admin_hard_restart", "admin_restart_cancel"):
        if not is_admin(from_id):
            send_telegram("Tombol ini hanya untuk admin.", chat_id_cq)
            return

        if data_cb == "admin_soft_restart":
            state.request_soft_restart = True
            state.force_pairs_refresh = True
            state.last_signal_time.clear()
            send_telegram("♻ Soft restart dimulai. Bot akan refresh koneksi & engine.", chat_id_cq)
            return

        if data_cb == "admin_hard_restart":
            send_telegram("🔄 Hard restart dimulai. Bot akan hidup kembali sebentar lagi...", chat_id_cq)
            hard_restart()
            return

        if data_cb == "admin_restart_cancel":
            send_telegram("❌ Restart dibatalkan.", chat_id_cq)
            return

    if not is_admin(from_id):
        send_telegram("Tombol ini hanya untuk admin.", chat_id_cq)
        return

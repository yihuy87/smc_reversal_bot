# binance/binance_pairs.py
# Fungsi ambil & filter pair USDT berdasarkan volume.

from typing import List, Dict

import requests

from config import BINANCE_REST_URL


def get_usdt_pairs(max_pairs: int, min_volume_usdt: float) -> List[str]:
    """
    Ambil semua pair USDT yang statusnya TRADING,
    lalu filter hanya yang 24h quote volume >= min_volume_usdt USDT.
    """
    info_url = f"{BINANCE_REST_URL}/fapi/v1/exchangeInfo"
    r = requests.get(info_url, timeout=10)
    r.raise_for_status()
    info = r.json()

    usdt_symbols = []
    for s in info["symbols"]:
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
        ):
            usdt_symbols.append(s["symbol"])

    ticker_url = f"{BINANCE_REST_URL}/fapi/v1/ticker/24hr"
    r2 = requests.get(ticker_url, timeout=10)
    r2.raise_for_status()
    tickers = r2.json()

    vol_map: Dict[str, float] = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym in usdt_symbols:
            try:
                qv = float(t.get("quoteVolume", "0"))
            except ValueError:
                qv = 0.0
            vol_map[sym] = qv

    min_vol = float(min_volume_usdt)
    filtered = [s for s in usdt_symbols if vol_map.get(s, 0.0) >= min_vol]

    filtered_sorted = sorted(filtered, key=lambda s: vol_map.get(s, 0.0), reverse=True)

    symbols_lower = [s.lower() for s in filtered_sorted]

    if max_pairs > 0:
        symbols_lower = symbols_lower[:max_pairs]

    print(f"Filter volume >= {min_vol:,.0f} USDT → {len(symbols_lower)} pair.")
    return symbols_lower


import logging
import requests
from typing import List, Dict, Optional

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

def screen_global_dips(
    min_drop_pct: float = 10.0, 
    min_market_cap: int = 2000000000
) -> List[Dict]:
    """
    Screen largest daily losers across multiple regions using Yahoo Finance.

    Regions: US, Europe, UK, Asia
    """
    regions = {
        "us": "day_losers",
        "europe": "day_losers_eu", 
        "uk": "day_losers_gb",
        "asia": "day_losers_asia"
    }

    all_losers = []
    seen_symbols = set()  # Deduplicate

    for region_name, screener_id in regions.items():
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds={screener_id}&count=100"
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            data = response.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])

            region_losers = []
            for q in quotes:
                symbol = q.get("symbol", "")
                if not symbol or symbol in seen_symbols:
                    continue

                change_pct = q.get("regularMarketChangePercent", 0) or 0
                if change_pct > -min_drop_pct:
                    continue

                market_cap = q.get("marketCap", 0) or 0
                if market_cap < min_market_cap:
                    continue

                quote_type = q.get("quoteType", "")
                if quote_type in ["ETF", "MUTUALFUND"] or len(symbol) > 5:
                    continue

                region_losers.append({
                    "symbol": symbol,
                    "name": q.get("longName") or q.get("shortName") or symbol,
                    "price": q.get("regularMarketPrice"),
                    "change_pct": round(change_pct, 2),
                    "market_cap": market_cap,
                    "region": region_name.upper()
                })

            all_losers.extend(region_losers)
            seen_symbols.update([q["symbol"] for q in region_losers])

            logging.info(f"{region_name}: {len(region_losers)} candidates")

        except Exception as e:
            logging.warning(f"Region {region_name} failed: {e}")
            continue

    logging.info(f"Total global candidates: {len(all_losers)}")
    return all_losers

# Test function
if __name__ == "__main__":
    candidates = screen_global_dips()
    print(f"Found {len(candidates)} global dips")
    for c in candidates[:5]:
        print(c)

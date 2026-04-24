import csv
import io
import os
import requests
from pathlib import Path

ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
LISTING_URL = 'https://www.alphavantage.co/query'
DATA_DIR = Path(__file__).resolve().parent
UNIVERSE_FILE = DATA_DIR / 'universe_us.csv'


def download_listing_status() -> str:
    params = {
        'function': 'LISTING_STATUS',
        'apikey': ALPHA_VANTAGE_API_KEY,
    }
    r = requests.get(LISTING_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def build_universe(min_exchange=None) -> list[dict]:
    text = download_listing_status()
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    allowed_exchanges = set(min_exchange or ['NYSE', 'NASDAQ', 'NYSE ARCA', 'AMEX'])

    for row in rows:
        status = (row.get('status') or '').strip().lower()
        asset_type = (row.get('assetType') or '').strip().lower()
        exchange = (row.get('exchange') or '').strip().upper()
        symbol = (row.get('symbol') or '').strip().upper()

        if not symbol or status != 'active':
            continue
        if asset_type not in {'stock', 'etf'}:
            continue
        if exchange not in allowed_exchanges:
            continue
        if '^' in symbol or '/' in symbol or ' ' in symbol:
            continue

        out.append({
            'symbol': symbol,
            'name': (row.get('name') or '').strip(),
            'exchange': exchange,
            'asset_type': asset_type,
            'ipo_date': row.get('ipoDate') or '',
            'delisting_date': row.get('delistingDate') or '',
            'status': row.get('status') or '',
        })

    return out


def save_universe(rows: list[dict], path: Path = UNIVERSE_FILE) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_universe(path: Path = UNIVERSE_FILE) -> list[dict]:
    if not path.exists():
        return []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


if __name__ == '__main__':
    rows = build_universe()
    save_universe(rows)
    print(f'Saved {len(rows)} rows to {UNIVERSE_FILE}')

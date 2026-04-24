import os
import time
import requests
from typing import Optional

ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
BASE_URL = 'https://www.alphavantage.co/query'
REQUEST_SLEEP = 15


def get_global_quote(symbol: str) -> Optional[dict]:
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get('Global Quote', {})
    if not data:
        return None
    return {
        'symbol': data.get('01. symbol', symbol),
        'price': float(data.get('05. price', 0) or 0),
        'previous_close': float(data.get('08. previous close', 0) or 0),
        'change': float(data.get('09. change', 0) or 0),
        'change_percent': float((data.get('10. change percent', '0%') or '0%').replace('%', '')),
    }


def crossed_drop_threshold(quote: dict, threshold: float = 10.0) -> bool:
    return quote.get('change_percent', 0) <= -threshold

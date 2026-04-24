import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent
STATE_FILE = DATA_DIR / 'state.json'


def _today() -> str:
    return datetime.now().date().isoformat()


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding='utf-8'))


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def already_alerted(symbol: str, date_str: str | None = None) -> bool:
    state = load_state()
    date_key = date_str or _today()
    return symbol in state.get(date_key, [])


def mark_alerted(symbol: str, date_str: str | None = None) -> None:
    state = load_state()
    date_key = date_str or _today()
    state.setdefault(date_key, [])
    if symbol not in state[date_key]:
        state[date_key].append(symbol)
    save_state(state)


def clear_old_state(keep_days: int = 7) -> None:
    state = load_state()
    keys = sorted(state.keys())
    if len(keys) <= keep_days:
        return
    trimmed = {k: state[k] for k in keys[-keep_days:]}
    save_state(trimmed)

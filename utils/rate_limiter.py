"""
rate_limiter.py — Protecção anti-spam para comandos Telegram.

Limita o número de chamadas por comando por janela de tempo.
Configurável via env vars:

    RATE_LIMIT_DEFAULT=5          # pedidos por janela (default todos os cmds)
    RATE_LIMIT_WINDOW=60          # janela em segundos
    RATE_LIMIT_ANALISAR=3         # override para /analisar (caro)
    RATE_LIMIT_SCAN=1             # override para /scan (muito caro)
    RATE_LIMIT_BACKTEST=2         # override para /backtest
    RATE_LIMIT_WATCHLIST=10       # override para /watchlist (operações rápidas)

Uso:
    from utils.rate_limiter import is_allowed, rate_status

    if not is_allowed("analisar"):
        return "⚠️ Rate limit atingido. Tenta em X segundos."
"""

import os
import time
import threading
import logging
from collections import deque
from typing import Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


_WINDOW:  int = _env_int("RATE_LIMIT_WINDOW",   60)
_DEFAULT: int = _env_int("RATE_LIMIT_DEFAULT",   5)

# Limites por comando (pedidos por _WINDOW segundos)
_LIMITS: dict[str, int] = {
    "analisar":  _env_int("RATE_LIMIT_ANALISAR",   3),
    "scan":      _env_int("RATE_LIMIT_SCAN",        1),
    "backtest":  _env_int("RATE_LIMIT_BACKTEST",    2),
    "watchlist": _env_int("RATE_LIMIT_WATCHLIST",  10),
    "carteira":  _env_int("RATE_LIMIT_DEFAULT",  _DEFAULT),
    "status":    _env_int("RATE_LIMIT_DEFAULT",  _DEFAULT),
    "tier3":     _env_int("RATE_LIMIT_DEFAULT",  _DEFAULT),
    "rejeitados":_env_int("RATE_LIMIT_DEFAULT",  _DEFAULT),
    "help":      99,  # sem limite prático
}

# _timestamps[cmd] = deque de timestamps das últimas chamadas
_timestamps: dict[str, deque] = {cmd: deque() for cmd in _LIMITS}
_lock = threading.Lock()


def is_allowed(cmd: str) -> tuple[bool, Optional[int]]:
    """
    Verifica se o comando pode ser executado agora.
    Devolve (True, None) se permitido.
    Devolve (False, segundos_até_próximo) se bloqueado.
    """
    limit = _LIMITS.get(cmd, _DEFAULT)
    now   = time.monotonic()

    with _lock:
        if cmd not in _timestamps:
            _timestamps[cmd] = deque()
        dq = _timestamps[cmd]

        # Limpa timestamps fora da janela
        while dq and now - dq[0] > _WINDOW:
            dq.popleft()

        if len(dq) < limit:
            dq.append(now)
            return True, None

        # Calcula quanto falta até a entrada mais antiga sair da janela
        wait = int(_WINDOW - (now - dq[0])) + 1
        return False, max(wait, 1)


def reset(cmd: Optional[str] = None) -> None:
    """
    Reinicia os contadores.
    Se cmd=None reinicia todos.
    """
    with _lock:
        if cmd is None:
            for dq in _timestamps.values():
                dq.clear()
        elif cmd in _timestamps:
            _timestamps[cmd].clear()


def rate_status() -> str:
    """Linha de debug com estado actual do rate limiter."""
    now   = time.monotonic()
    parts = []
    with _lock:
        for cmd, dq in sorted(_timestamps.items()):
            live  = sum(1 for ts in dq if now - ts <= _WINDOW)
            limit = _LIMITS.get(cmd, _DEFAULT)
            parts.append(f"/{cmd}:{live}/{limit}")
    return f"RateLimit (janela {_WINDOW}s) — " + " · ".join(parts)


logging.debug(
    "[rate_limiter] limites: " +
    ", ".join(f"/{k}={v}req/{_WINDOW}s" for k, v in _LIMITS.items())
)

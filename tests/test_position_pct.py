"""
test_position_pct.py — testa portfolio.get_position_pct usado pelo
allocation_engine para concentration cap (12%/30%).

Sem pytest dependency. Corre como `python tests/test_position_pct.py`.
Usa monkey-patching ao cache interno do portfolio.py para evitar I/O
(GSheets ou ficheiro JSON local).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MONTHLY_BUDGET_EUR", "1050")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio as pf


def _seed_cache(liquidity: float, positions: dict) -> None:
    """Substitui o cache interno do portfolio sem tocar em I/O."""
    pf._cache["liquidity"] = liquidity
    pf._cache["positions"] = positions


def _reset_cache() -> None:
    pf._cache["liquidity"] = None
    pf._cache["positions"] = None


# ── _is_usd_ticker heuristic ──────────────────────────────────────────────────

def test_is_usd_ticker_default():
    """Tickers sem ponto → USD."""
    assert pf._is_usd_ticker("MSFT")
    assert pf._is_usd_ticker("RKLB")
    assert pf._is_usd_ticker("XPEV")
    print("  ✓ USD inference: MSFT/RKLB/XPEV → USD")


def test_is_usd_ticker_dotted():
    """Tickers com sufixo (.DE/.L/.PA) → não USD."""
    assert not pf._is_usd_ticker("EUNL.DE")
    assert not pf._is_usd_ticker("IS3N.DE")
    assert not pf._is_usd_ticker("ALV.DE")
    print("  ✓ Non-USD inference: EUNL.DE/IS3N.DE → não USD")


def test_is_usd_ticker_explicit_eur_set():
    """EUR_TICKERS set explícito — IEMA está lá."""
    # IEMA está em EUR_TICKERS no portfolio.py
    assert "IEMA" in pf.EUR_TICKERS
    assert not pf._is_usd_ticker("IEMA")
    print("  ✓ IEMA via EUR_TICKERS set → não USD")


# ── get_position_pct casos típicos ────────────────────────────────────────────

def test_position_pct_no_position():
    """Ticker que não está no portfolio → 0.0."""
    _seed_cache(
        liquidity=500.0,
        positions={"MSFT": {"total_cost": 1000.0}},
    )
    pct = pf.get_position_pct("AAPL", usd_eur=0.92)
    assert pct == 0.0
    print(f"  ✓ Sem posição: pct = {pct}")


def test_position_pct_empty_portfolio():
    """Liquidity=0 e sem posições → 0.0."""
    _seed_cache(liquidity=0.0, positions={})
    pct = pf.get_position_pct("MSFT", usd_eur=0.92)
    assert pct == 0.0
    print(f"  ✓ Portfolio vazio: pct = {pct}")


def test_position_pct_single_usd_ticker():
    """1 posição USD ($1000) + €500 cash; @0.92 → cost €920 / total €1420 ≈ 64.8%."""
    _seed_cache(
        liquidity=500.0,
        positions={"MSFT": {"total_cost": 1000.0}},  # USD
    )
    pct = pf.get_position_pct("MSFT", usd_eur=0.92)
    expected = 920.0 / 1420.0
    assert abs(pct - expected) < 0.001, f"esperado {expected:.4f}, got {pct:.4f}"
    print(f"  ✓ MSFT só com cash: {pct*100:.1f}% (esperado {expected*100:.1f}%)")


def test_position_pct_eur_ticker_no_fx():
    """ETF EUR (.DE) não aplica FX."""
    _seed_cache(
        liquidity=200.0,
        positions={"EUNL.DE": {"total_cost": 800.0}},  # EUR
    )
    pct = pf.get_position_pct("EUNL.DE", usd_eur=0.92)
    expected = 800.0 / 1000.0
    assert abs(pct - expected) < 0.001
    print(f"  ✓ EUNL.DE sem FX: {pct*100:.1f}% = 80%")


def test_position_pct_mixed_currencies():
    """Mix USD + EUR — confirma que cada um é convertido na sua moeda."""
    _seed_cache(
        liquidity=100.0,                          # €100
        positions={
            "MSFT":    {"total_cost": 1000.0},    # $1000 → €920
            "EUNL.DE": {"total_cost": 500.0},     # €500
        },
    )
    pct_msft = pf.get_position_pct("MSFT", usd_eur=0.92)
    pct_eunl = pf.get_position_pct("EUNL.DE", usd_eur=0.92)
    # total_eur = 100 + 920 + 500 = 1520
    assert abs(pct_msft - 920.0/1520.0) < 0.001
    assert abs(pct_eunl - 500.0/1520.0) < 0.001
    # Soma das pcts dos 2 + cash% = 1.0
    cash_pct = 100.0 / 1520.0
    assert abs(pct_msft + pct_eunl + cash_pct - 1.0) < 0.001
    print(f"  ✓ Mix USD+EUR: MSFT {pct_msft*100:.1f}%, EUNL {pct_eunl*100:.1f}%")


def test_position_pct_fx_clamp():
    """FX defensivo: valores absurdos clamped para [0.5, 2.0]."""
    _seed_cache(
        liquidity=100.0,
        positions={"MSFT": {"total_cost": 1000.0}},
    )
    # FX absurdo (10.0) → clamp a 2.0 → cost €2000
    pct_high = pf.get_position_pct("MSFT", usd_eur=10.0)
    assert abs(pct_high - 2000.0/2100.0) < 0.001
    # FX 0.0 → clamp a 0.92 (default) — vamos testar com 0.1 → clamp a 0.5
    pct_low = pf.get_position_pct("MSFT", usd_eur=0.1)
    assert abs(pct_low - 500.0/600.0) < 0.001
    print(f"  ✓ FX clamp: high→{pct_high*100:.1f}%, low→{pct_low*100:.1f}%")


def test_position_pct_under_12pct_cap():
    """Posição pequena → < 12% → pode continuar a comprar."""
    _seed_cache(
        liquidity=5000.0,
        positions={
            "MSFT":  {"total_cost": 1000.0},  # €920
            "ADBE":  {"total_cost": 1500.0},  # €1380
            "VWCE.DE": {"total_cost": 3000.0},  # €3000
        },
    )
    # Total = 5000 + 920 + 1380 + 3000 = 10300
    # MSFT pct = 920 / 10300 = 8.93%
    pct = pf.get_position_pct("MSFT", usd_eur=0.92)
    assert pct < 0.12, f"esperado <12%, got {pct*100:.1f}%"
    assert pct > 0.08, "deve estar na zona de slowdown (>8%)"
    print(f"  ✓ MSFT pequeno: {pct*100:.2f}% — entre 8% e 12% (slowdown zone)")


def test_position_pct_over_30pct_cap():
    """Posição grande > 30% — usado para testar HC cap."""
    _seed_cache(
        liquidity=100.0,
        positions={
            "MSFT": {"total_cost": 4000.0},  # USD → €3680
            "AAPL": {"total_cost": 1000.0},  # USD → €920
        },
    )
    # Total = 100 + 3680 + 920 = 4700
    # MSFT pct = 3680 / 4700 = 78.3%
    pct = pf.get_position_pct("MSFT", usd_eur=0.92)
    assert pct > 0.30, f"esperado >30%, got {pct*100:.1f}%"
    print(f"  ✓ MSFT grande: {pct*100:.1f}% — acima cap HC 30%")


def test_position_pct_alias_normalization():
    """Ticker alias (EUNL → EUNL.DE) deve funcionar."""
    _seed_cache(
        liquidity=200.0,
        positions={"EUNL.DE": {"total_cost": 800.0}},
    )
    pct_alias = pf.get_position_pct("EUNL", usd_eur=0.92)  # sem .DE
    pct_full  = pf.get_position_pct("EUNL.DE", usd_eur=0.92)
    assert pct_alias == pct_full, f"alias deve normalizar: {pct_alias} vs {pct_full}"
    print(f"  ✓ Alias EUNL = EUNL.DE: {pct_alias*100:.1f}%")


def test_position_pct_clamp_to_one():
    """Edge: liquidity negativa não deve produzir pct > 1.0."""
    _seed_cache(
        liquidity=-100.0,  # liquidez negativa (corner case)
        positions={"MSFT": {"total_cost": 1000.0}},
    )
    pct = pf.get_position_pct("MSFT", usd_eur=0.92)
    assert 0.0 <= pct <= 1.0, f"pct fora de [0,1]: {pct}"
    print(f"  ✓ Edge liquidity<0: pct {pct*100:.1f}% (clamped)")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_is_usd_ticker_default,
        test_is_usd_ticker_dotted,
        test_is_usd_ticker_explicit_eur_set,
        test_position_pct_no_position,
        test_position_pct_empty_portfolio,
        test_position_pct_single_usd_ticker,
        test_position_pct_eur_ticker_no_fx,
        test_position_pct_mixed_currencies,
        test_position_pct_fx_clamp,
        test_position_pct_under_12pct_cap,
        test_position_pct_over_30pct_cap,
        test_position_pct_alias_normalization,
        test_position_pct_clamp_to_one,
    ]
    failed = 0
    for t in tests:
        _reset_cache()
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed == 0:
        print(f"\n✅ {len(tests)} testes passaram")
        return 0
    print(f"\n❌ {failed}/{len(tests)} falharam")
    return 1


if __name__ == "__main__":
    sys.exit(main())

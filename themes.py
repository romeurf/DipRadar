"""
themes.py — Registo dinâmico de temas/trends de mercado.

O DipRadar não compra tickers cegamente — compra teses. Quando um sector
está em trend (fotónica, GLP-1, IA, transição energética), um dip numa
empresa dessa área tem uma tese de recuperação muito mais forte do que
um dip num sector estagnado.

Este módulo:
  1. Mantém um registo de temas activos (ficheiro JSON persistido em /data/).
  2. Permite fazer match de tickers/nomes/sectores a temas.
  3. Devolve um bonus de sizing para empresas em temas quentes.
  4. Expõe funções para o bot Telegram gerir temas via /add_theme e /themes.

Filosofia:
  - Os temas são opiniões, não factos — têm uma confiança (0–1) e data.
  - Um tema nunca força uma compra. É um multiplicador de convicção sobre
    uma decisão já tomada pelos fundamentos e pelo ML.
  - Temas podem "esfriar" — o bonus decai com o tempo se não for renovado.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Persistência em /data/ (Railway Volume) ou /tmp fallback
_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_THEMES_PATH = _DATA_DIR / "themes.json"

# Bonus máximo que um tema pode dar à alocação (multiplicador sobre amount_eur)
THEME_BONUS_MAX   = 1.20   # +20% na alocação
THEME_BONUS_MIN   = 1.05   # mínimo (+5%) para não inflar demais
THEME_DECAY_DAYS  = 180    # após 6 meses sem renovação, o bonus cai para mínimo

# ── Temas de arranque (built-in) ───────────────────────────────────────────────
# Estes são os temas iniciais pré-configurados. O utilizador pode sobrescrever
# ou adicionar via /add_theme. Os temas built-in são usados como seed se o
# ficheiro JSON não existir.

_BUILTIN_THEMES: dict[str, dict] = {
    "photonics": {
        "label":     "Fotónica / Redes Ópticas",
        "rationale": (
            "Explosão de tráfego em datacenters de IA requer largura de banda "
            "óptica massiva. Cisco, Nvidia e hiperescaladores estão a investir "
            "em interconnects ópticos — a procura de transceivers e fibra está "
            "a crescer acima das estimativas."
        ),
        "tickers":   ["CIEN", "VIAV", "IIVI", "AAOI", "LITE", "FNSR", "COHR",
                      "IPGP", "II-VI", "NPKI", "OCLR"],
        "keywords":  ["optical", "photonic", "fiber", "transceiver", "coherent",
                      "interconnect", "wavelength"],
        "sectors":   [],   # sem sector exclusivo — match por ticker/keyword
        "confidence": 0.85,
        "added":      "2026-05",
        "added_by":   "system",
    },
    "glp1": {
        "label":     "GLP-1 / Obesidade",
        "rationale": (
            "Semaglutide e tirzepatide estão a remodelar a medicina metabólica. "
            "O pipeline vai além da obesidade: NASH, insuficiência cardíaca, "
            "renal. NVO e LLY têm décadas de runway; fornecedores de CDMO e "
            "devices de injecção beneficiam do volume."
        ),
        "tickers":   ["NVO", "LLY", "RPRX", "HLVX", "ZAFG", "ESTE", "SRPT",
                      "AMRN", "PODD", "INVA"],
        "keywords":  ["obesity", "glp-1", "semaglutide", "tirzepatide", "wegovy",
                      "ozempic", "metabolic", "diabetes"],
        "sectors":   ["Healthcare"],
        "confidence": 0.90,
        "added":      "2026-01",
        "added_by":   "system",
    },
    "ai_infra": {
        "label":     "Infraestrutura de IA",
        "rationale": (
            "Capex de IA dos hiperescaladores (Microsoft, Google, Meta, Amazon) "
            "está a crescer >50% YoY. Beneficiam: GPUs, cooling, energia, "
            "redes de alta velocidade, semicondutores de HBM e HPC."
        ),
        "tickers":   ["NVDA", "AMD", "AVGO", "MRVL", "TSM", "ANET", "SMCI",
                      "VRT", "DELL", "ARM", "LRCX", "AMAT", "KLAC"],
        "keywords":  ["gpu", "inference", "hpc", "ai chip", "data center",
                      "cooling", "hbm", "accelerator"],
        "sectors":   ["Technology"],
        "confidence": 0.92,
        "added":      "2025-06",
        "added_by":   "system",
    },
    "energy_transition": {
        "label":     "Transição Energética",
        "rationale": (
            "Descarbonização forçada por regulação + queda de custos de renováveis. "
            "Solar utility-scale, offshore wind, redes eléctricas smart, "
            "hidrogénio verde e storage. Ciclo longo — 2030-2040."
        ),
        "tickers":   ["ENPH", "SEDG", "FSLR", "CWEN", "AES", "NEP", "PLUG",
                      "BE", "BEP", "ITRI", "GEV"],
        "keywords":  ["solar", "wind", "renewables", "hydrogen", "battery",
                      "grid", "storage", "clean energy"],
        "sectors":   ["Energy", "Utilities"],
        "confidence": 0.78,
        "added":      "2025-01",
        "added_by":   "system",
    },
    "defence_cybersec": {
        "label":     "Defesa & Cibersegurança",
        "rationale": (
            "Rearmamento europeu pós-Ucrânia + ameaças cibernéticas crescentes. "
            "NATO a subir gastos para 2.5% PIB. Empresas de defesa têm backlog "
            "recorde; cibersegurança é gasto incontornável para governos e empresas."
        ),
        "tickers":   ["LMT", "RTX", "NOC", "GD", "BAESY", "CRWD", "ZS", "S",
                      "PANW", "FTNT", "CYBR", "HII"],
        "keywords":  ["defence", "defense", "cyber", "security", "missile",
                      "aerospace", "firewall", "endpoint"],
        "sectors":   ["Industrials"],
        "confidence": 0.82,
        "added":      "2025-03",
        "added_by":   "system",
    },
    "quantum": {
        "label":     "Computação Quântica",
        "rationale": (
            "Ainda pré-comercial mas com catalisadores próximos: Google Willow, "
            "Microsoft topological qubits. Empresas de hardware quântico, "
            "software e criogénia. Alta especulação — apenas posições pequenas."
        ),
        "tickers":   ["IONQ", "QBTS", "RGTI", "QUBT", "IBM", "MSFT"],
        "keywords":  ["quantum", "qubit", "cryogenic", "entanglement", "topological"],
        "sectors":   [],
        "confidence": 0.55,   # especulativo
        "added":      "2025-12",
        "added_by":   "system",
    },
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load() -> dict[str, dict]:
    """Carrega temas do ficheiro JSON. Se não existir, usa os built-in."""
    if _THEMES_PATH.exists():
        try:
            data = json.loads(_THEMES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception as e:
            log.warning(f"[themes] Falha a ler {_THEMES_PATH}: {e} — a usar built-ins")
    return dict(_BUILTIN_THEMES)


def _save(themes: dict[str, dict]) -> None:
    try:
        _THEMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _THEMES_PATH.write_text(
            json.dumps(themes, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        log.error(f"[themes] Falha a persistir temas: {e}")


# ── Matching ──────────────────────────────────────────────────────────────────

def _theme_bonus_for_confidence(confidence: float, added: str) -> float:
    """Calcula bonus [THEME_BONUS_MIN, THEME_BONUS_MAX] com decay temporal.

    Temas mais antigos (sem renovação) têm bonus decaído.
    Confiança alta + tema recente → bonus máximo.
    """
    try:
        added_date = date.fromisoformat(added + "-01")
        days_old = (date.today() - added_date).days
        decay = max(0.0, 1.0 - days_old / THEME_DECAY_DAYS)
    except Exception:
        decay = 0.5

    raw = THEME_BONUS_MIN + (THEME_BONUS_MAX - THEME_BONUS_MIN) * confidence * decay
    return round(min(THEME_BONUS_MAX, max(THEME_BONUS_MIN, raw)), 3)


def get_stock_themes(
    ticker: str,
    sector: str = "",
    company_name: str = "",
) -> list[dict]:
    """Devolve lista de temas que matcham este stock (pode ser vazia).

    Match prioritário (sem fall-through ambíguo):
      1. Ticker explícito na lista do tema  → match forte
      2. Keyword do tema no nome da empresa → match nominal
      3. Sector match REQUER keyword match  → sector sozinho é demasiado
         genérico (DUOL é Technology mas não é AI infra). Por sí só
         dispara temas amplos para qualquer stock do sector.
    """
    themes = _load()
    ticker_u = ticker.upper().strip()
    sector_l = (sector or "").strip()
    name_l   = (company_name or "").lower()

    matches: list[dict] = []
    for key, t in themes.items():
        matched = False
        reason  = ""

        # 1. Match por ticker — forte, sempre vence
        if ticker_u in [x.upper() for x in t.get("tickers", [])]:
            matched = True
            reason  = "ticker"

        # 2. Match por keyword no nome
        elif name_l:
            for kw in t.get("keywords", []):
                if kw.lower() in name_l:
                    matched = True
                    reason  = f"keyword:{kw}"
                    break

        # 3. Sector match — APENAS se também já houve keyword match acima
        # (i.e., reforça o match nominal). Sector sozinho não basta para
        # evitar mismatches como DUOL → "ai_infra" só por ser Technology.

        if matched:
            bonus = _theme_bonus_for_confidence(
                t.get("confidence", 0.7),
                t.get("added", str(date.today()))[:7],
            )
            matches.append({
                "key":        key,
                "label":      t["label"],
                "rationale":  t.get("rationale", ""),
                "confidence": t.get("confidence", 0.7),
                "bonus":      bonus,
                "match_via":  reason,
                "added":      t.get("added", ""),
            })

    return matches


def get_theme_bonus(
    ticker: str,
    sector: str = "",
    company_name: str = "",
) -> float:
    """Devolve o bonus de sizing (multiplicador >= 1.0) para este stock.

    Se o stock estiver em múltiplos temas, usa o maior bonus (não acumula,
    para evitar inflacção excessiva).
    """
    matches = get_stock_themes(ticker, sector, company_name)
    if not matches:
        return 1.0
    return max(m["bonus"] for m in matches)


# ── CRUD para o bot ───────────────────────────────────────────────────────────

def add_theme(
    key: str,
    label: str,
    tickers: list[str],
    rationale: str = "",
    keywords: list[str] | None = None,
    sectors: list[str] | None = None,
    confidence: float = 0.75,
    added_by: str = "user",
) -> dict:
    """Adiciona ou actualiza um tema. Persiste no ficheiro JSON."""
    themes = _load()
    now = date.today().strftime("%Y-%m")
    themes[key.lower().strip()] = {
        "label":      label,
        "rationale":  rationale,
        "tickers":    [t.upper().strip() for t in tickers],
        "keywords":   keywords or [],
        "sectors":    sectors or [],
        "confidence": float(max(0.0, min(1.0, confidence))),
        "added":      now,
        "added_by":   added_by,
    }
    _save(themes)
    log.info(f"[themes] Adicionado/actualizado: {key} — {label} ({len(tickers)} tickers)")
    return themes[key.lower().strip()]


def remove_theme(key: str) -> bool:
    """Remove um tema pelo key. Devolve True se removido."""
    themes = _load()
    k = key.lower().strip()
    if k not in themes:
        return False
    del themes[k]
    _save(themes)
    log.info(f"[themes] Removido: {k}")
    return True


def list_themes() -> list[dict]:
    """Devolve todos os temas activos ordenados por confiança."""
    themes = _load()
    result = []
    for key, t in themes.items():
        bonus = _theme_bonus_for_confidence(
            t.get("confidence", 0.7),
            t.get("added", str(date.today()))[:7],
        )
        result.append({
            "key":        key,
            "label":      t["label"],
            "confidence": t.get("confidence", 0.7),
            "bonus":      bonus,
            "n_tickers":  len(t.get("tickers", [])),
            "added":      t.get("added", ""),
            "added_by":   t.get("added_by", ""),
            "rationale":  t.get("rationale", ""),
        })
    return sorted(result, key=lambda x: -x["confidence"])


# ── Formatação Telegram ───────────────────────────────────────────────────────

def format_theme_tag(themes_matched: list[dict]) -> str:
    """Linha compacta para o alerta de dip (ex: '🔥 AI Infra | GLP-1')."""
    if not themes_matched:
        return ""
    labels = [m["label"] for m in themes_matched[:2]]  # máx 2 para não sobrecarregar
    return "🔥 *Tema em trend:* " + " + ".join(f"_{l}_" for l in labels)


def format_themes_list() -> str:
    """Mensagem Telegram completa para o comando /themes."""
    themes = list_themes()
    if not themes:
        return "_Sem temas registados. Usa /add\\_theme para adicionar._"

    lines = ["🔥 *Temas em trend activos*", ""]
    for t in themes:
        bar = "█" * int(t["confidence"] * 5) + "░" * (5 - int(t["confidence"] * 5))
        lines.append(
            f"*{t['label']}* (`{t['key']}`)\n"
            f"  Confiança: {bar} {t['confidence']:.0%}  |  "
            f"Bonus sizing: +{(t['bonus']-1)*100:.0f}%  |  "
            f"{t['n_tickers']} tickers  |  desde {t['added']}\n"
            f"  _{t['rationale'][:120]}{'...' if len(t['rationale']) > 120 else ''}_\n"
        )
    lines.append(
        "_Usa `/add\\_theme <key> <label> <TICK1,TICK2,...>` para adicionar._\n"
        "_Usa `/remove\\_theme <key>` para remover._"
    )
    return "\n".join(lines)

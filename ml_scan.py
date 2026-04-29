"""
ml_scan.py — Bloco de inferência ML + formatação de alertas Telegram.
Chunk 5: integração do Tridente no pipeline do scan.

Importado por main.py dentro do loop run_scan().
Função principal: process_ml_candidate()
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from alert_clustering import was_alerted_recently, register_alert

# Tridente ML — carregado em lazy singleton (só quando o .pkl existir)
_trident = None
_MODEL_PATH = Path(os.getenv("MODELS_DIR", "models")) / "dip_model.pkl"


def _load_trident():
    global _trident
    if _trident is not None:
        return _trident
    if not _MODEL_PATH.exists():
        logging.warning("[ml_scan] dip_model.pkl não encontrado — modo pré-treino activo")
        return None
    try:
        from ml_pipeline import TridentModel
        _trident = TridentModel.load(_MODEL_PATH)
        logging.info("[ml_scan] Tridente carregado de %s", _MODEL_PATH)
    except Exception as e:
        logging.error("[ml_scan] Erro ao carregar Tridente: %s", e)
        _trident = None
    return _trident


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

_SIZING_TABLE = {
    # (ml_class, category)  → pct_liquidez
    # Hold Forever e Apartamento — acumulação, peso fixo independente da classe ML
    ("WIN_40", "Hold Forever"): 0.25,
    ("WIN_20", "Hold Forever"): 0.20,
    ("WIN_40", "Apartamento"):  0.20,
    ("WIN_20", "Apartamento"):  0.15,
    # Rotação — flip táctico, peso pelo sinal ML
    ("WIN_40", "Rotação"):      0.20,
    ("WIN_20", "Rotação"):      0.12,
}
_DEFAULT_PCT = 0.10


def _suggest_position_size(category: str, ml_class: str, liquidity_eur: float) -> str:
    """
    Devolve string formatada do tipo:
      "Forte (20% da liquidez — €126.00)"
    """
    # Normaliza categoria para chave da tabela
    cat_key = "Rotação"
    if "Hold" in category or "Forever" in category:
        cat_key = "Hold Forever"
    elif "Apart" in category or "🏠" in category:
        cat_key = "Apartamento"

    pct = _SIZING_TABLE.get((ml_class, cat_key), _DEFAULT_PCT)
    eur = liquidity_eur * pct

    if pct >= 0.20:
        label = "Forte"
    elif pct >= 0.15:
        label = "Moderado"
    else:
        label = "Leve"

    return f"{label} ({int(pct*100)}% da liquidez — €{eur:.2f})"


# ---------------------------------------------------------------------------
# Formatação do alerta Telegram
# ---------------------------------------------------------------------------

def _format_alert(
    symbol: str,
    price: float,
    score: float,
    category: str,
    ml_result: dict,
    sizing_str: str,
) -> str:
    """
    Formata a mensagem Telegram segundo as duas vias estruturais:
      - Rotação      → mostra Alvo Dinâmico (MFE) + Risco (MAE)
      - Hold Forever / Apartamento → oculta alvos (acumulação LP)
    """
    ml_class    = ml_result["class"]
    confidence  = ml_result["confidence"] * 100
    mfe_target  = ml_result["mfe_target"]
    mae_risk    = ml_result["mae_risk"]
    mae_flag    = ml_result["mae_risk_flag"]
    target_price = price * (1 + mfe_target / 100) if mfe_target else price
    ts = datetime.now().strftime("%d/%m %H:%M")

    is_rotacao = "Rota" in category or "🔄" in category

    if is_rotacao:
        # ── Via 1: Rotação — alvos visíveis ──────────────────────────────
        mae_alert_str = " ⚠️" if mae_flag else ""
        msg = (
            f"🔥 *ALERTA DE DIPS — {symbol}*\n"
            f"*Preço Actual:* ${price:.2f} | *Score:* {score:.0f}/100\n"
            f"*Categoria:* 🔄 Rotação\n"
            f"\n"
            f"🤖 *ML Decision:* {ml_class} (Confiança: {confidence:.0f}%)\n"
            f"🎯 *Alvo Dinâmico (Take Profit):* +{mfe_target:.1f}% (${target_price:.2f})\n"
            f"📉 *Risco (Drawdown):* {mae_risk:.1f}%{mae_alert_str} (Alerta >5%: {'Sim' if mae_flag else 'Não'})\n"
            f"💰 *Sugestão de Peso:* {sizing_str}\n"
            f"\n"
            f"_Comando rápido:_ `/buy {symbol} {price:.2f} [qtd]`\n"
            f"_⏰ {ts}_"
        )
    else:
        # ── Via 2: Hold Forever / Apartamento — alvos ocultos ─────────────
        cat_emoji = "💎" if "Hold" in category or "Forever" in category else "🏠"
        cat_label = "Hold Forever" if ("Hold" in category or "Forever" in category) else "Apartamento"
        msg = (
            f"🔥 *ALERTA DE DIPS — {symbol}*\n"
            f"*Preço Actual:* ${price:.2f} | *Score:* {score:.0f}/100\n"
            f"*Categoria:* {cat_emoji} {cat_label}\n"
            f"\n"
            f"🤖 *ML Decision:* {ml_class} (Confiança: {confidence:.0f}%)\n"
            f"🎯 *Plano de Saída:* Sem alvo táctico (Acumulação Longo Prazo)\n"
            f"💰 *Sugestão de Peso:* {sizing_str}\n"
            f"\n"
            f"_Comando rápido:_ `/buy {symbol} {price:.2f} [qtd]`\n"
            f"_⏰ {ts}_"
        )
    return msg


# ---------------------------------------------------------------------------
# Função principal — chamada pelo run_scan() em main.py
# ---------------------------------------------------------------------------

def process_ml_candidate(
    symbol: str,
    price: float,
    score: float,
    category: str,
    features,          # pd.Series ou dict com FEATURE_COLS
    liquidity_eur: float,
    send_fn,           # callable: send_fn(msg: str) -> bool
    pretrain_alert_fn = None,  # callable opcional: envia alerta clássico se sem modelo
) -> bool:
    """
    Orquestra todo o Chunk 5:
      1. Anti-clustering — ignora se ticker foi alertado há < 20 dias
      2. ML inference  — Porteiro → guilhotina NEUTRAL/LOSS → Sommelier + Gestor
      3. Formatação    — duas vias (Rotação vs Hold/Apartamento)
      4. Envio         — send_fn()
      5. Registo       — alert_clustering.register_alert()

    Devolve True se o alerta foi enviado, False caso contrário.
    """
    # ── 1. Anti-clustering ────────────────────────────────────────────────────
    if was_alerted_recently(symbol):
        logging.info("[ml_scan] %s — ignorado (clustering 20d)", symbol)
        return False

    # ── 2. Inferência ML ──────────────────────────────────────────────────────
    trident = _load_trident()

    if trident is None:
        # Modo pré-treino: modelo ainda não existe
        # Envia alerta clássico se caller forneceu fallback
        logging.info("[ml_scan] %s — modelo ausente, modo pré-treino", symbol)
        if pretrain_alert_fn is not None:
            sent = pretrain_alert_fn()
            if sent:
                register_alert(symbol, ml_class="PRE_TRAIN", score=score)
            return sent
        return False

    try:
        import pandas as pd
        if isinstance(features, dict):
            features = pd.Series(features)
        ml_result = trident.predict(features)
    except Exception as e:
        logging.error("[ml_scan] Erro na inferência %s: %s", symbol, e)
        # Fallback: envia alerta clássico sem ML
        if pretrain_alert_fn is not None:
            sent = pretrain_alert_fn()
            if sent:
                register_alert(symbol, ml_class="ML_ERROR", score=score)
            return sent
        return False

    # ── 3. Guilhotina — filtra NEUTRAL e LOSS_15 ──────────────────────────────
    ml_class = ml_result.get("class", "NEUTRAL")
    if ml_class in ("NEUTRAL", "LOSS_15"):
        logging.info(
            "[ml_scan] %s — descartado pelo Porteiro (%s conf=%.0f%%)",
            symbol, ml_class, ml_result.get("confidence", 0) * 100,
        )
        return False

    # ── 4. Position sizing ───────────────────────────────────────────────────
    sizing_str = _suggest_position_size(category, ml_class, liquidity_eur)

    # ── 5. Formatar mensagem ─────────────────────────────────────────────────
    msg = _format_alert(
        symbol=symbol,
        price=price,
        score=score,
        category=category,
        ml_result=ml_result,
        sizing_str=sizing_str,
    )

    # ── 6. Enviar ────────────────────────────────────────────────────────────
    sent = send_fn(msg)

    # ── 7. Registar no anti-clustering ───────────────────────────────────────
    if sent:
        register_alert(symbol, ml_class=ml_class, score=score)
        logging.info(
            "[ml_scan] ✅ Alerta enviado: %s | %s | conf=%.0f%% | MFE=%.1f%% | MAE=%.1f%%",
            symbol, ml_class,
            ml_result.get("confidence", 0) * 100,
            ml_result.get("mfe_target", 0),
            ml_result.get("mae_risk", 0),
        )

    return sent

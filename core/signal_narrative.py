"""
signal_narrative.py — Narrativa em linguagem natural dos sinais do DipRadar.

Transforma números em texto interpretável: o utilizador percebe PORQUE o
modelo está confiante (ou não), sem ter de decifrar métricas técnicas.

Princípio: cada sinal tem uma narrativa específica que explica o QUE significa
e o QUE implica para a decisão de compra. O texto é composto dinamicamente
com base nos valores reais, não é um template fixo.

Exemplo de output para um STRONG buy:
  "O modelo prevê que a empresa vai superar o mercado em +9% nos próximos 90
  dias, com 74% de confiança. Um executivo comprou €800k em ações esta semana
  — o sinal mais forte de que o dip é temporário. Os shorts estão a cobrir
  (-18%), o que sugere que o dinheiro inteligente já não está pessimista."

Exemplo para situação com red flag:
  "O modelo vê potencial (+5%), mas com baixa convicção (48%). A empresa
  publicou um 8-K esta semana indicando restatement de resultados anteriores.
  Red flag estrutural — este dip pode não ser temporário."
"""

from __future__ import annotations

import math
from typing import Optional


def _fmt_pct(v: float, signed: bool = True) -> str:
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _is_valid(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def generate_signal_narrative(
    # ML core
    pred_alpha_90d:  Optional[float] = None,  # alpha previsto (log-return)
    win_prob:        Optional[float] = None,  # P(win) calibrado [0,1]
    pred_drawdown:   Optional[float] = None,  # max drawdown esperado

    # Insider signals (Form 4)
    insider_recent:         Optional[float] = None,  # 0/1
    insider_amount_score:   Optional[float] = None,  # [0,1]
    insider_buy_amount_usd: Optional[float] = None,  # valor real em USD
    insider_name:           Optional[str]   = None,  # ex: "John Smith"
    insider_title:          Optional[str]   = None,  # ex: "Chief Executive Officer"
    # 8-K exact description
    eight_k_description:    Optional[str]   = None,  # descrição precisa do item code

    # 8-K event
    eight_k_score:   Optional[float] = None,  # [-1,+1]

    # Short interest
    short_trend:     Optional[float] = None,  # variação mensal

    # Technical context
    consecutive_red: Optional[float] = None,  # dias em queda
    ma_200d_ratio:   Optional[float] = None,  # price/MA200

    # Stock context
    stock_type:      str = "SPECULATIVE",   # BLUE_CHIP/QUALITY/SPECULATIVE
    ticker:          str = "",
) -> str:
    """Gera 2-4 frases em português que explicam o sinal de forma natural.

    Prioridade na narrativa:
    1. Red flags absolutos (8-K negativo grave) — mencionados primeiro
    2. Sinal principal do ML (alpha + confiança)
    3. Confirmações ou contradições dos sinais secundários
    4. Contexto técnico (se relevante)
    """
    parts: list[str] = []
    red_flags: list[str] = []

    # ── 1. Red flags (8-K grave) — mencionados imediatamente ─────────────────
    if _is_valid(eight_k_score):
        ek   = float(eight_k_score)
        desc = eight_k_description or ""
        if ek <= -0.7:
            event_str = f'"{desc}"' if desc else "evento grave (restatement ou default)"
            red_flags.append(
                f"A empresa publicou um 8-K com {event_str}. "
                f"Este dip pode ser estrutural — o Shield recomenda cautela máxima."
            )
        elif ek <= -0.4:
            event_str = f'"{desc}"' if desc else "evento negativo"
            red_flags.append(
                f"Houve um 8-K recente com {event_str}. "
                f"Confirma os fundamentos antes de entrar."
            )
        elif ek >= 0.4:
            event_str = f'"{desc}"' if desc else "evento favorável"
            parts.append(
                f"O 8-K mais recente indica {event_str} — "
                f"pode explicar a queda como ruído temporário."
            )

    # ── 2. Sinal principal ML ─────────────────────────────────────────────────
    if _is_valid(pred_alpha_90d) and _is_valid(win_prob):
        alpha_pct = float(pred_alpha_90d) * 100
        prob      = float(win_prob)

        if prob >= 0.70 and alpha_pct >= 7:
            parts.append(
                f"O modelo está muito confiante: prevê que a empresa vai superar "
                f"o mercado em *{_fmt_pct(alpha_pct)}* nos próximos 90 dias, "
                f"com *{prob:.0%}* de probabilidade."
            )
        elif prob >= 0.55 and alpha_pct >= 3:
            parts.append(
                f"O modelo prevê *{_fmt_pct(alpha_pct)}* de alpha em 90 dias "
                f"({prob:.0%} de confiança) — sinal positivo mas moderado."
            )
        elif prob < 0.40 or alpha_pct < 0:
            parts.append(
                f"O modelo não está otimista: prevê *{_fmt_pct(alpha_pct)}* "
                f"de alpha com apenas *{prob:.0%}* de confiança. "
                f"O Shield não recomenda entrada neste momento."
            )
        else:
            parts.append(
                f"O modelo prevê *{_fmt_pct(alpha_pct)}* de alpha em 90 dias "
                f"({prob:.0%} de confiança)."
            )

        if _is_valid(pred_drawdown) and pred_drawdown < -0.10:
            dd_pct = float(pred_drawdown) * 100
            parts.append(
                f"O modelo prevê que antes de recuperar podes ver uma queda "
                f"adicional de até *{dd_pct:.0f}%* — tem liquidez para reforçar."
            )

    # ── 3. Insider buying ─────────────────────────────────────────────────────
    if _is_valid(insider_amount_score) and float(insider_amount_score) >= 0.3:
        score_val  = float(insider_amount_score)
        amount_usd = float(insider_buy_amount_usd) if _is_valid(insider_buy_amount_usd) and insider_buy_amount_usd else 0

        # Formatar o montante real se disponível
        if amount_usd >= 1_000_000:
            amount_str = f"*${amount_usd/1_000_000:.1f}M*"
        elif amount_usd >= 100_000:
            amount_str = f"*${amount_usd/1_000:.0f}k*"
        elif amount_usd > 0:
            amount_str = f"*${amount_usd:,.0f}*"
        else:
            amount_str = "uma quantidade significativa"

        # Nome e cargo do insider
        _name  = (insider_name or "").strip()
        _title = (insider_title or "").strip()
        if _name and _title:
            who = f"*{_name}* ({_title})"
        elif _name:
            who = f"*{_name}*"
        elif _title:
            who = f"o *{_title}*"
        else:
            who = "um executivo"

        if score_val >= 0.8:
            parts.append(
                f"{who} comprou {amount_str} em ações recentemente "
                f"(Form 4 SEC) — raramente insiders apostam este montante "
                f"se não tiverem convicção de que o dip é temporário."
            )
        elif score_val >= 0.5:
            parts.append(
                f"{who} comprou {amount_str} em ações recentemente "
                f"(Form 4 SEC). Quem conhece melhor a empresa está a comprar."
            )
        else:
            parts.append(
                f"{who} comprou {amount_str} em ações recentemente — sinal positivo."
            )
    elif _is_valid(insider_recent) and float(insider_recent) > 0:
        parts.append(
            "Houve compras de insider recentes (Form 4 SEC), "
            "o que sugere que a queda é vista como temporária internamente."
        )

    # ── 4. Short interest trend ───────────────────────────────────────────────
    if _is_valid(short_trend):
        st = float(short_trend)
        if st <= -0.20:
            parts.append(
                f"Os shorts estão a cobrir posições (*{st*100:.0f}%* este mês) — "
                "o dinheiro inteligente pessimista está a sair, o que é "
                "frequentemente um sinal de reversão próxima."
            )
        elif st >= 0.30:
            parts.append(
                f"O short interest aumentou *{st*100:.0f}%* este mês. "
                "Muitos estão a apostar na descida — se recuperar, "
                "o squeeze pode amplificar o movimento."
            )

    # ── 5. Contexto técnico ───────────────────────────────────────────────────
    if _is_valid(consecutive_red) and float(consecutive_red) >= 5:
        days = int(float(consecutive_red))
        parts.append(
            f"A ação caiu *{days} dias consecutivos* — "
            "padrão típico de capitulação onde o sentimento está no extremo negativo."
        )

    if _is_valid(ma_200d_ratio):
        ratio = float(ma_200d_ratio)
        if ratio <= 0.75:
            pct_below = (1 - ratio) * 100
            parts.append(
                f"Está *{pct_below:.0f}%* abaixo da média de 200 dias — "
                "território de dip profundo, histórica e estatisticamente "
                "zona de risco/retorno favorável para qualidade."
            )

    # ── 6. Stock type context ─────────────────────────────────────────────────
    if stock_type == "BLUE_CHIP" and red_flags:
        parts.append(
            "Mesmo sendo uma empresa de qualidade, os red flags acima "
            "não devem ser ignorados — até blue chips sofrem quebras estruturais."
        )
    elif stock_type == "BLUE_CHIP" and not red_flags:
        parts.append("Empresa de qualidade sólida — dip com tese intacta.")

    # ── Composição final ──────────────────────────────────────────────────────
    # Red flags primeiro, depois sinal principal, depois confirmações
    all_parts = red_flags + parts
    if not all_parts:
        return ""

    return "\n".join(all_parts)


def generate_8k_veto(eight_k_score: Optional[float]) -> Optional[str]:
    """Se o 8-K indica evento estruturalmente negativo, devolve mensagem de veto.

    Retorna None se o 8-K for neutro/positivo ou se não houver dados.
    O veto é usado pelo scan para mudar o verdict de COMPRAR para MONITORIZAR.
    """
    if not _is_valid(eight_k_score):
        return None
    ek = float(eight_k_score)
    if ek <= -0.75:
        return (
            "8-K NEGATIVO GRAVE: restatement ou default detectado. "
            "O Shield elevou o alerta para MONITORIZAR — verificar contexto completo."
        )
    return None

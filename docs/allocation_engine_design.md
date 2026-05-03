# Allocation Engine — Design Document

> **Status**: design only (não codificado). Documento vivo, revisitar à medida que o ML evolui.
> **Owner**: pg45861
> **Última actualização**: 2026-05-02

## 1. Objectivo

Transformar o DipRadar de **detector de oportunidades** em **decisor de tamanho de posição**. O motor recebe um sinal (alerta dip, watchlist hit, rebalanceamento mensal) e responde com:

1. **Categoria** (ETF core / apartamento / growth / flip / pass).
2. **Sizing absoluto** em euros.
3. **Trigger** (porquê agora, e em que condições reavaliar).
4. **Exit policy** (target ou regra de deterioração de tese).

## 2. Perfil do investidor (input fixo)

| Parâmetro | Valor |
|---|---|
| Idade | 25 anos |
| Horizonte | 30+ anos |
| Tolerância ao risco | Alta |
| Cash flow líquido investível | **€1 050/mês** |
| Buffer de segurança | 6 meses de despesas (não tocar) |
| Currency | EUR (mas opera em USD/EUR via broker) |
| Restrições | sem alavancagem; sem opções complexas; tax loss harvesting permitido |

## 3. Categorias de capital

Cada euro investido pertence a **uma e só uma** categoria. As regras de saída diferem por categoria.

### 3.1 ETF Core (anchor)
- Exemplos: VWCE, IWDA (all-world), depois adicionar EIMI / VFEM (emerging markets) — meta de 10 % do core para EM.
- Comportamento: **DCA mensal automático**, peso fixo no plano.
- Exit: **nunca** (a menos que substituído por ETF mais barato/diversificado).
- Por que existe: âncora passiva. Reduz behavioral risk de excesso de stock picking.

### 3.2 Apartamento (long-run quality / dividend / value)
- Exemplos do investidor: NVO, MSFT (quando abaixo da SMA200w), JNJ, etc.
- Comportamento: **acumular em dips agressivos** quando ML score `WIN` ou `WIN_STRONG` E condição de qualidade + valor.
- Exit: **só por deterioração de tese** (ver § 6).
- Sub-categoria: **Long-run quality watcher** — empresa blue-chip abaixo de SMA200w → trigger automático.

### 3.3 Growth picks (active stock picking)
- Tese: ML detecta dip + score alto + thesis-fit do utilizador.
- Comportamento: tamanho mais pequeno (single-name risk), com regra de pyramiding em fold-throughs.
- Exit: target dinâmico (+15 a +25 % short term) OU stop-on-thesis-break.

### 3.4 Flip (tactical / event-driven)
- Tese: dip muito severo + recuperação esperada em 60 dias (modelo v3 directamente).
- Comportamento: posições pequenas, alta rotação. Sai a +15 % ou em time decay (60 d).
- Exit: target ou time-stop. **Nunca** vira "apartamento" sem reclassificação manual.

### 3.5 Cash buffer (estrutural)
- 5 a 10 % do património líquido investível em money-market ETF (e.g. XEON) para munições em crashes (VIX > 30).
- Não conta para "investido" mas é parte do plano.

## 4. Pesos-alvo (allocation policy)

Definir **dois modos** que se alternam por regime de mercado.

| Modo | Trigger | ETF core | Apartamento | Growth | Flip | Cash |
|---|---|---|---|---|---|---|
| **Default** | VIX < 25, macro_score > 0 | 50 % | 25 % | 15 % | 5 % | 5 % |
| **Risk-on bias** | VIX < 15, macro_score > 0.5 | 45 % | 25 % | 18 % | 7 % | 5 % |
| **Defensive** | VIX > 25 OU macro_score < -0.3 | 55 % | 30 % | 5 % | 0 % | 10 % |
| **Crash buying** | VIX > 35 | desactiva DCA, despeja cash em flip+growth (pre-aprovação manual) | | | | |

Os pesos são **alvos** dos novos €1 050/mês, não rebalance forçado. Se o portfolio drifta acima do peso target → próximo mês alimenta os outros.

## 5. Sinais e triggers de entrada

| Sinal | Camada | Categoria sugerida | Sizing |
|---|---|---|---|
| ML `WIN_STRONG` (score > 0.10) + qualidade alta | apartamento | up to 8 % do mensal | máx €100/posição |
| ML `WIN` (score > 0.05) + qualidade média | growth | 4 a 6 % do mensal | máx €60/posição |
| ML `WEAK` + dip muito severo (drawdown 52w > 35 %) | flip | 2 a 3 % do mensal | máx €30/posição |
| Long-run quality (MSFT/JNJ abaixo SMA200w) | apartamento | up to 12 % do mensal | máx €120/posição |
| DCA mensal | ETF core | peso fixo | conforme tabela § 4 |

**Cap por nome**: máximo 10 % do património em uma só posição (não-ETF). Cap por sector: 30 %.

## 6. Deterioração de tese — gatilhos de saída

Para apartamento e ETF core. Saída só quando:

1. **Fundamentos**: revenue_growth turn-negative durante 2 trimestres consecutivos.
2. **Margem**: gross_margin caiu > 500 bps em ano-sobre-ano.
3. **Endividamento**: de_ratio > 2.0 OU degradação de credit rating.
4. **Capital allocation**: dividendo cortado / buybacks parados sem razão de M&A.
5. **ML signal**: score persistentemente `NO_WIN` durante 6 meses (gatilho informativo, não vinculativo).

**Nenhuma saída por preço.** Se a tese aguenta, dips são oportunidade de adicionar.

## 7. Integração com o ML actual (v3)

O modelo v3 dá `pred_up` (retorno máximo previsto a 60 d) e `pred_down` (drawdown previsto, mas com rho ≈ 0 por agora). O motor de alocação consome:

- `pred_up` (post-Fix A) → score directo. Threshold em 0.05 (`WIN`) e 0.10 (`WIN_STRONG`).
- `win_prob` (post-Fix B, calibrado) → probabilidade real de winning.
- `pred_down` → guardrail informativo. Se `pred_down < -0.20` → desclassificar para flip ou pass.

**Quando rho_down passar para < -0.10** (após retreino com features melhoradas), reactivar `score = pred_up / |pred_down|` opcional.

## 8. Operacionalização (mensal)

```
Dia 25 do mês →
   1. Recolher cash injectado (€1050)
   2. Avaliar regime (VIX + macro_score) → escolher modo (§ 4)
   3. Listar candidatos:
      - ETF core (sempre, peso fixo)
      - Apartamento (de score histórico do mês + watchlist long-run-quality)
      - Growth (top-K do ML em score WIN_STRONG)
      - Flip (top-K em ML WIN, drawdown 52w > 30%)
   4. Aplicar caps por nome / sector
   5. Gerar plano: lista de [ticker, categoria, €amount, exit_rule]
   6. Enviar plano via Telegram para aprovação manual (no início)
   7. Após aprovação: agendar ordens via broker API (futuro)
```

## 9. Saídas do motor (formato)

```python
@dataclass
class AllocationDecision:
    ticker:        str
    category:      str            # "ETF_CORE" | "APARTAMENTO" | "GROWTH" | "FLIP"
    amount_eur:    float
    rationale:     str
    exit_rule:     str            # "NEVER" | "THESIS_BREAK" | "TARGET_+15%" | "TIME_60D"
    target_price:  float | None
    confidence:    str            # "Alta" | "Média" | "Baixa"
    ml_metrics:    dict           # pred_up, pred_down, win_prob, score
```

E um sumário Telegram:

```
📊 *Plano €1050 — Maio 2026*  (modo Default | VIX 17.5 | macro +0.4)

🟦 ETF Core (50%) — €525
  • VWCE: €425 (DCA)
  • EIMI: €100 (DCA)

🟪 Apartamento (25%) — €263
  • NVO: €120 (drop -8% / WIN_STRONG / never sell)
  • MSFT: €143 (abaixo SMA200w / long-run-quality / never sell)

🟧 Growth (15%) — €157
  • SHOP: €60 (ML 0.082 / target +18%)
  • NU:   €60 (ML 0.071 / target +15%)
  • SOFI: €37 (ML 0.066 / target +15%)

🟨 Flip (5%) — €52
  • XPEV: €30 (severe dip / 60d hold / target +15%)
  • RIVN: €22 (oversold / 60d hold)

🟩 Cash buffer (5%) — €52 → XEON

✅ Approve? (responde 'ok' ou ajusta)
```

## 10. Roadmap de implementação

**Fase 0 (concluído)**: ML v3 funcional, score baseado em pred_up.

**Fase 1 (próxima)**: motor read-only — só calcula plano e envia para Telegram. Aprovação 100 % manual. Sem execução.

**Fase 2**: log de decisões (qual aprovaste / rejeitaste, com porquê). Permite analisar o teu próprio bias.

**Fase 3**: integração broker (Trading 212 / DEGIRO / IBKR) via API para execução de ordens aprovadas.

**Fase 4**: feedback loop. As decisões aprovadas alimentam o ML como labels reforçadas (positivo) ou rejeitadas (negativo).

## 11. Open questions

1. **Broker**: qual? T212 tem API limitada. DEGIRO não tem. IBKR sim. Decidir antes da Fase 3.
2. **Tax loss harvesting**: implementar? Em PT requer planeamento de mais valias. Defer para Fase 4.
3. **ETF emerging markets**: VFEM (FTSE) ou IEMG (MSCI)? Ler tracking error.
4. **Long-run-quality watchlist**: lista hardcoded vs dinâmica? Sugiro hardcoded inicial (10 nomes blue-chip) com revisão semestral.
5. **Crash buying mode**: trigger automático ou aprovação manual? Recomendo manual (psicologia importa em VIX > 35).

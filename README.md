# 📡 DipRadar — Honest Quality-Compounder Screener

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)

> **Honest watchlist** for quality-compounder investing. Filters quality businesses in material drawdown with near catalyst — and surfaces them WITHOUT claiming edge ranking. Execution is always manual by the user in T212. Tax-aware (IRS Jovem).
>
> The system was ML-ranked through Q2 2026. After backtest validation (F.5) showed the rule-based dislocation_z scorer failed its binding gate (top-decile −1.09pp vs SPY, worst-fold −22.48pp in 2025), the production default switched to a watchlist that doesn't pretend to know which name has the most alpha. See [history](#-history-why-watchlist-and-not-ranking).

---

## 📋 What ships today

| Component | Mode | Role |
|---|---|---|
| **Watchlist scorer** | ✅ Default (`DIPRADAR_SCORING_MODE=watchlist`) | Quality + DD ≥ 25% + catalyst ≤ 30d. Sector-relative shown as column, **not** used to filter. |
| **CORE radar** | ✅ Functional, rule-based | 230 curated quality compounders; provides the signals the watchlist filters |
| **DIP scanner** | ✅ Functional | Detects abrupt drops vs sector; ML inert (`DIPRADAR_USE_ML_SCORE=0` default) |
| **Sector rotation** | ✅ Functional | 24 ETFs ranked monthly, tax-aware via IRS Jovem 7% gate |
| **Momentum scanner** | ✅ Functional (rule-based) | Breakout detector (return_20d ≥ 12%, RSI 52-78, beats sector) |
| **Momentum trainer** | ❌ **RETIRED** | Anti-predictive (CV IC = −0.0042). Stub raises clear retirement error. |
| **Conviction radar** | ✅ Functional | ATR trailing-stop monitor for CORE/MOMENTUM/DIP holdings |
| **Market regime detector** | ✅ Functional | 0-100 score → FROTHY/NEUTRAL/PANIC + cash buffer target |
| **Composite z-score** | ⚠️ Legacy (fallback) | Heuristic z (`DIPRADAR_SCORING_MODE=composite`) |
| **Dislocation z-score** | ⚠️ Experimental | Rule-based 7-signal composite that **failed** F.5 gate (`DIPRADAR_SCORING_MODE=dislocation`) |

**One env switch controls every scoring path**: `/scan_now`, `/analisar`, watchdog cron, `/cash`. Never alert with one logic and respond with another.

---

## 🤖 Commands

### Honest scoring (default = watchlist)

| Command | What it does |
|---|---|
| `/scan_now` | Watchlist now: list of qualifiers, DD-sorted, with sector-relative column + news headlines. **No ranking by alpha.** |
| `/analisar <TICKER>` | Full analysis + watchlist criteria check (✅/❌ per criterion). |
| `/regime` | Market regime score + cash-buffer guidance (FROTHY/NEUTRAL/PANIC). |
| `/cash_status` | Compact regime + portfolio + concrete action recommendation. |
| `/cash <eur>` | Allocation proposal. In watchlist mode = equal-weight across qualifiers, concentration cap 30%, regime cash buffer first. |
| `/deployed TICKER AMOUNT [reason]` | Log a real deploy after T212 execution (compares to last `/cash` proposal). |
| `/deploy_history [N]` | Recent `/deployed` history. |
| `/deploy_track [N]` | User picks vs system top picks, live forward returns. |

### Portfolio

| Command | What it does |
|---|---|
| `/sync_portfolio` | Snapshot of current holdings grouped by bucket. |
| `/sync_portfolio TICK:shares:price:BUCKET[:DATE]` | Bulk import positions. |
| `/sync_portfolio --update-avg TICK:...` | Overwrite avg price + shares. |
| `/carteira` | Real-time portfolio snapshot. |
| `/buy TICK PRICE SHARES` / `/sell TICK PRICE [SHARES]` | Manual position updates. |

Buckets: `CORE` (hold-for-life) · `ROTATION` (sector ETFs) · `MOMENTUM` (breakout holdings) · `DIP` (active recovery)

### Sectors and rotation

| Command | What it does |
|---|---|
| `/sectors` | Rotation engine ranking real-time. |
| `/sectors TICKER` | ETF detail. |
| `/sectors holdings` | Your ETFs status. |
| `/admin_calibrate_rotation [years]` | Calibrate rotation alpha (linear / poly / buckets / lightgbm). |

### Conviction (exit signals)

| Command | What it does |
|---|---|
| `/conviction list` | Active positions with trailing stops. |
| `/conviction TICKER` | Deep dive. |
| `/conviction close TICKER PRICE [reason]` | Close a position. |
| `/conviction transition TICKER TYPE` | Change bucket (DIP→MOMENTUM→CORE). |

### Admin / introspection

| Command | What it does |
|---|---|
| `/admin_check_config` | Models / scorer / cron state. |
| `/admin_core_debug TICKER` | Debug Core score for a ticker. |
| `/admin_pipeline_status` | Last cron firings + retrains. |
| `/health` | System dashboard. |
| `/backtest` | Backtest summary. |
| `/admin_disk` | Disk usage by artifact. |
| `/admin_retrain` | Retrain DIP model (ML inert in watchlist mode but command preserved). |
| `/admin_momentum_retrain` | **RAISES** retirement error (Momentum ML retired). |

---

## 💰 `/cash` in watchlist mode

```
/cash 1050
```

1. Reads current portfolio + cash inflow + market regime.
2. Computes regime cash buffer (e.g. NEUTRAL → 20% of inflow held).
3. Pulls the watchlist (names that pass quality + DD ≥ 25% + catalyst ≤ 30d).
4. Allocates **equal-weight** across qualifiers, respecting 30% per-ticker concentration cap.
5. Shows each qualifier with full criteria breakdown + **vs sector** column for visual triage.
6. Persists `watchdog_state.json` so the daily watchdog cron knows what's held.

**No edge-weighted Kelly. No alpha promise. Equal weight + concentration cap.** The user decides which names have a story; the system filters for the universe that's worth considering.

Empty watchlist day → suggestion is "hold cash" — there's no list to allocate against.

---

## 🚨 Auto-alerts (watchdog cron)

| Cron | Time (Lisbon) | What |
|---|---|---|
| DIP scanner hourly | 15h, 16h, 18h, 19h, 20h Mon-Fri | DIP alerts (intra-session) |
| US EOD scan | 21h15 Mon-Fri | DIP + thesis degradation + recovery + weekly DIP scan |
| Momentum scanner | 17h50 + 21h30 Mon-Fri | Rule-based breakout alerts |
| Conviction daily | 22h05 Mon-Fri | CORE/MOMENTUM/DIP trailing-stop monitor |
| **Watchlist watchdog** | **18h00 Mon-Fri** | **Push notification when a NEW name enters the watchlist (and held_cash > €100)** |
| Heartbeat | 09h00 Mon-Fri | Bot alive check |
| Daily alert reset | 06h00 daily | Reset alerted_today.json |
| Backup | 23h30 daily | data/ backup |

The **watchlist watchdog** is the answer to "auto-alert when something becomes a good investment":
- Runs daily at 18h Lisbon
- Scans the CORE universe through the watchlist criteria
- If a ticker NOT in `deployed_tickers` and NOT in `notified_tickers` qualifies → push notification with the full criteria breakdown + headlines
- "Good investment" = passes the watchlist filter, nothing more, nothing less. No alpha promise.

---

## 🏛️ Architecture (post-F.5 reality)

### Scoring dispatcher

`core/scoring_dispatch.py` — `ScoringSession` reads `DIPRADAR_SCORING_MODE` ONCE per scan. Three modes:

```python
DIPRADAR_SCORING_MODE=watchlist     # default — honest filter
DIPRADAR_SCORING_MODE=composite     # legacy heuristic z-score
DIPRADAR_SCORING_MODE=dislocation   # experimental, failed F.5 gate
```

Backward-compat: `DIPRADAR_USE_DISLOCATION=0/1` still respected for one more release (will be removed).

### Honest watchlist

`core/honest_watchlist.py` — three filter criteria, sector-relative as **display column**:

| Criterion | Threshold | Source |
|---|---|---|
| Quality gate | revenue_growth ≥ −10%, gross_margin ≥ 25%, FCF ≥ 0 (or investment phase) | `core_radar._score_core_dip` |
| Drawdown depth | dd_from_52w_high ≤ −25% | price history |
| Catalyst proximity | days_to_earnings in [−10, +30] | yfinance earnings calendar |

Sector-relative (60d) is computed via `data/sector_relative.py` and **shown** in every entry but **never used to exclude**. Per the design discipline: removing it would re-shut the door on absolute drops (ADBE-style); showing it lets the user visually triage "fell with market" (sector-rel ≈ 0) from "fell alone" (sector-rel deeply negative).

### Tax engine (`core/tax_engine.py`)

- IRS Jovem 75% exemption default (effective CGT 7%)
- Override via `EFFECTIVE_CGT_PCT` env var
- `should_rotate(unrealized_gain, current_score, target_score)`:
  - Override structural: `exit_strength > 70`, `secular_risk > 60`, `current_score < 30`
  - Loss positions → always rotate (loss = tax deduction)
  - Profit → `score gap ≥ tax_cost_pct × TAX_FRICTION_MULTIPLIER (500)`

### Market regime (`core/market_regime.py`)

0-100 score from 5 components (VIX percentile, SPY vs MA200d, quality_dd, sectoral divergence, SPY 3m). Output:
- **FROTHY** (broad market expensive) → hold cash buffer high
- **NEUTRAL** (default) → normal deploy
- **PANIC** (generational buying zone) → deploy aggressively

Surfaced in `/cash` header and used to compute the cash buffer in watchlist allocation.

### Sector rotation (`core/sector_radar.py`)

- 24 ETFs (sector SPDR + thematic)
- Backtest validated: +15pp alpha vs SPY in 4 years with monthly rebalance
- Calibration multi-method: linear → polynomial → buckets → lightgbm (best CV r²)
- US tickers for data, UCITS mapping for deploy

### Conviction radar (`core/conviction_radar.py`)

- ATR trailing stops (12-21% depending on volatility)
- Structural decline + 8-K event triggers
- Daily check 22h05 Mon-Fri
- Auto-sync with `/sync_portfolio`

### Dislocation scorer (experimental, failed)

`core/dislocation_score.py` — 7 sub-scores (estimate revisions, guidance, drop attribution, catalyst, valuation, quality, governance) composed into a z-score. Built per the plan to differentiate "fear-driven dislocation" from value traps. **F.5 backtest verdict**: REVERT — top-decile underperformed SPY by 1.09pp, worst-fold −22.48pp in 2025 (AI-led regime broke mean reversion). Code retained behind env flag for future re-validation if PIT data improves (e.g., subscribing to Tiingo Fundamentals).

---

## ⚙️ Env vars

### Required

| Var | Description |
|---|---|
| `TELEGRAM_TOKEN` | Bot token |
| `TELEGRAM_CHAT_ID` | Chat ID |
| `MONTHLY_BUDGET_EUR` | Monthly inflow (e.g. `1050`) |
| `TZ` | `Europe/Lisbon` |

### Scoring + allocation

| Var | Default | Description |
|---|---|---|
| `DIPRADAR_SCORING_MODE` | `watchlist` | `watchlist` (ship) / `composite` (legacy) / `dislocation` (experimental) |
| `DIPRADAR_USE_DISLOCATION` | unset | Legacy flag (=0 → composite, =1 → dislocation); deprecated, prefer `DIPRADAR_SCORING_MODE` |
| `DIPRADAR_USE_ML_SCORE` | `0` | ML scoring inert by default (set `=1` for legacy DIP ML path) |
| `EFFECTIVE_CGT_PCT` | `0.07` | Effective CGT (IRS Jovem 75% = 7%) |
| `SECTOR_CONCENTRATION_CAP` | `0.35` | Sector cap (legacy reference) |

### Risk + execution

| Var | Default | Description |
|---|---|---|
| `POSITION_STOP_LOSS_PCT` | `0.12` | DIP positions stop-loss |
| `MIN_DIP_SCORE` | `50` | Minimum DIP score for alert |
| `INLINE_TICKER_TIMEOUT` | `25` | Per-ticker download timeout |

### APIs

| Var | Source | Description |
|---|---|---|
| `TIINGO_API_KEY` | api.tiingo.com | EOD prices |
| `FMP_API_KEY` | financialmodelingprep.com | Analyst estimates (free tier, used by experimental dislocation mode) |
| `FRED_API_KEY` | fred.stlouisfed.org | Macro: recession probability |

---

## 🚀 Cold start

1. Configure Railway env vars (above).
2. Deploy → bot starts automatically.
3. Telegram: `/admin_check_config` to validate config.
4. First time: `/sync_portfolio TICK:shares:price:BUCKET` to register existing positions.

### Monthly cadence

1. Receive inflow (~€1050).
2. `/cash 1050` → watchlist proposal (equal-weight across qualifiers).
3. Execute manually in T212.
4. `/deployed TICKER AMOUNT reason` for each actual purchase (compares to proposal).
5. Watchdog cron at 18h Lisbon notifies if a new name enters the watchlist mid-month.

---

## 📚 History: why watchlist and not ranking?

DipRadar was built around an ML-ranked alpha thesis. Through Q2 2026 we shipped:

- DIP scanner with ML alpha prediction (CV IC = +0.063 — weak)
- Core radar with ML alpha (IC = +0.119 — better but unvalidated at top decile)
- Momentum ML model (CV IC = −0.0042 — **anti-predictive**, kept for retraining)
- Sector rotation with calibrated linear/poly/bucket alpha (r² ≤ 0.007 — noise)
- Composite z-score combining quality + drawdown + valuation
- Dislocation z-score (rule-based 7-sub-score composite)

The **F.5 backtest gate** (binding pre-registered: top-decile 6m return ≥ SPY + 500bps AND hit-rate ≥ 65% AND worst-fold ≥ −500bps, run on PIT-clean spine 2021-2026) produced:

- top-decile_6m_excess = **−1.09pp** (need ≥ +5pp)
- hit_rate = 59.3% (need ≥ 65%)
- worst-fold = **−22.48pp** in 2025Q4 (need ≥ −5pp)

All three criteria failed. The signals that distinguish "fear" from "deterioration" — estimate revisions, news guidance, forward valuation — need PIT historical data we don't have freely. Without them, ranking confused dislocations with value traps. The 2025 AI-led regime broke mean-reversion entirely.

Per the binding rule: **REVERT**. Default scoring switched to watchlist. The honest version filters for quality + drawdown + catalyst and shows sector-relative as a column for the user to judge. No alpha claim, no edge ranking, no pretending to know which name will win. **Útil sem mentir.**

The ML files are retained for one transition release; the Momentum trainer is fully retired (stubbed).

---

## 📁 Layout

```
DipRadar/
├── main.py                          # Entry + APScheduler crons
├── bot_commands.py                  # Telegram dispatcher
├── state.py                         # JSON persistence
├── config.py                        # Central config
├── core/
│   ├── honest_watchlist.py         # Watchlist (default scorer)
│   ├── scoring_dispatch.py          # One env switch, 3 modes
│   ├── dislocation_score.py         # Experimental rule-based scorer (F.5 failed)
│   ├── composite_score.py           # Legacy heuristic z-score
│   ├── core_radar.py                # Quality compounder screening
│   ├── sector_radar.py              # Sector rotation
│   ├── conviction_radar.py          # Exit signals
│   ├── market_regime.py             # FROTHY/NEUTRAL/PANIC
│   └── tax_engine.py                # IRS Jovem rotation gate
├── data/
│   ├── fundamental_signals.py       # SEC EDGAR + Form 4 + 8-K
│   ├── sector_relative.py           # Ticker vs sector ETF return
│   ├── valuation_percentile.py      # P/E vs own 3y range
│   ├── tiingo_client.py             # EOD prices
│   └── validation_cache/            # F.5 backtest cache (committed for reproducibility)
├── ml/                              # Inert ML (DIPRADAR_USE_ML_SCORE=0 default)
├── ml_training/                     # Training infra (retired but kept)
├── momentum_radar/
│   ├── scanner.py                   # Rule-based breakout detector (FUNCTIONAL)
│   ├── trainer.py                   # RETIRED stub (anti-predictive)
│   └── scripts/build_dataset.py     # RETIRED stub
├── scanners/                        # DIP scanner + alert DB
├── portfolio/                       # Position tracking
├── utils/
│   ├── news_headlines.py            # yfinance news with as_of_date PIT filter
│   ├── http_timeout.py              # Monkey-patch yfinance to add timeouts
│   └── ...
├── scripts/                         # Build/calibrate/validate one-shot scripts
└── tests/                           # pytest suite (55 tests post-watchlist ship)
```

Module docstrings carry the technical detail per file.

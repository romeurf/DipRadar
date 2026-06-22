"""One-shot script to refactor /cash command block. Run once, then delete."""
from pathlib import Path

src = Path("bot_commands.py")
text = src.read_text(encoding="utf-8")

START_MARKER = (
    "                # ╔════════════════════════════════════════════════════════╗\n"
    "                # ║ PASS 2 — Build candidate BUY queue, rank by ROI        ║\n"
    "                # ╚════════════════════════════════════════════════════════╝\n"
)

END_MARKER = (
    "                # ── Cash bucket: whatever's left ───────────────────────\n"
    "                cash_bucket = max(0.0, remaining)\n"
)

start_idx = text.find(START_MARKER)
if start_idx == -1:
    raise SystemExit("START_MARKER not found")
end_idx = text.find(END_MARKER, start_idx)
if end_idx == -1:
    raise SystemExit("END_MARKER not found")
end_idx_after = end_idx + len(END_MARKER)

new_block = """                # ╔════════════════════════════════════════════════════════╗
                # ║ PASS 2 — Unified queue via composite z-score          ║
                # ╚════════════════════════════════════════════════════════╝
                # Screener mode: Kelly-Full + ML alpha promises retired.
                # Composite z-score = quality_z + dd_z + valuation_z, computed
                # against the CORE universe distribution.
                #   CORE picks → CORE horizon (52w drawdown + valuation vs MA200w)
                #   DIP alerts → DIP  horizon (20d return + valuation vs MA200w)
                # Rotation kept on equal-share among valid top-3 (ETFs aren't in
                # CORE universe → composite_z not applicable).
                #
                # Sizing pipeline: softmax(z × 2) with threshold z>0.3,
                # min alloc €100, concentration cap 30%, overflow redistributed.
                # No alpha promises, no expected return numbers — just "how
                # compelling vs universe right now" + transparent sizing.
                candidates:    list[dict] = []
                hold_actions:  list[dict] = []
                sell_actions_to_show = sell_actions
                ranked_deploys: list[dict] = []
                buy_actions = []

                # Build universe stats once (cached daily to /data/universe_stats.json)
                try:
                    from core.composite_score import (
                        build_universe_stats, composite_from_signals, softmax_sizing,
                    )
                    universe_stats = build_universe_stats()
                except Exception as _e:
                    info_lines.append(
                        f"⚠️ Composite stats failed: `{_e}` — sem ranking unificado"
                    )
                    universe_stats = {}

                # ── Rotation candidates (equal-share among valid top-3) ─
                if not skip_rotation and rotation_top3:
                    ROTATION_MIN_SCORE = 50.0
                    valid_rotation = [
                        r for r in rotation_top3
                        if float(r["score"]) >= ROTATION_MIN_SCORE
                    ]
                    for r in rotation_top3:
                        if float(r["score"]) < ROTATION_MIN_SCORE:
                            info_lines.append(
                                f"⏭️ `{r['ucits']}` ROTATION skipped — "
                                f"score {r['score']:.1f} < {ROTATION_MIN_SCORE:.0f}"
                            )
                    if valid_rotation:
                        # Rotation capped at 20% of total budget in screener
                        # mode (was full-Kelly-of-total which over-allocated
                        # given rotation calibration r²=0.007).
                        _rot_pool  = total_budget * 0.20
                        _rot_share = _rot_pool / len(valid_rotation)
                        for r in valid_rotation:
                            candidates.append({
                                "ticker":      r["ucits"],
                                "bucket":      "ROTATION",
                                "reason":      (
                                    f"rotation top-{rotation_top3.index(r)+1} "
                                    f"score {r['score']:.0f}"
                                ),
                                "suggested":   _rot_share,
                                "composite_z": float("nan"),
                                "ann_alpha":   _ROTATION_FLAT_FALLBACK,
                                "confidence": 1.0,
                                "extra": {
                                    "us_proxy":    r["us_ticker"],
                                    "score":       r["score"],
                                    "kelly_alloc": _rot_share,
                                },
                            })

                # ── CORE candidates via composite z-score ──────────────
                try:
                    from core.core_radar import scan_core_universe
                    core_opps = scan_core_universe(min_score=60.0)
                    for p in core_opps:
                        cb = composite_from_signals(
                            p["ticker"], p, horizon="CORE", stats=universe_stats,
                        )
                        candidates.append({
                            "ticker":      p["ticker"],
                            "bucket":      "CORE",
                            "reason": (
                                f"Core score {p['score']:.0f} "
                                f"(${p.get('current_price', 0):.2f})  "
                                f"composite_z {cb.composite_z:+.2f} "
                                f"(q{cb.quality_z:+.1f} dd{cb.dd_z:+.1f} v{cb.valuation_z:+.1f})"
                            ),
                            "suggested":   0.0,
                            "composite_z": float(cb.composite_z),
                            "ann_alpha":   0.0,
                            "confidence":  1.0,
                            "extra": {
                                "score":          p["score"],
                                "price":          p.get("current_price"),
                                "composite_z":    float(cb.composite_z),
                                "quality_z":      float(cb.quality_z),
                                "dd_z":           float(cb.dd_z),
                                "valuation_z":    float(cb.valuation_z),
                                "dd_from_52w":    p.get("dd_from_52w_high"),
                            },
                        })
                except Exception as _e:
                    info_lines.append(f"⚠️ Core picks falhou: `{_e}`")

                # ── DIP candidates via composite z-score (DIP horizon) ──
                try:
                    import state as _st
                    from scanners.alert_db import get_recent_alert_tickers
                    from core.core_radar import _fetch_hist, _score_core_dip

                    today_set  = list(_st.load_alerts())
                    recent_set = get_recent_alert_tickers(lookback_days=3)
                    raw_alerts = list({*today_set, *recent_set})
                    today_alerts = [
                        t for t in raw_alerts
                        if not str(t).upper().startswith("WL_")
                        and "_" not in str(t)
                    ]
                    for ticker in today_alerts:
                        try:
                            hist = _fetch_hist(ticker)
                            if hist is None or hist.empty:
                                continue
                            close = hist["Close"]
                            current_price = float(close.iloc[-1])
                            volume = hist["Volume"] if "Volume" in hist.columns else None
                            _score_x, dip_signals = _score_core_dip(
                                close, current_price, 0.0,
                                ticker=ticker, volume=volume,
                            )
                            cb = composite_from_signals(
                                ticker, dip_signals, horizon="DIP",
                                stats=universe_stats,
                            )
                            candidates.append({
                                "ticker":      ticker,
                                "bucket":      "DIP",
                                "reason": (
                                    f"DIP alert composite_z {cb.composite_z:+.2f} "
                                    f"(q{cb.quality_z:+.1f} dd{cb.dd_z:+.1f} v{cb.valuation_z:+.1f})"
                                ),
                                "suggested":   0.0,
                                "composite_z": float(cb.composite_z),
                                "ann_alpha":   0.0,
                                "confidence":  1.0,
                                "extra": {
                                    "score":       dip_signals.get("core_score"),
                                    "composite_z": float(cb.composite_z),
                                    "quality_z":   float(cb.quality_z),
                                    "dd_z":        float(cb.dd_z),
                                    "valuation_z": float(cb.valuation_z),
                                    "ret_20d":     dip_signals.get("return_20d"),
                                },
                            })
                        except Exception:
                            continue
                except Exception as _e:
                    info_lines.append(f"⚠️ DIP queue falhou: `{_e}`")

                # ── HOLD: SELL+BUY same-ticker contradiction ───────────
                # If a ticker is queued to SELL (e.g. ATR stop break) AND
                # would also be bought from another radar (DIP alert), the
                # two signals contradict each other on the same name. Tax
                # friction + bid/ask + slippage + the conviction signal of
                # "model still likes this at lower price" all favor HOLD
                # over churn. Pull both rows out, surface, recompute budget.
                _sell_set = {s["ticker"].upper() for s in sell_actions}
                _cand_set = {c["ticker"].upper() for c in candidates}
                _hold_tickers = _sell_set & _cand_set
                if _hold_tickers:
                    for _ht in _hold_tickers:
                        _sell_row = next(s for s in sell_actions if s["ticker"].upper() == _ht)
                        _buy_row  = next(c for c in candidates    if c["ticker"].upper() == _ht)
                        hold_actions.append({
                            "ticker":      _sell_row["ticker"],
                            "bucket":      _sell_row.get("bucket", "?"),
                            "shares":      _sell_row.get("shares", 0),
                            "value":       _sell_row.get("value", 0.0),
                            "sell_reason": _sell_row.get("reason", ""),
                            "buy_reason":  _buy_row.get("reason", ""),
                            "would_alloc": _buy_row.get("suggested", 0.0),
                        })
                    sell_actions = [s for s in sell_actions if s["ticker"].upper() not in _hold_tickers]
                    candidates   = [c for c in candidates   if c["ticker"].upper() not in _hold_tickers]
                    if ignore_sells:
                        freed_cash = 0.0
                    else:
                        freed_cash = sum(s["value"] for s in sell_actions)
                    total_budget = cash_amount + freed_cash
                    sell_actions_to_show = sell_actions

                # ── Per-bucket cap to keep proposal scannable ──────────
                # CORE can yield 8+ picks in a deep drawdown month; cap at
                # top 5 by composite_z to keep the message readable. DIP
                # capped at 3 (already short-list of recent alerts).
                _PER_BUCKET_CAP = {"CORE": 5, "DIP": 3, "ROTATION": 3}
                _by_bucket: dict[str, list[dict]] = {}
                for c in candidates:
                    _by_bucket.setdefault(c["bucket"], []).append(c)
                _capped: list[dict] = []
                for bucket, group in _by_bucket.items():
                    if bucket == "ROTATION":
                        group.sort(key=lambda x: -float(x.get("extra", {}).get("score", 0)))
                    else:
                        group.sort(key=lambda x: -float(x.get("composite_z", 0.0)))
                    _capped.extend(group[:_PER_BUCKET_CAP.get(bucket, 5)])
                candidates = _capped

                # ── Allocate: rotation first (fixed), then softmax CORE+DIP ──
                rot_total = 0.0
                for c in candidates:
                    if c["bucket"] != "ROTATION":
                        continue
                    buy_actions.append({
                        "ticker":        c["ticker"],
                        "bucket":        c["bucket"],
                        "reason":        c["reason"],
                        "value":         round(c["suggested"], 0),
                        "ann_alpha_pct": c["ann_alpha"] * 100,
                        "extra":         c.get("extra") or {},
                    })
                    rot_total += c["suggested"]
                    ranked_deploys.append({
                        "ticker": c["ticker"],
                        "bucket": c["bucket"],
                        "roi":    0.0,
                        "alpha":  c["ann_alpha"] * 100,
                    })

                remaining_budget = max(0.0, total_budget - rot_total)

                cd_pairs = [
                    (c["ticker"], c["composite_z"])
                    for c in candidates if c["bucket"] in ("CORE", "DIP")
                ]
                _current_positions_eur = {
                    sym.upper(): float(val)
                    for sym, val in ticker_current_value.items()
                }

                if cd_pairs and remaining_budget > 50:
                    allocs, sm_skipped = softmax_sizing(
                        cd_pairs,
                        budget_eur=remaining_budget,
                        threshold_z=0.3,
                        min_alloc_eur=100.0,
                        concentration_cap_pct=0.30,
                        portfolio_value_eur=portfolio_value,
                        current_positions_eur=_current_positions_eur,
                    )
                    for ticker, reason in sm_skipped:
                        info_lines.append(f"⏭️ `{ticker}` skipped — {reason}")

                    cand_by_ticker = {
                        c["ticker"]: c for c in candidates
                        if c["bucket"] in ("CORE", "DIP")
                    }
                    # Sort allocations by composite_z desc for display order
                    for ticker in sorted(
                        allocs.keys(),
                        key=lambda t: -float(cand_by_ticker.get(t, {}).get("composite_z", 0.0)),
                    ):
                        eur = allocs[ticker]
                        c = cand_by_ticker.get(ticker)
                        if not c:
                            continue
                        buy_actions.append({
                            "ticker":        ticker,
                            "bucket":        c["bucket"],
                            "reason":        c["reason"],
                            "value":         round(eur, 0),
                            "ann_alpha_pct": 0.0,
                            "extra":         c.get("extra") or {},
                        })
                        ranked_deploys.append({
                            "ticker": ticker,
                            "bucket": c["bucket"],
                            "roi":    c["composite_z"],
                            "alpha":  0.0,
                        })
                elif cd_pairs and remaining_budget <= 50:
                    info_lines.append(
                        f"ℹ️ Budget restante €{remaining_budget:.0f} ≤ €50 — "
                        f"sem deploy CORE/DIP (rotation consumiu)"
                    )

                # Update virtual portfolio so concentration math reflects new buys
                for b in buy_actions:
                    ticker_current_value[b["ticker"].upper()] = (
                        ticker_current_value.get(b["ticker"].upper(), 0.0) + b["value"]
                    )

                # ── Cash bucket: whatever's left ───────────────────────
                _deployed_total_local = sum(b["value"] for b in buy_actions)
                cash_bucket = max(0.0, total_budget - _deployed_total_local)
                remaining = cash_bucket  # alias for downstream code compatibility
"""

new_text = text[:start_idx] + new_block + text[end_idx_after:]
src.write_text(new_text, encoding="utf-8")
print(f"OK: replaced {end_idx_after - start_idx} chars with {len(new_block)} chars")
print(f"Delta: {len(new_block) - (end_idx_after - start_idx):+d} chars")

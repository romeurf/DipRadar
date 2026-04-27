def calculate_dip_score(fundamentals, rsi):
    score = 0
    if fundamentals.get("fcf_yield", 0) > 0.05: score += 3
    if fundamentals.get("revenue_growth", 0) > 0.10: score += 2
    if fundamentals.get("gross_margin", 0) > 0.40: score += 2
    if rsi and rsi < 30: score += 2
    if fundamentals.get("debt_equity", 999) < 1: score += 1
    return score

# Alert só se score >= 8

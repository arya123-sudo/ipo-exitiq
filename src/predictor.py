"""Listing-gain predictor v1: interpretable weighted model.
Same features the v2 XGBoost model will use. Weights are priors from
well-known IPO patterns (GMP + QIB subscription are strongest signals);
they get replaced by learned weights once we train on real 2019-2026 data.
"""

def predict_listing(ipo):
    """ipo: dict with gmp_pct, gmp_trend (-1..1), qib_x, nii_x, retail_x,
    issue_size_cr, roe_pct, debt_equity, pe_vs_peers (1.0=fair, >1 expensive),
    nifty_trend_pct, sentiment (-1..1)
    Returns (expected_gain_pct, profit_probability_0_100, reasons list)"""
    reasons, score = [], 0.0

    g = ipo["gmp_pct"]
    score += g * 0.55
    reasons.append(f"GMP {g:+.1f}% is the single strongest listing signal (weight 55%)")
    if ipo["gmp_trend"] < -0.3:
        score -= 6; reasons.append("GMP is FALLING in last 3 days - smart money cooling off (-6)")
    elif ipo["gmp_trend"] > 0.3:
        score += 4; reasons.append("GMP rising into listing - momentum building (+4)")

    qib = ipo["qib_x"]
    if qib > 20: score += 8; reasons.append(f"QIB subscribed {qib}x - institutions very bullish (+8)")
    elif qib > 5: score += 4; reasons.append(f"QIB subscribed {qib}x - healthy institutional demand (+4)")
    elif qib < 1: score -= 8; reasons.append(f"QIB only {qib}x - institutions are NOT interested (-8)")

    total_sub = (ipo["qib_x"] + ipo["nii_x"] + ipo["retail_x"]) / 3
    if total_sub > 15: score += 4; reasons.append(f"Overall subscription very high (~{total_sub:.0f}x avg) (+4)")

    if ipo["pe_vs_peers"] > 1.4: score -= 5; reasons.append("Priced much costlier than listed peers - little left on table (-5)")
    if ipo["debt_equity"] > 1.5: score -= 4; reasons.append(f"High debt (D/E {ipo['debt_equity']}) - fundamental risk (-4)")
    if ipo["roe_pct"] > 18: score += 3; reasons.append(f"Strong ROE {ipo['roe_pct']}% (+3)")

    score += ipo["nifty_trend_pct"] * 1.2
    if abs(ipo["nifty_trend_pct"]) > 1: reasons.append(f"Market mood adds {ipo['nifty_trend_pct']*1.2:+.1f}")
    score += ipo["sentiment"] * 5
    if abs(ipo["sentiment"]) > 0.2: reasons.append(f"News sentiment {ipo['sentiment']:+.2f} moves score {ipo['sentiment']*5:+.1f}")

    gain = round(score, 1)
    # convert expected gain to a rough profit probability (logistic-ish, clipped)
    prob = max(5, min(95, int(50 + gain * 1.6)))
    return gain, prob, reasons

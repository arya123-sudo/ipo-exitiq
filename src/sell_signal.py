"""Sell/Hold engine: the heart of ExitIQ.
Philosophy: never one binary guess. Give action + confidence + reasons +
trailing stop, and prefer PARTIAL SELL to kill regret on both sides
(sell half, let half ride with a stop-loss)."""

def sell_signal(pos):
    """pos: issue_price, listing_price, current_price, day_change_pct,
    peak_price_since_listing, volume_vs_avg (1.0=normal), sentiment_now,
    sentiment_listing_day, days_since_listing
    Returns dict: action, confidence, reasons, trailing_stop, gain_pct_now"""
    issue, cur, peak = pos["issue_price"], pos["current_price"], pos["peak_price_since_listing"]
    gain_now = (cur - issue) / issue * 100
    fall_from_peak = (peak - cur) / peak * 100 if peak else 0

    sell_votes, hold_votes, reasons = 0.0, 0.0, []

    if gain_now >= 20:
        sell_votes += 2; reasons.append(f"You are +{gain_now:.1f}% above issue price - a bird in hand. Listing gains this size usually cool off.")
    elif gain_now <= -5:
        sell_votes += 2.5; reasons.append(f"You are {gain_now:.1f}% BELOW issue price - capital protection first.")
    else:
        hold_votes += 1; reasons.append(f"Gain is modest ({gain_now:+.1f}%) - no urgent reason either way from price alone.")

    if fall_from_peak >= 8:
        sell_votes += 2.5; reasons.append(f"Down {fall_from_peak:.1f}% from its peak (₹{peak}) - momentum has reversed.")
    if pos["day_change_pct"] <= -4:
        sell_votes += 1.5; reasons.append(f"Today alone it is {pos['day_change_pct']}% - heavy selling today.")
    elif pos["day_change_pct"] >= 3:
        hold_votes += 1.5; reasons.append(f"Today it is +{pos['day_change_pct']}% - buyers still in control today.")

    if pos["volume_vs_avg"] >= 2 and pos["day_change_pct"] < 0:
        sell_votes += 1.5; reasons.append("Volume is 2x+ average ON a down day - distribution, big holders exiting.")
    sent_drop = pos["sentiment_listing_day"] - pos["sentiment_now"]
    if sent_drop >= 0.4:
        sell_votes += 1.5; reasons.append("News sentiment has turned sharply negative since listing.")
    elif pos["sentiment_now"] > 0.3:
        hold_votes += 1; reasons.append("News sentiment is still positive.")

    if pos["days_since_listing"] == 0 and gain_now >= 25:
        sell_votes += 1; reasons.append("Listing-day pop above +25% - historically most such pops are best booked at least partially on day 1.")

    total = sell_votes + hold_votes
    sell_prob = int(sell_votes / total * 100) if total else 50

    if sell_prob >= 65: action = "SELL"
    elif sell_prob >= 45: action = "PARTIAL SELL (book 50%, ride rest with stop-loss)"
    else: action = "HOLD"

    # trailing stop: 8% below peak, but never below issue price if we are in profit zone locking gains
    trailing_stop = round(max(peak * 0.92, issue * 1.02 if gain_now > 10 else issue * 0.95), 2)

    return {"action": action, "sell_confidence_pct": sell_prob, "gain_pct_now": round(gain_now, 1),
            "trailing_stop": trailing_stop, "reasons": reasons}

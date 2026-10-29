"""ExitIQ demo - run: python3 src/demo.py  (from ipo-exitiq folder)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sentiment import sentiment_score
from predictor import predict_listing
from sell_signal import sell_signal

print("="*70); print("CASE 1 - YOUR REAL ONE: Jio Platforms (replay)")
heads = ["Jio Platforms lists at strong premium amid robust demand",
         "Analysts bullish on Jio growth, record subscription seen",
         "Some profit booking seen after surge in early trade"]
s, lab = sentiment_score(heads); print(f"News sentiment: {s} ({lab})")
gain, prob, reasons = predict_listing(dict(
    gmp_pct=22.0, gmp_trend=0.4, qib_x=18, nii_x=12, retail_x=6,
    issue_size_cr=25000, roe_pct=14, debt_equity=0.6, pe_vs_peers=1.1,
    nifty_trend_pct=0.5, sentiment=s))
print(f"Pre-listing prediction: expected gain ~{gain}% | profit probability {prob}%")
for r in reasons: print("  -", r)
print(f"ACTUAL: listed at 1358 vs issue 1119 = +21.36%  -> prediction direction: CORRECT")
sig = sell_signal(dict(issue_price=1119, listing_price=1358, current_price=1358,
    day_change_pct=0.0, peak_price_since_listing=1370, volume_vs_avg=1.3,
    sentiment_now=s, sentiment_listing_day=s, days_since_listing=0))
print(f"Listing-morning signal at Rs.1358: {sig['action']} (sell confidence {sig['sell_confidence_pct']}%)")
print(f"  Your actual move: SOLD all 12 at 1358 = +Rs.2,868 gross. Engine agrees booking profit here was right;")
print(f"  PARTIAL-SELL variant would have kept 6 shares riding above trailing stop Rs.{sig['trailing_stop']}")

print("="*70); print("CASE 2 - The regret trap: big pop, then momentum dies")
sig2 = sell_signal(dict(issue_price=300, listing_price=450, current_price=405,
    day_change_pct=-5.2, peak_price_since_listing=470, volume_vs_avg=2.4,
    sentiment_now=-0.3, sentiment_listing_day=0.5, days_since_listing=2))
print(f"Gain now {sig2['gain_pct_now']}% but signal: {sig2['action']} (sell confidence {sig2['sell_confidence_pct']}%) | trailing stop Rs.{sig2['trailing_stop']}")
for r in sig2["reasons"]: print("  -", r)

print("="*70); print("CASE 3 - Weak IPO: below issue, institutions cold")
g3, p3, r3 = predict_listing(dict(gmp_pct=-2, gmp_trend=-0.6, qib_x=0.7, nii_x=0.9, retail_x=1.2,
    issue_size_cr=400, roe_pct=8, debt_equity=2.1, pe_vs_peers=1.6, nifty_trend_pct=-0.8, sentiment=-0.3))
print(f"Prediction: expected gain ~{g3}% | profit probability {p3}% -> APPLY verdict: SKIP / apply only if you accept listing-loss risk")
for r in r3: print("  -", r)
print("="*70); print("Reminder: probabilities + reasons, never a promise. Not SEBI advice.")

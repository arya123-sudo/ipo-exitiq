"""Use the trained v2 models on a new IPO.
Run: .venv/bin/python src/predict.py --subscription 65 --issue-size 1200 --issue-price 480 --lot 31
GMP note: if you know the GMP %, the simple GMP rule was our single best
listing-gain estimator on the GMP subset (MAE 11.3%) - compare both."""
import pickle, argparse, numpy as np, pandas as pd, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--subscription", type=float, required=True, help="total subscription x")
ap.add_argument("--issue-size", type=float, required=True, help="Rs crore")
ap.add_argument("--issue-price", type=float, required=True)
ap.add_argument("--lot", type=float, default=30)
ap.add_argument("--gmp-pct", type=float, default=None)
a = ap.parse_args()
row = {"log_issue_size": np.log1p(a.issue_size), "log_issue_price": np.log1p(a.issue_price),
       "log_subscription": np.log1p(a.subscription), "lot_size": a.lot, "nifty_chg_pct": 0.0,
       "vix": 14.0, "brokers_subscribe": 5, "brokers_avoid": 1, "prev_quarter_success_close": 1, "year": 2026}
with open(f"{BASE}/models/xgb_listing_gain.pkl", "rb") as fh: reg = pickle.load(fh)
with open(f"{BASE}/models/xgb_profit_classifier.pkl", "rb") as fh: clf = pickle.load(fh)
X = pd.DataFrame([row])[reg["features"]]
gain = reg["model"].predict(X)[0]; prob = clf["model"].predict_proba(X)[0][1] * 100
print(f"XGBoost predicted listing-close gain: {gain:+.1f}%")
print(f"XGBoost profit probability: {prob:.0f}%")
if a.gmp_pct is not None: print(f"Simple GMP rule estimate: {a.gmp_pct:+.1f}%  (historically MAE ~11% - our best single signal)")
print("Verdict:", "APPLY - strong listing setup" if prob >= 65 and gain > 8 else
      "MAY APPLY - modest/uncertain" if prob >= 50 else "SKIP - weak setup, listing-loss risk")
print("Reminder: model trained on 2009-2023 mainline IPOs; probabilities, not promises.")

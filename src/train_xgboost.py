"""ExitIQ v2 - train XGBoost on REAL data (data/ipo_clean.csv, 418 mainline IPOs).
Honest evaluation: TIME split (train on past, test on future) - random splits flatter.
Run: .venv/bin/python src/train_xgboost.py"""
import pandas as pd, numpy as np, pickle, os, json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f"{BASE}/data/ipo_clean.csv", parse_dates=["listing_date"])
df["log_issue_size"] = np.log1p(df["issue_size_cr"])
df["log_issue_price"] = np.log1p(df["issue_price"])
df["log_subscription"] = np.log1p(df["total_subscription"])
df["year"] = df["listing_date"].dt.year

CORE = ["log_issue_size", "log_issue_price", "log_subscription", "lot_size",
        "nifty_chg_pct", "vix", "brokers_subscribe", "brokers_avoid",
        "prev_quarter_success_close", "year"]

def evaluate(name, y_true, y_pred, y_prob=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    direction = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
    line = f"{name}: MAE {mae:.1f}% | RMSE {rmse:.1f}% | R2 {r2:.3f} | direction-correct {direction:.1f}%"
    print(line); return line

report = ["# ExitIQ v2 - XGBoost on real data", "",
          f"Dataset: 418 real mainline Indian IPOs (2009-2023), source: HuggingFace sohomghosh/Indian_IPO_datasets (CC BY-NC-SA 4.0).",
          f"Base rate: 67.0% of IPOs closed listing day above issue price; median gain +9.2%.", ""]

# ---------- TIME SPLIT: train <=2021, test 2022-2023 ----------
tr = df[df["listing_date"] < "2022-01-01"]; te = df[df["listing_date"] >= "2022-01-01"]
print(f"time split: train {len(tr)} (<=2021) | test {len(te)} (2022-23)")
Xtr, ytr = tr[CORE], tr["gain_close_pct"]; Xte, yte = te[CORE], te["gain_close_pct"]

report.append("## Listing-gain regression - TIME split (train past, test future)")
report.append(evaluate("Naive (always predict train median)", yte, np.full(len(yte), ytr.median())))
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr.fillna(-1), ytr)
report.append(evaluate("RandomForest", yte, rf.predict(Xte.fillna(-1))))
xgb = XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=3, subsample=0.85,
                   colsample_bytree=0.85, random_state=42).fit(Xtr.fillna(-1), ytr)
pred = xgb.predict(Xte.fillna(-1))
report.append(evaluate("XGBoost", yte, pred))

# classifier: profit on listing close?
ctr, cte = tr["profit_close"], te["profit_close"]
base_acc = max(cte.mean(), 1 - cte.mean()) * 100
xgbc = XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.85,
                     colsample_bytree=0.85, random_state=42, eval_metric="logloss").fit(Xtr.fillna(-1), ctr)
prob = xgbc.predict_proba(Xte.fillna(-1))[:, 1]
acc = accuracy_score(cte, prob > 0.5) * 100; auc = roc_auc_score(cte, prob)
line = f"Profit classifier (XGBoost): accuracy {acc:.1f}% vs always-say-profit baseline {base_acc:.1f}% | AUC {auc:.3f}"
print(line); report += ["", "## Profit / loss classifier - TIME split", line]

imp = sorted(zip(CORE, xgb.feature_importances_), key=lambda x: -x[1])
report += ["", "## What the model actually relies on (XGBoost importance)"]
for f, v in imp: line = f"- {f}: {v:.3f}"; print(line); report.append(line)

# ---------- GMP subset (112 matched IPOs) ----------
g = df.dropna(subset=["gmp_pct"]).copy()
gtr, gte = g[g["listing_date"] < "2022-01-01"], g[g["listing_date"] >= "2022-01-01"]
if len(gte) >= 10:
    report += ["", f"## GMP subset ({len(g)} IPOs with grey-market data) - TIME split train {len(gtr)} / test {len(gte)}"]
    report.append(evaluate("Naive GMP rule (gain = GMP%)", gte["gain_close_pct"], gte["gmp_pct"]))
    GF = CORE + ["gmp_pct"]
    xg2 = XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.85,
                       colsample_bytree=0.85, random_state=42).fit(gtr[GF].fillna(-1), gtr["gain_close_pct"])
    report.append(evaluate("XGBoost + GMP", gte["gain_close_pct"], xg2.predict(gte[GF].fillna(-1))))
    imp2 = sorted(zip(GF, xg2.feature_importances_), key=lambda x: -x[1])[:3]
    report.append("Top features with GMP: " + ", ".join(f"{f} {v:.2f}" for f, v in imp2))

os.makedirs(f"{BASE}/models", exist_ok=True)
with open(f"{BASE}/models/xgb_listing_gain.pkl", "wb") as fh: pickle.dump({"model": xgb, "features": CORE}, fh)
with open(f"{BASE}/models/xgb_profit_classifier.pkl", "wb") as fh: pickle.dump({"model": xgbc, "features": CORE}, fh)
with open(f"{BASE}/MODELS_REPORT.md", "w") as fh: fh.write("\n".join(report) + "\n")
print("\nSaved: models/xgb_listing_gain.pkl, models/xgb_profit_classifier.pkl, MODELS_REPORT.md")

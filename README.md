# IPO ExitIQ — Sell / Hold Intelligence for IPO Investors

**The problem (Arya's own experience):**
You apply for an IPO → sometimes no allotment. When you *do* get allotment,
listing day is confusing: price spikes, then falls, then rises again.
Sell early → regret when it goes up. Hold → regret when it crashes.
There is no structured, data-backed answer to one question: **"Should I sell now?"**

**What ExitIQ does (decision-support, NOT a guarantee):**
No model can predict the market with certainty — anyone claiming that is lying.
ExitIQ instead gives a *probability + reasons*, so the decision stops being pure emotion/FOMO.

## The 3 engines

### 1. Apply or Skip? — Listing-Gain Predictor (ML)
Predicts expected listing gain % and profit/loss probability *before* listing.
Features:
- Grey Market Premium (GMP) trend — current GMP, 3-day GMP slope (rising/falling)
- Subscription: QIB / NII / Retail oversubscription (x times)
- Issue size, price band, sector hotness, market mood (Nifty trend)
- Fundamentals: ROE, debt/equity, P/E vs listed peers, revenue growth
- Anchor investor quality
Model (v2): XGBoost / RandomForest regression + classifier, SHAP explanations.
Model (v1, in this repo): interpretable weighted scorer — same features, transparent weights,
so we can demo and explain it today and swap in the trained model once we have real data.

### 2. Got allotment — Sell or Hold? — Exit Signal Engine (ML + DL + Rules)
Inputs after listing: current price vs issue/listing price, intraday momentum,
volume spike, delivery %, news sentiment shift, GMP collapse, broad-market direction.
Output: `SELL` / `PARTIAL SELL (50%)` / `HOLD` + confidence (0-100) + human reasons +
a trailing stop-loss price ("if it falls below ₹X, exit no matter what").
v2 DL layer: LSTM/Transformer on 1-min / daily candles to estimate
P(price higher in 3 days) — added once we collect post-listing price history.

### 3. What is everyone saying? — NLP Sentiment Engine
- News headlines + articles about the IPO/company → sentiment score (-1 to +1)
- DRHP/RHP prospectus: LLM extracts risk factors, objects of issue, promoter history
- Social buzz (X/Reddit/YouTube comments) sentiment trend
v1 (in this repo): lexicon-based scorer, zero downloads, works offline.
v2: FinBERT (finance-tuned BERT) + GenAI summary of the prospectus in plain language.

### Quantum — honest placement
Quantum ML (QSVM / variational classifier via Qiskit/PennyLane) is an *experimental*
side-module for sentiment classification. On today's data sizes it will NOT beat
classical ML — we include it as a research/novelty experiment for hackathons and
papers, never as a marketing claim in the core product.

### Regret Backtester
For every past IPO: what if you sold on Day 1 vs Day 3 vs Day 7 vs trailing-stop rule?
Shows which *strategy* (not guess) historically minimised regret. This is how we
validate the sell engine before trusting it with real money.

## Your real case, built in as Test #1
Jio Platforms (JPL): issue ₹1,119, allotted 12 shares, listed ₹1,358 (+21.36%),
sold at ₹1,358 → +₹2,868 gross. ExitIQ replays this case in `src/demo.py`.

## Roadmap
- [x] v1: Interpretable predictor + sell/hold engine + NLP lexicon + demo (pure Python, no deps)
- [x] v2 DONE (2026-10-29): Real dataset — 418 mainline IPOs (2009–2023) from the published
      HuggingFace research dataset (sohomghosh/Indian_IPO_datasets) + GMP merged for 112.
      XGBoost trained with an honest TIME split (train ≤2021, test 2022–23): MAE 15.8%, R² 0.41,
      direction 73.4% (naive median: MAE 22.1%). RandomForest slightly beat it (MAE 14.3%).
      Big finding: the simple GMP rule (gain ≈ GMP%) is the best single estimator where GMP
      exists (MAE 11.3%, R² 0.74). Full numbers + caveats: MODELS_REPORT.md.
      Try it: `.venv/bin/python src/predict.py --subscription 65 --issue-size 1200 --issue-price 480 --gmp-pct 28`
- [ ] v3: Streamlit web app (Arya prefers clickable demos) — enter IPO → get Apply/Sell verdict
- [ ] v4: Post-listing price feed (yfinance/Zerodha) + LSTM next-3-day model
- [ ] v5: FinBERT sentiment + LLM prospectus summariser (GenAI — Arya's strength)
- [ ] v6: Alerts (WhatsApp/Telegram) on listing morning: "ExitIQ: SELL 70% confidence, reasons..."
- [ ] Exp: Quantum QSVM notebook (PennyLane) vs classical SVM comparison

## Disclaimer
Educational / decision-support tool. Not SEBI-registered investment advice.
Every output carries probability + reasons, never a promise.

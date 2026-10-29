# IPO ExitIQ — Apply / Sell-Hold decision support for Indian IPOs

**Live app:** https://arya123-sudo.github.io/ipo-exitiq/
**Live API:** https://ipo-exitiq-api.onrender.com

ExitIQ answers two questions for retail IPO investors:

1. **Should I apply?** — a listing-gain estimate with a range, a profit probability, and the reasons for and against.
2. **Should I sell or hold?** — a listing-day exit signal with a trailing stop-loss, a scenario table, and a "Time Machine" that rehearses the rest of the day 1,000 times from real historical distributions.

It is decision support: **chances + reasons, never a guarantee.** It is a student-built educational tool, **not SEBI-registered investment advice**.

## Which model is deployed (and why)

The web app runs a **compact XGBoost** model (regressor + profit classifier) exported to JSON and executed in the browser — no server needed for the prediction itself.

- Features (exactly these): total subscription, issue size, issue price, lot size, Nifty change, India VIX, broker subscribe/avoid counts, previous-quarter listing success rate, year.
- The detailed form's QIB / NII / retail split and OFS % drive the **risk flags and verdict logic**, not the regression.

Research models (in `MODELS_REPORT.md`, **not** deployed): the full XGBoost (MAE 15.8%) and a RandomForest (MAE 14.3%). The RandomForest scored slightly better but is 400 deep trees — too heavy to ship to a phone browser as JSON; the compact XGBoost (MAE 15.4%) exports cleanly and runs on-device. That trade-off is stated openly rather than hidden.

## Honest benchmark (time split — train 2009–2021, test 2022–23, 94 IPOs)

Random splits leak market regimes, so only a time split is reported.

| Estimator | Test set | MAE | Direction |
|---|---|---|---|
| Naive (always the train median) | 94 IPOs | 22.1% | — |
| Deployed XGBoost | 94 IPOs | 15.4% | 73% |
| Always predict UP (baseline) | 94 IPOs | — | 74.5% |
| GMP rule (gain ≈ GMP %) | 72 IPOs with GMP | **11.3%** | 71% |
| XGBoost on those same 72 | 72 IPOs | 15.0% | 65% |
| Always predict UP on those 72 | 72 IPOs | — | 75% |

Read it straight:

- In 2022–23 ~3 of 4 IPOs rose, so "always say up" beats everyone on direction. The model's real edge is **gain sizing** (15.4% vs 22.1% error) and **risk ranking** (profit-probability AUC 0.764).
- The **GMP rule is the sharpest single estimator** where GMP exists (bootstrap 95% CI for the MAE gap: +1.6 to +6.0 points). A model+GMP blend was backtested and did **not** beat the GMP rule — so the app **headlines GMP when available** and uses the model as the probability/risk layer. When model and GMP conflict (>10 points or opposite signs), the app flags it and lowers confidence.
- Caveats: GMP data exists for only ~112 of the 418 IPOs and skews toward hotter issues (biased subset); GMP itself is unofficial and can be operator-influenced; training data ends in 2023, so the **live scorecard is the real out-of-sample evidence**.

Full methodology: `MODELS_REPORT.md`.

## Public scorecard

Every live prediction is written to **`predictions.json`** in this repo *before* the IPO lists — the git commit date is the timestamp and cannot be backdated. A **miss** is defined in advance: the actual listing-day closing gain has the opposite sign to the headline estimate, or is more than 15 points away. Results are filled in after each listing, win or lose.

## The multi-agent live API

`api/main.py` (FastAPI, deployed on Render) answers `GET /ipo?name=<any IPO>` with three agents working **in parallel**:

- **Data Scout** — resolves the name and reads the IPO's live IPOTrack page (price, lot, size, subscription split, GMP, OFS, dates).
- **Cross-Checker** — independently reads IPOWatch, verifies every figure, fills gaps, flags disagreements.
- **Sentiment Agent** — keyword-scores the latest Google News headlines about the IPO (a simple mood gauge; headlines are shown for judgement).

Responses carry per-agent status, a confidence grade, source links and an IST fetch timestamp. Missing data stays `null` — never guessed. The web app also ships a built-in library of current IPOs (`ipos.json`, fuzzy-matched) so common checks answer instantly, with the live agents verifying silently in the background.

## Sell / Hold engine — status

Rule-based expert weights (price vs issue, fall from peak, day move, volume, news mood, anchor lock-in windows at 30/90 days) + a trailing stop-loss. **It is not yet backtested** — the app labels it as such. The scenario table and Time Machine use the real open→close drift distribution of the 418-IPO dataset. A **Regret Backtester** (Day-1 vs Day-3 vs Day-7 vs trailing-stop exits across history) is the top roadmap item to validate this engine.

## Data

- Training: 418 mainline Indian IPOs, 2009–2023 — HuggingFace `sohomghosh/Indian_IPO_datasets` (CC BY-NC-SA 4.0); GMP merged for 112.
- Live: IPOTrack + IPOWatch (cross-checked), Google News RSS for headlines.

## Roadmap

- [ ] Regret Backtester for the sell/hold engine
- [ ] Fresher training data (2024–26 IPOs)
- [ ] FinBERT sentiment upgrade (current scorer is keyword-based, labelled as such)
- [ ] Post-listing price tracker + LSTM once enough price history accumulates
- [ ] Quantum module stays an explicitly-labelled experiment, never a core claim

## Disclaimer

Educational tool only. Nothing here is investment advice or a recommendation to buy or sell any security. IPO investing carries risk, including loss of capital. Consult a SEBI-registered adviser for personal advice.

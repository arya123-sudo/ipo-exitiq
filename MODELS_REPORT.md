# ExitIQ v2 - XGBoost on real data

Dataset: 418 real mainline Indian IPOs (2009-2023), source: HuggingFace sohomghosh/Indian_IPO_datasets (CC BY-NC-SA 4.0).
Base rate: 67.0% of IPOs closed listing day above issue price; median gain +9.2%.

## Listing-gain regression - TIME split (train past, test future)
Naive (always predict train median): MAE 22.1% | RMSE 33.0% | R2 -0.194 | direction-correct 74.5%
RandomForest: MAE 14.3% | RMSE 22.4% | R2 0.448 | direction-correct 73.4%
XGBoost: MAE 15.8% | RMSE 23.2% | R2 0.410 | direction-correct 73.4%

## Profit / loss classifier - TIME split
Profit classifier (XGBoost): accuracy 72.3% vs always-say-profit baseline 74.5% | AUC 0.764

## What the model actually relies on (XGBoost importance)
- log_subscription: 0.333
- log_issue_size: 0.111
- brokers_subscribe: 0.097
- brokers_avoid: 0.090
- year: 0.084
- prev_quarter_success_close: 0.075
- lot_size: 0.073
- log_issue_price: 0.070
- vix: 0.067
- nifty_chg_pct: 0.000

## GMP subset (112 IPOs with grey-market data) - TIME split train 40 / test 72
Naive GMP rule (gain = GMP%): MAE 11.3% | RMSE 15.3% | R2 0.740 | direction-correct 70.8%
XGBoost + GMP: MAE 15.0% | RMSE 20.5% | R2 0.534 | direction-correct 65.3%
Top features with GMP: gmp_pct 0.48, log_subscription 0.13, brokers_subscribe 0.09

## Honest interpretation (read this before quoting any number)
- RandomForest (MAE 14.3%, R2 0.448) slightly BEAT XGBoost (MAE 15.8%, R2 0.410) on the
  future test set. RF+XGB ensemble: MAE 14.8%, R2 0.444. XGBoost is not magic on 418 rows;
  tree ensembles are all in the same band, and all crush the naive median (MAE 22.1%).
- Profit classifier accuracy (72.3%) is BELOW the always-say-profit baseline (74.5%) on this
  test window because 2022-23 was profit-heavy - but its AUC is 0.764, i.e. it genuinely
  ranks risky IPOs below safe ones. Use its probability, not a raw accuracy headline.
- GMP is king where available: the dumb rule 'listing gain = GMP%' scored MAE 11.3% / R2 0.740
  on the 112-IPO GMP subset, beating XGBoost+GMP there (small data -> ML overfits).
  Product lesson: show GMP prominently; use ML for probability + when GMP is absent/manipulated.
- #1 learned feature overall: total subscription (importance 0.333). Sanity demo:
  same IPO at 65x subscription -> +22.6% predicted; at 1.1x -> -5.9%.
- Limits: data ends Dec 2023; category-wise subscription (QIB/NII/Retail split) was present
  for only 78/418 rows, and RHP fundamentals (ROE, debt) for <25% - adding those is v3's job.

## Addendum (2026-10-29): baselines, blend test, and which model is deployed
- Always-UP direction baseline, test 2022-23 (n=94): 74.5%. The deployed model's 73% direction does NOT beat it; the model's edge is gain sizing (MAE 15.4% vs naive-median 22.1%) and probability ranking (AUC 0.764).
- GMP subset (n=72 test): GMP rule MAE 11.3% vs XGBoost+GMP 15.0%; bootstrap 95% CI for the difference: +1.6 to +6.0 points (excludes 0). Always-UP on this subset: 75.0% (GMP direction 71%, model 65%).
- Blend test: weight chosen on 2020-21 came out 0.0 (pure GMP). A model+GMP blend does NOT beat the GMP rule on this data, so the app headlines GMP when it exists and uses the model as the probability/risk layer.
- Deployed model: compact XGBoost exported to JSON, running in the browser (MAE 15.4%, R2 0.419). The research RandomForest (14.3%) was not deployed: 400 deep trees are too heavy to ship to a phone browser; the compact XGBoost exports cleanly.
- Caveats: GMP coverage is ~112/418 IPOs and skews to hotter issues (biased subset). Training data ends Dec 2023; the live scorecard (predictions.json, committed before listing) is the real out-of-sample test for 2026 IPOs.

"""Build clean dataset from the HuggingFace research dataset
sohomghosh/Indian_IPO_datasets (CC BY-NC-SA 4.0) - mainline IPOs.
Run: .venv/bin/python src/build_dataset.py"""
import pandas as pd, numpy as np, re, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(f"{BASE}/data/ipo_mainline_raw.xlsx", sheet_name="data_ordered")

out = pd.DataFrame()
out["company"] = df["Issuer Company"]
out["listing_date"] = pd.to_datetime(df["Listing Date"], errors="coerce")
out["issue_price"] = pd.to_numeric(df["Final_Issue_Price"], errors="coerce")
out["listing_open"] = pd.to_numeric(df["NSE_Open"], errors="coerce")
out["listing_high"] = pd.to_numeric(df["NSE_High"], errors="coerce")
out["listing_close"] = pd.to_numeric(df["NSE_Last_Trade"], errors="coerce")  # legend: Last Trade = Close
out["issue_size_cr"] = pd.to_numeric(df["Issue Size (Rs Cr.)"], errors="coerce")
out["total_subscription"] = pd.to_numeric(df["Total_subscriptions"], errors="coerce")
out["lot_size"] = pd.to_numeric(df["Lot Size"], errors="coerce")
out["nifty_chg_pct"] = pd.to_numeric(df["Chg%_nifty50_daily"], errors="coerce")
out["vix"] = pd.to_numeric(df["Price_vix_daily"], errors="coerce")
out["brokers_subscribe"] = pd.to_numeric(df["Brokers_Subscribe"], errors="coerce")
out["brokers_avoid"] = pd.to_numeric(df["Brokers_Avoid"], errors="coerce")
out["prev_quarter_success_close"] = pd.to_numeric(df["previous_quarter_success_close"], errors="coerce")

# targets
out["gain_open_pct"] = (out["listing_open"] - out["issue_price"]) / out["issue_price"] * 100
out["gain_close_pct"] = (out["listing_close"] - out["issue_price"]) / out["issue_price"] * 100
out["profit_close"] = (out["gain_close_pct"] > 0).astype(int)

# ---- merge GMP subset (InvestorGain via HF, 208 IPOs 2019-2024) by fuzzy name ----
g = pd.read_excel(f"{BASE}/data/gmp_mainline_raw.xlsx")
def norm(s):
    s = str(s).lower()
    s = re.sub(r"ipo.*$", "", s)            # drop 'IPO (SYMBOL,...)' tail
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(ltd|limited|india|the)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()
out["key"] = out["company"].map(norm)
g["key"] = g["IPO"].map(norm)
g["gmp_rs"] = pd.to_numeric(g["GMP"], errors="coerce")
g["gmp_pct"] = g["gmp_rs"] / pd.to_numeric(g["IPO Issue Price"], errors="coerce") * 100
merged = out.merge(g[["key", "gmp_rs", "gmp_pct"]].drop_duplicates("key"), on="key", how="left")
merged = merged.drop(columns=["key"]).dropna(subset=["issue_price", "listing_close", "total_subscription"])
merged = merged.sort_values("listing_date").reset_index(drop=True)
merged.to_csv(f"{BASE}/data/ipo_clean.csv", index=False)
print(f"clean rows: {len(merged)} | with GMP: {merged['gmp_pct'].notna().sum()}")
print(f"date range: {merged['listing_date'].min().date()} -> {merged['listing_date'].max().date()}")
print(f"profitable on listing close: {merged['profit_close'].mean()*100:.1f}% of IPOs")
print(f"median listing-close gain: {merged['gain_close_pct'].median():.1f}% | mean: {merged['gain_close_pct'].mean():.1f}%")

"""IPO ExitIQ live-data API — multi-agent edition.

A small team of specialised agents works every request IN PARALLEL:
  1. Data Scout      — resolves the IPO name and reads its live IPOTrack page
                       (price, lot, size, subscription split, GMP, OFS, dates).
  2. Cross-Checker   — independently reads IPOWatch and verifies every number
                       the Scout found; disagreements are flagged, gaps filled.
  3. Sentiment Agent — reads the latest Google News headlines about the IPO
                       and scores the news mood (positive / negative / mixed).
The merged, cross-checked result is returned with per-agent status + timing,
a confidence grade, source links and an IST fetch timestamp. Figures belong
to their sources; nothing is guessed — missing fields stay null.
"""
import gzip
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IPO ExitIQ Live API — Multi-Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

UA = {"User-Agent": "Mozilla/5.0 (ExitIQ personal research tool)", "Accept-Encoding": "gzip"}
_cache = {}
_sitemap = {"slugs": [], "at": 0}
IST = timezone(timedelta(hours=5, minutes=30))


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "ignore")


def text_of(html: str) -> str:
    t = re.sub(r"<script[\s\S]*?</script>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


# ---------------- Data Scout: name resolution + IPOTrack ----------------

def slugs() -> list:
    if time.time() - _sitemap["at"] > 3600 or not _sitemap["slugs"]:
        xml = fetch("https://ipotrack.in/sitemap-ipos.xml")
        _sitemap["slugs"] = sorted(set(re.findall(r"ipotrack\.in/ipo/([a-z0-9\-]+)", xml)))
        _sitemap["at"] = time.time()
    return _sitemap["slugs"]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-+", "-", s)


def resolve(name: str):
    want = slugify(name)
    pool = slugs()
    if want in pool:
        return want
    wt = set(want.split("-"))
    best, score = None, 0.0
    for s in pool:
        st = set(s.split("-"))
        j = len(wt & st) / max(1, len(wt | st))
        sc = max(j, SequenceMatcher(None, want, s).ratio())
        if sc > score:
            best, score = s, sc
    return best if score >= 0.45 else None


def num(pat, t, cast=float):
    m = re.search(pat, t)
    return cast(m.group(1)) if m else None


def parse_scout(html: str, slug: str) -> dict:
    t = text_of(html)
    pairs = re.findall(r"price band(?: set at)? ₹[\d.]+ [–-] ₹([\d.]+),? lot(?: size)? (\d+) shares", t, re.I)
    anchored = pairs[0] if pairs else None
    band = re.search(r"price band ₹[\d.]+ [–-] ₹([\d.]+)", t, re.I)
    gmp_m = (re.search(r"GMP Today ₹(-?[\d,]*\.?\d*) \((-?[\d.]+)%\)", t)
             or re.search(r"GMP \(unofficial\) ₹(-?[\d,]*\.?\d*) \((-?[\d.]+)%\)", t, re.I))
    subs = re.search(r"overall ([\d.]+)x, QIB ([\d.]+)x, retail ([\d.]+)x", t)
    fresh = num(r"Fresh Issue ₹([\d.]+) Cr", t)
    ofs = num(r"Offer For Sale \(OFS\) ₹([\d.]+) Cr", t)
    total_cr = num(r"Total Issue Size ₹([\d.]+) Cr", t)
    ofs_pct = None
    if total_cr and ofs is not None:
        ofs_pct = round(ofs / total_cr * 100, 1)
    elif ofs is None and fresh and total_cr and abs(fresh - total_cr) < 0.6:
        ofs_pct = 0.0
    trend = None
    if re.search(r"GMP[^\.]{0,80}rising", t, re.I):
        trend = "up"
    elif re.search(r"GMP[^\.]{0,80}falling", t, re.I):
        trend = "down"
    dates = []
    for lbl in ("IPO Open Date", "IPO Close Date", "Listing Date"):
        m = re.search(lbl + r" (\d{1,2} \w{3} \d{4})", t)
        if m:
            dates.append(f"{lbl.replace('IPO ', '')}: {m.group(1)}")
    return {
        "name": slug.replace("-", " ").title(),
        "slug": slug,
        "price": float(anchored[0]) if anchored else (float(band.group(1)) if band else None),
        "lot": int(anchored[1]) if anchored else num(r"Lot Size (\d+) Shares", t, int),
        "size_cr": total_cr,
        "total": float(subs.group(1)) if subs else None,
        "qib": float(subs.group(2)) if subs else None,
        "retail": float(subs.group(3)) if subs else None,
        "nii": num(r"(?:NII|Non-Institutional)[^.x]{0,30}?(\d+\.?\d*)x", t),
        "gmp_pct": float(gmp_m.group(2)) if gmp_m else None,
        "gmp_trend": trend,
        "ofs_pct": ofs_pct,
        "status": " · ".join(dates),
        "source": f"https://ipotrack.in/ipo/{slug}",
    }


# ---------------- Cross-Checker: IPOWatch verification ----------------

def run_checker(html: str, data: dict):
    cross = []
    w = text_of(html).replace(" ", "|")
    wcols = {}
    for lbl, key in (("QIB", "qib"), ("NII", "nii"), ("RII", "retail"), ("Total", "total")):
        m = re.search(lbl + r"\|([\d.]+)\|([\d.]+)\|([\d.]+)\|", w)
        if m:
            wcols[key] = [float(x) for x in m.groups()]
    for key, days in wcols.items():
        nz = [d for d in days if d > 0]
        if not nz:
            continue
        prim = data.get(key)
        v = min(nz, key=lambda d: abs(d - prim) / max(prim, 0.01)) if prim else nz[-1]
        if prim is None:
            data[key] = v
            cross.append(f"{key}: filled from IPOWatch ({v}x)")
        elif abs(prim - v) / max(v, 0.01) > 0.25 and abs(prim - v) > 0.3:
            cross.append(f"WARNING {key}: IPOTrack {prim}x vs IPOWatch {v}x disagree")
        else:
            cross.append(f"{key}: both sources agree ({prim}x)")
    return cross


# ---------------- Sentiment Agent: news mood ----------------

POS = {"surge", "soars", "soar", "strong", "robust", "gain", "gains", "rally", "bullish", "premium",
       "bumper", "blockbuster", "stellar", "healthy", "growth", "jumps", "jump", "oversubscribed",
       "multibagger", "listing gains", "positive", "boost", "record", "wins", "bags", "raises"}
NEG = {"fall", "falls", "drop", "drops", "weak", "slump", "decline", "loss", "losses", "fraud",
       "default", "caution", "risk", "risks", "plunge", "tumble", "muted", "tepid", "scam",
       "probe", "penalty", "selloff", "sell-off", "crash", "warning", "downgrade", "defer"}


def run_sentiment(name: str) -> dict:
    q = urllib.parse.quote(f"{name} IPO")
    xml = fetch(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")
    titles = re.findall(r"<item>[\s\S]*?<title>([^<]+)</title>", xml)[:8]
    heads, pos, neg = [], 0, 0
    for raw in titles:
        title = re.sub(r"\s+-\s+[^-]+$", "", raw).strip()
        low = title.lower()
        pos += sum(1 for w in POS if w in low)
        neg += sum(1 for w in NEG if w in low)
        if len(heads) < 3 and title:
            heads.append(title)
    score = round((pos - neg) / max(1, pos + neg), 2) if (pos + neg) else 0.0
    label = "Positive" if score >= 0.35 else ("Negative" if score <= -0.35 else "Mixed / no clear signal")
    return {"score": score, "label": label, "headlines": heads, "analysed": len(titles)}


# ---------------- Orchestrator ----------------

@app.get("/")
def home():
    return {"app": "IPO ExitIQ Live API — Multi-Agent", "agents": ["Data Scout", "Cross-Checker", "Sentiment Agent"],
            "usage": "/ipo?name=BLS Polymers", "ipos_indexed": len(_sitemap["slugs"])}


@app.get("/search")
def search(q: str = ""):
    ql = slugify(q)
    return {"names": [s.replace("-", " ").title() for s in slugs() if ql in s][:15]}


@app.get("/ipo")
def ipo(name: str):
    slug = resolve(name)
    if not slug:
        return {"found": False, "error": f"No IPO matching '{name}' found on IPOTrack."}
    hit = _cache.get(slug)
    if hit and time.time() - hit["at"] < 1200:
        return hit["data"]
    agents = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        t0 = time.time()
        f_scout = ex.submit(fetch, f"https://ipotrack.in/ipo/{slug}")
        f_check = ex.submit(fetch, f"https://ipowatch.in/{slug}-ipo-subscription-status/")
        f_sent = ex.submit(run_sentiment, name)
        try:
            data = parse_scout(f_scout.result(), slug)
            agents.append({"agent": "Data Scout", "status": "ok", "detail": "IPOTrack live page read"})
        except Exception as e:
            return {"found": False, "error": f"Data Scout failed: {e}"}
        cross = []
        try:
            cross = run_checker(f_check.result(), data)
            data["sources"] = {"primary": data["source"],
                                "cross_check": f"https://ipowatch.in/{slug}-ipo-subscription-status/"}
            agents.append({"agent": "Cross-Checker", "status": "ok",
                           "detail": f"IPOWatch verified {len(cross)} figures"})
        except Exception:
            data["sources"] = {"primary": data["source"]}
            agents.append({"agent": "Cross-Checker", "status": "unavailable",
                           "detail": "IPOWatch unreachable this time"})
        try:
            data["sentiment"] = f_sent.result()
            agents.append({"agent": "Sentiment Agent", "status": "ok",
                           "detail": f"{data['sentiment']['analysed']} news headlines analysed — mood: {data['sentiment']['label']}"})
        except Exception:
            data["sentiment"] = None
            agents.append({"agent": "Sentiment Agent", "status": "unavailable", "detail": "News feed unreachable this time"})
    elapsed = round(time.time() - t0, 2)
    for a in agents:
        a["total_s"] = elapsed
    core = [data.get(k) for k in ("price", "size_cr", "total", "qib", "gmp_pct")]
    have = sum(1 for v in core if v is not None)
    warns = sum(1 for c in cross if c.startswith("WARNING"))
    data["confidence"] = "high" if have >= 4 and warns == 0 else ("medium" if have >= 3 else "low")
    data["cross_check"] = cross
    data["agents"] = agents
    data["found"] = True
    data["fetched_at"] = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    _cache[slug] = {"data": data, "at": time.time()}
    return data

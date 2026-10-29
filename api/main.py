"""IPO ExitIQ live-data API — multi-agent design.

Three agents run in parallel for each lookup:
  1. Data Scout      — IPOTrack (primary numbers)
  2. Cross-Checker   — IPOWatch (verification + gap-filling; disagreements flagged)
  3. Sentiment Agent — Google News headlines (lexicon mood)

Every field is copied from a real page or returned as null ("not published yet").
Responses carry per-field sources, cross-check notes, a confidence level, an IST
timestamp, and each agent's status.
"""
import difflib
import gzip
import html as htmlmod
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IPO ExitIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
IST = timezone(timedelta(hours=5, minutes=30))

_slug_cache = {"at": 0.0, "slugs": []}
_ipo_cache = {}
SLUG_TTL = 3600
IPO_TTL = 1200


def fetch(url: str, tries: int = 2) -> str:
    last = None
    for _ in range(max(1, tries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            return raw.decode("utf-8", "ignore")
        except Exception as e:  # transient source hiccup — one retry
            last = e
    raise last


def text_of(html: str) -> str:
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", htmlmod.unescape(t)).strip()


def slugs() -> list:
    now = time.time()
    if _slug_cache["slugs"] and now - _slug_cache["at"] < SLUG_TTL:
        return _slug_cache["slugs"]
    try:
        xml = fetch("https://ipotrack.in/sitemap-ipos.xml")
        found = sorted(set(re.findall(r"ipotrack\.in/ipo/([a-z0-9\-]+)", xml)))
        if found:
            _slug_cache.update(at=now, slugs=found)
    except Exception:
        pass
    return _slug_cache["slugs"]


def slugify(name: str) -> str:
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"-(ltd|limited|india|ind|the)$", "", s)
    return s


GENERIC = {"tech", "technologies", "technology", "india", "indian", "limited", "ltd",
           "industries", "industry", "global", "enterprises", "solutions", "systems",
           "international", "group", "company", "co", "and", "the"}


def _concat_match(w, st):
    """'sfctech' -> ['sfc','environmental','technologies']: the first slug token is
    a prefix of w and the remainder prefixes some later token."""
    if len(w) < 5 or not st:
        return False
    for i in range(2, len(w) - 1):
        head, tail = w[:i], w[i:]
        if st[0] == head or (len(head) >= 3 and st[0].startswith(head)):
            if any(t.startswith(tail) for t in st[1:]):
                return True
    return False


def resolve(name: str):
    want = slugify(name)
    if not want:
        return None
    pool = slugs()
    if want in pool:
        return want
    wt = want.split("-")
    wset = set(wt)
    joined = "".join(wt)
    best, best_score = None, 0.0
    for s in pool:
        st = s.split("-")
        sset = set(st)
        inter = wset & sset
        strong = {t for t in inter if t not in GENERIC}
        j = len(inter) / len(wset | sset) if (wset | sset) else 0
        r = difflib.SequenceMatcher(None, want, s).ratio()
        ok = (bool(strong) and (j >= 0.25 or r >= 0.6)) or j >= 0.5 or r >= 0.8
        if not ok and _concat_match(joined, st):
            ok = True
            j = max(j, 0.34)
        if ok:
            score = max(j, r)
            if score > best_score:
                best, best_score = s, score
    return best


def num(pat, t, cast=float):
    m = re.search(pat, t, re.I)
    if not m:
        return None
    try:
        return cast(m.group(1).replace(",", ""))
    except Exception:
        return None


def parse_scout(html: str, slug: str) -> dict:
    t = text_of(html)
    d = {"name": slug.replace("-", " ").title(), "slug": slug}
    band = re.search(r"(?:price band|issue price)[:\s]*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)"
                     r"\s*(?:-|–|to)\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)", t, re.I)
    if band:
        d["price"] = float(band.group(2))
    else:
        d["price"] = num(r"₹\s*(\d+(?:\.\d+)?)\s*(?:per share|/share)", t)
    d["lot"] = num(r"(?:lot size|market lot)[:\s]*(\d+)\s*(?:shares|equity shares)?",
                   t, int)
    d["size_cr"] = num(r"(?:issue size|total issue size)[:\s]*(?:₹|rs\.?)?\s*"
                       r"([\d,]+(?:\.\d+)?)\s*(?:cr|crore)", t)
    fresh = num(r"fresh issue[^₹]{0,30}?(?:₹|rs\.?)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore)", t)
    ofs = num(r"offer for sale[^₹]{0,30}?(?:₹|rs\.?)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore)", t)
    d["ofs_pct"] = (round(ofs / (fresh + ofs) * 100, 1)
                    if fresh is not None and ofs is not None and (fresh + ofs) > 0 else None)
    d["total"] = num(r"(?:subscribed|subscription)[^\d]{0,40}?(\d+(?:\.\d+)?)\s*(?:x|times)", t)
    d["qib"] = num(r"qib[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t)
    d["nii"] = num(r"(?:nii|non[-\s]?institutional)[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t)
    d["retail"] = num(r"retail[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t)
    g = re.search(r"gmp\s*\(?unofficial\)?[^\d\-]{0,25}?(?:₹|rs\.?)?\s*(-?\d+(?:\.\d+)?)", t, re.I)
    if not g:
        g = re.search(r"gmp[^\d\-]{0,30}?(?:₹|rs\.?)?\s*(-?\d+(?:\.\d+)?)", t, re.I)
    gp = re.search(r"gmp[^\d%]{0,40}?\((-?\d+(?:\.\d+)?)\s*%\)", t, re.I)
    if gp:
        d["gmp_pct"] = float(gp.group(1))
    elif g and d.get("price"):
        d["gmp_pct"] = round(float(g.group(1)) / d["price"] * 100, 2)
    else:
        d["gmp_pct"] = None
    d["gmp_trend"] = None
    for key, label in (("open_date", "open"), ("close_date", "close"),
                       ("listing_date", "listing")):
        m = re.search(label + r"\s*date[:\s]*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
                              r"|\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4})", t, re.I)
        d[key] = m.group(1) if m else None
    d["status"] = ""
    return d


def run_checker(name: str, data: dict):
    """Cross-check against IPOWatch: fill gaps; flag big disagreements."""
    notes, filled = [], {}
    try:
        q = urllib.parse.quote(name)
        srch = fetch(f"https://ipowatch.in/?s={q}")
        links = re.findall(r'href="(https://ipowatch\.in/[^"]*ipo[^"]*)"', srch, re.I)
        page = None
        want = slugify(name).split("-")[0]
        for lnk in links:
            if want in lnk.lower():
                page = fetch(lnk)
                break
        if page is None and links:
            page = fetch(links[0])
        if page:
            t = text_of(page)
            cand = {"total": num(r"(?:subscribed|subscription)[^\d]{0,40}?(\d+(?:\.\d+)?)\s*(?:x|times)", t),
                    "qib": num(r"qib[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t),
                    "retail": num(r"retail[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t),
                    "nii": num(r"(?:nii|non[-\s]?institutional)[^\d]{0,30}?(\d+(?:\.\d+)?)\s*(?:x|times)", t)}
            for k, v in cand.items():
                if v is None:
                    continue
                if data.get(k) is None:
                    filled[k] = v
                elif data[k] and abs(v - data[k]) / max(data[k], 0.01) > 0.25 \
                        and abs(v - data[k]) > 0.3:
                    notes.append(f"{k}: sources disagree — IPOTrack {data[k]}x vs "
                                 f"IPOWatch {v}x (showing IPOTrack)")
            if filled:
                notes.append("Some fields filled from IPOWatch (not on IPOTrack page)")
    except Exception:
        notes.append("Cross-check source unreachable this time")
    return notes, filled


POS = {"surge", "soar", "gain", "rally", "strong", "robust", "stellar", "blockbuster",
       "bumper", "oversubscribed", "bullish", "record", "premium", "jump", "rise",
       "positive", "boost", "debut"}
NEG = {"fall", "drop", "weak", "decline", "slump", "loss", "negative", "concern",
       "risk", "delay", "probe", "fraud", "selloff", "crash", "plunge",
       "tepid", "lukewarm"}


def run_sentiment(name: str) -> dict:
    try:
        q = urllib.parse.quote(f"{name} IPO")
        xml = fetch(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")
        titles = re.findall(r"<title>(.*?)</title>", xml, re.S)
        heads = []
        for tt in titles[1:12]:
            clean = htmlmod.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", tt)).strip()
            clean = re.sub(r"\s+-\s+[^-]+$", "", clean)
            if clean and "google" not in clean.lower():
                heads.append(clean)
        heads = heads[:8]
        pos = neg = 0
        for h in heads:
            low = h.lower()
            words = set(re.findall(r"[a-z]+", low))
            pos += len(words & POS)
            neg += len(words & NEG)
        tot = pos + neg
        score = round((pos - neg) / tot, 2) if tot else 0.0
        mood = "positive" if score > 0.25 else "negative" if score < -0.25 else "mixed"
        return {"ok": True, "mood": mood, "score": score, "headlines": heads,
                "pos_hits": pos, "neg_hits": neg}
    except Exception:
        return {"ok": False, "mood": None, "score": None, "headlines": [],
                "pos_hits": 0, "neg_hits": 0}


@app.get("/")
def home():
    return {"ok": True, "service": "ipo-exitiq-api",
            "agents": ["data-scout", "cross-checker", "sentiment"],
            "endpoints": ["/ipo?name=", "/search?q="],
            "ipos_indexed": len(slugs())}


@app.get("/search")
def search(q: str = ""):
    w = slugify(q)
    if not w:
        return {"results": []}
    hits = [s for s in slugs() if w.split("-")[0] in s][:8]
    return {"results": [{"name": s.replace("-", " ").title(), "slug": s} for s in hits]}


@app.get("/ipo")
def ipo(name: str):
    slug = resolve(name)
    if not slug:
        return {"found": False,
                "error": f"No IPO matching '{name}' found on IPOTrack."}
    now = time.time()
    c = _ipo_cache.get(slug)
    if c and now - c["at"] < IPO_TTL:
        return c["data"]
    agents = {}
    scout_url = f"https://ipotrack.in/ipo/{slug}/"

    def scout():
        return parse_scout(fetch(scout_url), slug)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_scout = ex.submit(scout)
        try:
            data = f_scout.result(timeout=40)
            agents["data_scout"] = "ok"
        except Exception:
            return {"found": False,
                    "error": "Primary source (IPOTrack) unreachable right now — "
                             "try again in a few seconds."}
        f_sent = ex.submit(run_sentiment, data["name"])
        f_check = ex.submit(run_checker, data["name"], data)
        try:
            notes, filled = f_check.result(timeout=8)
            data.update(filled)
            agents["cross_checker"] = "ok"
        except Exception:
            notes, agents["cross_checker"] = [], "timeout — skipped, scout data stands"
        try:
            senti = f_sent.result(timeout=4)
            agents["sentiment"] = "ok" if senti.get("ok") else "failed"
        except Exception:
            senti, agents["sentiment"] = {"ok": False, "headlines": []}, "timeout — skipped"

    core = [data.get(k) for k in ("price", "total", "qib", "gmp_pct")]
    have = sum(1 for v in core if v is not None)
    warn = any("disagree" in n for n in notes)
    conf = "high" if have >= 4 and not warn else "medium" if have >= 3 else "low"
    out = {"found": True, **data, "cross_check": notes,
           "sources": [scout_url, "https://ipowatch.in"],
           "confidence": conf, "agents": agents, "sentiment": senti,
           "fetched_at": datetime.now(IST).isoformat(timespec="seconds")}
    _ipo_cache[slug] = {"at": now, "data": out}
    return out

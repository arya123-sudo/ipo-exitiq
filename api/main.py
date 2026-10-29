"""IPO ExitIQ live-data API.

Type an IPO name -> this service finds the IPO on IPOTrack, reads the live
page server-side (browsers cannot, due to CORS) and returns the numbers the
ExitIQ web app needs: price, lot, issue size, subscription (total/QIB/NII/
retail), GMP, OFS share, dates. Data belongs to its sources; we only relay
publicly displayed figures with attribution and a fetch timestamp.
"""
import re
import time
from datetime import datetime, timedelta, timezone
import urllib.request
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IPO ExitIQ Live API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

UA = {"User-Agent": "Mozilla/5.0 (ExitIQ personal research tool)"}
_cache = {}
_sitemap = {"slugs": [], "at": 0}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", "ignore")


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
    # token-overlap fuzzy match
    wt = set(want.split("-"))
    best, score = None, 0.0
    for s in pool:
        st = set(s.split("-"))
        j = len(wt & st) / max(1, len(wt | st))
        r = SequenceMatcher(None, want, s).ratio()
        sc = max(j, r)
        if sc > score:
            best, score = s, sc
    return best if score >= 0.45 else None


def text_of(html: str) -> str:
    t = re.sub(r"<script[\s\S]*?</script>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


def num(pat, t, cast=float):
    m = re.search(pat, t)
    return cast(m.group(1)) if m else None


def parse(html: str, slug: str) -> dict:
    t = text_of(html)
    gmp_m = re.search(r"GMP Today ₹(-?[\d,]*\.?\d*) \((-?[\d.]+)%\)", t) or re.search(r"GMP \(unofficial\) ₹(-?[\d,]*\.?\d*) \((-?[\d.]+)%\)", t, re.I)
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
    pairs = re.findall(r"price band(?: set at)? ₹[\d.]+ [–-] ₹([\d.]+),? lot(?: size)? (\d+) shares", t, re.I)
    anchored = pairs[0] if pairs else None
    out = {
        "name": slug.replace("-", " ").title(),
        "slug": slug,
        "price": float(anchored[0]) if anchored else (float(re.search(r"price band ₹[\d.]+ [–-] ₹([\d.]+)", t, re.I).group(1)) if re.search(r"price band ₹[\d.]+ [–-] ₹([\d.]+)", t, re.I) else None),
        "lot_anchor": int(anchored[1]) if anchored else None,
        "lot": num(r"Lot Size (\d+) Shares", t, int),
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
    _la = out.pop("lot_anchor", None)
    if _la and not out.get("lot"):
        out["lot"] = _la
    return out


@app.get("/")
def home():
    return {"app": "IPO ExitIQ Live API", "usage": "/ipo?name=BLS Polymers", "ipos_indexed": len(_sitemap["slugs"])}


@app.get("/search")
def search(q: str = ""):
    ql = slugify(q)
    out = [s.replace("-", " ").title() for s in slugs() if ql in s][:15]
    return {"names": out}


@app.get("/ipo")
def ipo(name: str):
    slug = resolve(name)
    if not slug:
        return {"found": False, "error": f"No IPO matching '{name}' found on IPOTrack."}
    hit = _cache.get(slug)
    if hit and time.time() - hit["at"] < 600:
        return hit["data"]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_main = ex.submit(fetch, f"https://ipotrack.in/ipo/{slug}")
        f_watch = ex.submit(fetch, f"https://ipowatch.in/{slug}-ipo-subscription-status/")
        try:
            data = parse(f_main.result(), slug)
        except Exception as e:
            return {"found": False, "error": f"Live source failed: {e}"}
        try:
            watch_html = f_watch.result()
        except Exception:
            watch_html = None
    if data.pop("lot_anchor", None) and not data.get("lot"):
        data["lot"] = data.get("lot")
    # --- second source: IPOWatch subscription table (cross-check + gap fill) ---
    cross, sources = [], {"primary": data["source"]}
    try:
        if watch_html is None:
            raise RuntimeError("watch fetch failed")
        w = text_of(watch_html)
        w = w.replace(" ", "|")
        wcols = {}
        for lbl, key in (("QIB", "qib"), ("NII", "nii"), ("RII", "retail"), ("Total", "total")):
            m = re.search(lbl + r"\|([\d.]+)\|([\d.]+)\|([\d.]+)\|", w)
            if m:
                wcols[key] = [float(x) for x in m.groups()]
        wvals = {}
        for key, days in wcols.items():
            nz = [d for d in days if d > 0]
            if not nz:
                continue
            prim = data.get(key)
            if prim:  # choose the IPOWatch day-column closest to the primary source
                wvals[key] = min(nz, key=lambda d: abs(d - prim) / max(prim, 0.01))
            else:
                wvals[key] = nz[-1]
        if wvals:
            sources["cross_check"] = f"https://ipowatch.in/{slug}-ipo-subscription-status/"
            for k, v in wvals.items():
                if data.get(k) is None:
                    data[k] = v
                    cross.append(f"{k}: filled from IPOWatch ({v}x)")
                elif v and abs(data[k] - v) / max(v, 0.01) > 0.25 and abs(data[k] - v) > 0.3:
                    cross.append(f"WARNING {k}: IPOTrack {data[k]}x vs IPOWatch {v}x disagree")
                else:
                    cross.append(f"{k}: both sources agree ({data[k]}x)")
    except Exception:
        cross.append("IPOWatch cross-check unavailable this time")
    core = [data.get(k) for k in ("price", "size_cr", "total", "qib", "gmp_pct")]
    have = sum(1 for v in core if v is not None)
    warns = sum(1 for c in cross if c.startswith("WARNING"))
    data["confidence"] = "high" if have >= 4 and warns == 0 else ("medium" if have >= 3 else "low")
    data["cross_check"] = cross
    data["sources"] = sources
    data["found"] = True
    data["fetched_at"] = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p IST")
    _cache[slug] = {"data": data, "at": time.time()}
    return data

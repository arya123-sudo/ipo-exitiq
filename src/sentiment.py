"""NLP sentiment v1: lexicon scorer (offline, no deps). v2 = FinBERT."""
POS = {"surge":2,"rally":2,"gain":1,"strong":1,"oversubscribed":2,"robust":1,"growth":1,
"profit":1,"beat":1,"upgrade":1,"bullish":2,"record":1,"premium":1,"demand":1,"positive":1,"wins":1,"order":1}
NEG = {"fall":-1,"crash":-2,"plunge":-2,"loss":-1,"debt":-1,"fraud":-2,"probe":-2,"weak":-1,
"decline":-1,"risk":-1,"bearish":-2,"downgrade":-2,"negative":-1,"lawsuit":-2,"default":-2,"selloff":-2,"slump":-1}

def sentiment_score(texts):
    """texts: list[str] headlines -> score -1..+1 and label"""
    total, hits = 0, 0
    for t in texts:
        for w in t.lower().replace(",", " ").replace(".", " ").split():
            if w in POS: total += POS[w]; hits += 1
            elif w in NEG: total += NEG[w]; hits += 1
    if hits == 0: return 0.0, "NEUTRAL (no signal words)"
    score = max(-1.0, min(1.0, total / (hits * 1.5)))
    label = "POSITIVE" if score > 0.15 else "NEGATIVE" if score < -0.15 else "NEUTRAL"
    return round(score, 3), label

import yaml, sys
from collections import Counter
import re

def kws(text):
    # grab 1–3 word phrases preferring content words
    words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)]
    STOP = set("""
        the and or nor but not yes no in on at to from for with within without between into across about
        is are was were be been being of as by this that those these there here then than so if while
        a an it its itself themselves ourselves yourself yourselves myself ourselves each any some many
        more most less least very even just only still also already ever never again
        do does did doing done can could may might shall should will would
    """.split())
    words = [w for w in words if w not in STOP and len(w) >= 4]
    # bigrams/trigrams preferred
    bigrams  = [" ".join([words[i], words[i+1]]) for i in range(len(words)-1)]
    trigrams = [" ".join([words[i], words[i+1], words[i+2]]) for i in range(len(words)-2)]
    return trigrams + bigrams + words

def distinct_questions(supports):
    qs = {(s.get("q") or "").strip() for s in supports if s.get("q")}
    return len(qs)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 detect_candidate_axioms.py <eval_yaml> <candidates_out_yaml>")
        sys.exit(1)

    ev_path, out_path = sys.argv[1], sys.argv[2]
    data = yaml.safe_load(open(ev_path, "r"))

    # pick novel-ish & non-contradictory frontier
    pool = [
        d for d in data
        if d["novelty_score"] >= 0.48
        and d["stance"] in {"uncertain","related"}
        and not d["contradiction"]
    ]

    # tally keywords/phrases
    all_kw = []
    for d in pool:
        all_kw += kws((d.get("q","") + " " + d.get("a","")).strip())

    counts = Counter(all_kw).most_common(50)
    themes = [k for k,_ in counts[:3]]

    # discard junk heads; prefer phrases; drop generic singletons
    BAD_HEADS = {"not","but","and","only","also","just","then","than","because"}
    themes = [t for t in themes if t.split()[0] not in BAD_HEADS]

    GENERIC = {"life","everything","something","anything","nothing","thing","things","world","people"}
    filtered = []
    for t in themes:
        parts = t.split()
        if len(parts) >= 2:
            filtered.append(t)
        else:
            if t not in GENERIC:
                filtered.append(t)
    themes = filtered

    if not themes:
        yaml.dump([], open(out_path,"w"), allow_unicode=True)
        print(f"✅ Wrote candidate axioms → {out_path} (found: 0)")
        sys.exit(0)

    # assemble candidates
    cands = []
    for t in themes:
        supports = [d for d in pool if t in kws((d.get("q","")+" "+d.get("a","")).strip())]

        # require ≥3 distinct questions to avoid one-phrase repeats
        if distinct_questions(supports) < 3:
            continue

        avg_coh = sum(s["coherence_score"] for s in supports)/len(supports)
        avg_nov = sum(s["novelty_score"]   for s in supports)/len(supports)

        stmt = f"({t}) appears as a recurring relational pattern not fully captured by current axioms; it may express a bridging constraint within Fractal Monism."
        cands.append({
            "proposed_id": None,
            "title": f"Candidate: {t.capitalize()} Principle",
            "statement": stmt,
            "support_count": len(supports),
            "avg_coherence": round(avg_coh,4),
            "avg_novelty":  round(avg_nov,4),
            "evidence": [{
                "q": s["q"], "a": s["a"],
                "coherence_score": s["coherence_score"],
                "novelty_score":   s["novelty_score"],
                "top_axioms":      s["top_axioms"]
            } for s in supports[:10]],
            "status": "proposed"
        })

    yaml.dump(cands, open(out_path,"w"), allow_unicode=True)
    print(f"✅ Wrote candidate axioms → {out_path} (found: {len(cands)})")

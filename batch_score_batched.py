import sys, yaml, requests
from openai import OpenAI

QDRANT_URL   = "http://localhost:6333"
AXIOM_COLL   = "core_axioms"
EMBED_MODEL  = "text-embedding-3-large"  # not used; we reuse vectors
client = OpenAI()

def scroll_vectors(coll, limit=1000, offset=None):
    payload = {"with_payload": True, "with_vectors": True, "limit": limit}
    if offset is not None:
        payload["offset"] = offset
    r = requests.post(f"{QDRANT_URL}/collections/{coll}/points/scroll", json=payload).json()
    pts = r.get("result", {}).get("points", [])
    next_off = r.get("result", {}).get("next_page_offset")
    return pts, next_off

def search_axioms(vec, top=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{AXIOM_COLL}/points/search",
        json={"vector": vec, "top": top, "with_payload": True}
    ).json()
    return r.get("result", [])

def stance_from(best):
    if best >= 0.70:
        return "aligned", False
    if best >= 0.50:
        return "related", False
    if best >= 0.30:
        return "uncertain", False
    if best >= 0.22:
        return "scale_overlap", False   # formerly potential_contradiction
    return "potential_contradiction", True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 batch_score_batched.py <collection_name> <out_eval.yaml> [--batch 2000] [--max 0]")
        sys.exit(1)

    coll = sys.argv[1]
    outp = sys.argv[2]
    BATCH = 2000
    MAX   = 0   # 0 = no limit

    # optional args
    for i,a in enumerate(sys.argv):
        if a == "--batch" and i+1 < len(sys.argv):
            BATCH = int(sys.argv[i+1])
        if a == "--max" and i+1 < len(sys.argv):
            MAX = int(sys.argv[i+1])

    written = 0
    offset = None
    results = []

    # -------- batched scroll/score loop (INDENTED under __main__) --------
    while True:
        pts, offset = scroll_vectors(coll, limit=BATCH, offset=offset)
        if not pts: break
        if MAX and (written + len(pts) > MAX):
            pts = pts[: MAX - written]
            offset = None

        # score the chunk
        for p in pts:
            vec = p.get("vector")
            pl  = p.get("payload", {}) or {}
            if vec is None:
                continue

            ax = search_axioms(vec, top=5)
            best = max((a0.get("score", 0.0) for a0 in ax), default=0.0)
            stance, contradiction = stance_from(best)

            q = pl.get("question","")
            a = pl.get("answer","")

            # tolerant top_axioms: handle either item_id / id / title (no KeyError)
            ta = []
            for a0 in ax:
                pp = a0.get("payload", {}) or {}
                aid = pp.get("axiom_id") or pp.get("item_id") or pp.get("id") or pp.get("title") or "axiom"
                ta.append({
                    "axiom_id": aid,
                    "score": round(float(a0.get("score", 0.0)), 4),
                    "title": pp.get("title",""),
                    "statement": pp.get("statement","")
                })

            results.append({
                "q": q, "a": a,
                "coherence_score": round(best,4),
                "novelty_score": round(1-best,4),
                "stance": stance,
                "contradiction": contradiction,
                "top_axioms": ta
            })

        written += len(pts)
        # write progress every chunk
        yaml.dump(results, open(outp,"w"), allow_unicode=True)
        print(f"wrote {written} items → {outp}")
        if not offset: break

    print(f"✅ Done. Total scored: {written}. Output: {outp}")

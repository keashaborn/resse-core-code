import os, sys, yaml, requests
from openai import OpenAI

QDRANT_URL = "http://localhost:6333"
AXIOMS = "core_axioms"
EMBED_MODEL = "text-embedding-3-large"
client = OpenAI()

def embed(t): 
    return client.embeddings.create(input=[t], model=EMBED_MODEL).data[0].embedding

def search(coll, vec, top=5):
    return requests.post(f"{QDRANT_URL}/collections/{coll}/points/search",
                         json={"vector": vec, "top": top, "with_payload": True}).json()["result"]

def scroll_all(coll, page=256):
    # light scroll
    off = None; out=[]
    while True:
        payload = {"with_payload": True, "limit": page}
        if off is not None: payload["offset"] = off
        r = requests.post(f"{QDRANT_URL}/collections/{coll}/points/scroll", json=payload).json()
        pts = r.get("result",{}).get("points",[])
        if not pts: break
        out.extend(pts)
        off = r.get("result",{}).get("next_page_offset")
        if not off: break
    return out

def stance_from(best):
    if best >= 0.70: return "aligned", False
    if best >= 0.50: return "related", False
    if best >= 0.30: return "uncertain", False
    return "potential_contradiction", True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 batch_score_against_axioms.py <collection_name> <output_yaml>")
        sys.exit(1)

    coll, out_path = sys.argv[1], sys.argv[2]
    points = scroll_all(coll)
    results = []

    for p in points:
        pl = p["payload"]
        q = (pl.get("question") or "").strip()
        a = (pl.get("answer") or "").strip()
        text = f"Q: {q}\nA: {a}" if a else q
        vec = embed(text)
        ax = search(AXIOMS, vec, top=5)
        best = max((x["score"] for x in ax), default=0.0)
        stance, contradiction = stance_from(best)
        results.append({
            "source_collection": coll,
            "q": q, "a": a,
            "coherence_score": round(best,4),
            "novelty_score": round(1-best,4),
            "stance": stance,
            "contradiction": contradiction,
            "top_axioms": [
                {"axiom_id": x["payload"]["axiom_id"], "score": round(x["score"],4)}
                for x in ax
            ]
        })

    with open(out_path, "w") as f:
        yaml.dump(results, f, allow_unicode=True)

    print(f"✅ Wrote batch evaluation → {out_path} (items: {len(results)})")

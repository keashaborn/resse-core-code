# build_retrieval_weights.py
import yaml, hashlib, sys, json

if len(sys.argv)<3:
    print("Usage: python3 build_retrieval_weights.py <eval.yaml> <weights.json>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(inp))
W = {}
for r in d:
    q=(r.get("q") or "").strip()
    a=(r.get("a") or "").strip()
    key=f"Q:{q}|A:{a}"
    h=hashlib.sha1(key.encode("utf-8")).hexdigest()
    # use coherence_score as the weight
    W[h] = float(r.get("coherence_score", 0.0))

json.dump(W, open(outp,"w"))
print("wrote", len(W), "weights →", outp)


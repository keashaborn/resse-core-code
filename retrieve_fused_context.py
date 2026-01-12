import requests
from openai import OpenAI
import hashlib, json, os

CORE_WEIGHTS = {}
if os.path.exists("core_weights.json"):
    CORE_WEIGHTS = json.load(open("core_weights.json"))

def hash_qa(payload):
    q=(payload.get("question") or "").strip()
    a=(payload.get("answer")   or "").strip()
    return hashlib.sha1(f"Q:{q}|A:{a}".encode("utf-8")).hexdigest()

def weight_of(payload):
    # default neutral weight if not found
    return CORE_WEIGHTS.get(hash_qa(payload), 0.5)


QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-large"
COLLS = [("core_axioms",3), ("supporting_principles",2), ("foundational_core",5)]

client = OpenAI()

def embed(txt):
    return client.embeddings.create(input=[txt], model=EMBED_MODEL).data[0].embedding

def search(coll, vec, top):
    r = requests.post(f"{QDRANT_URL}/collections/{coll}/points/search",
                      json={"vector": vec, "top": top, "with_payload": True}).json()
    return r.get("result", [])

def block(name, hits):
    out = [f"\n=== {name} ==="]
    for i,h in enumerate(hits,1):
        p=h["payload"]
        if name=="Axioms":
            out.append(f"[{i}] ({p.get('axiom_id','')}) {p.get('title','')}: {p.get('statement','')}")
        elif name=="Supporting":
            title = p.get("title") or (p.get("statement","").strip().split("\n", 1)[0][:80]) or p.get("item_id","(untitled)")
            stmt  = p.get("statement","").strip()
            out.append(f"[{i}] {title}: {stmt}")        
        else:
            out.append(f"[{i}] Q: {p.get('question','')}\n    A: {p.get('answer','')}")
    return "\n".join(out)

if __name__ == "__main__":
    q = input("Ask RESSE (fusion): ").strip()
    v = embed(q)
    ax = search("core_axioms", v, 3)
    sp = search("supporting_principles", v, 2)
    fd = search("foundational_core", v, 5)
    print(block("Axioms", ax))
    print(block("Supporting", sp))
    print(block("Foundational", fd))


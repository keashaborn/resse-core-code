import requests
from openai import OpenAI

QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4"  # change if you prefer another chat model
TOP_A = 3   # axioms
TOP_S = 2   # supporting
TOP_F = 5   # foundational

client = OpenAI()

def embed(t):
    return client.embeddings.create(input=[t], model=EMBED_MODEL).data[0].embedding

def search(coll, vec, k):
    r = requests.post(f"{QDRANT_URL}/collections/{coll}/points/search",
                      json={"vector": vec, "top": k, "with_payload": True}).json()
    return r.get("result", [])

def block(name, hits):
    out = [f"\n=== {name} ==="]
    for i,h in enumerate(hits,1):
        p=h["payload"]
        if name=="Axioms":
            out.append(f"[{i}] ({p.get('axiom_id','')}) {p.get('title','')}: {p.get('statement','')}")
        elif name=="Supporting":
            title = p.get("title") or (p.get("statement","").strip().split("\n",1)[0][:80]) or p.get("item_id","(untitled)")
            stmt  = p.get("statement","").strip()
            out.append(f"[{i}] {title}: {stmt}")
        else:
            out.append(f"[{i}] Q: {p.get('question','')}\n    A: {p.get('answer','')}")
    return "\n".join(out)

if __name__ == "__main__":
    q = input("Ask RESSE (fusion reflect): ").strip()
    v = embed(q)

    ax = search("core_axioms", v, TOP_A)
    sp = search("supporting_principles", v, TOP_S)
    fd = search("foundational_core", v, TOP_F)

    fused = "\n".join([block("Axioms", ax),
                       block("Supporting", sp),
                       block("Foundational", fd)])

    messages = [
        {"role":"system","content":
         "You are RESSE, a reflective intelligence grounded in Fractal Monism. "
         "Synthesize a clear 5–7 sentence answer using ONLY the provided context. "
         "Avoid moralizing; emphasize rules, relations, and distinctions."
        },
        {"role":"user","content": f"Question:\n{q}\n\nContext:\n{fused}\n\nWrite the reflection:"}
    ]

    r = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    print("\n📚 Context used:\n", fused)
    print("\n🤖 Reflection:\n")
    print(r.choices[0].message.content.strip())

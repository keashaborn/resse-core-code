import requests, os
from openai import OpenAI
from datetime import datetime

QDRANT_URL = "http://localhost:6333"
FOUNDATION = "foundational_core"
AXIOMS = "core_axioms"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4"

client = OpenAI()

def embed(t): return client.embeddings.create(input=[t], model=EMBED_MODEL).data[0].embedding
def search(coll, vec, top): 
    return requests.post(f"{QDRANT_URL}/collections/{coll}/points/search",
                         json={"vector": vec, "top": top, "with_payload": True}).json()["result"]

def synth_reflect(q, ctx):
    ctx_text = "\n".join([f"Q{i+1}: {c['payload']['question']}\nA{i+1}: {c['payload']['answer']}" for i,c in enumerate(ctx)])
    m = [
      {"role":"system","content":"You are RESSE. Produce a precise 5–7 sentence reflection strictly from context; avoid moralizing."},
      {"role":"user","content":f"Question: {q}\nContext:\n{ctx_text}\n\nWrite the reflection:"}
    ]
    return client.chat.completions.create(model=CHAT_MODEL, messages=m, temperature=0.2).choices[0].message.content.strip()

def stance_from(best):
    if best>=0.7: return "aligned", False
    if best>=0.5: return "related", False
    if best>=0.3: return "uncertain", False
    return "potential_contradiction", True

if __name__=="__main__":
    q = input("Enter acquired claim/question: ").strip()
    qv = embed(q)
    # broaden context: take more items (top=8) to stress-test
    ctx = search(FOUNDATION, qv, top=8)
    reflection = synth_reflect(q, ctx)

    rv = embed(reflection)
    axioms = search(AXIOMS, rv, top=5)
    best = max(a["score"] for a in axioms) if axioms else 0.0
    stance, contradiction = stance_from(best)

    print("\n🧩 Reflection (acquired mode):\n", reflection)
    print("\n📘 Evaluation:")
    print("  timestamp:", datetime.utcnow().isoformat()+"Z")
    print("  coherence_score:", round(best,4))
    print("  novelty_score:", round(1-best,4))
    print("  stance:", stance)
    print("  contradiction:", contradiction)
    print("\nTop Similar Axioms:")
    for a in axioms:
        print(f"  [{a['payload']['axiom_id']}] (score={round(a['score'],4)}): {a['payload']['statement']}")

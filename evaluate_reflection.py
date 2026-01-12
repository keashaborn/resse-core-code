# evaluate_reflection.py  (clean full version)

import os, requests, uuid, yaml
from datetime import datetime
from openai import OpenAI

QDRANT_URL = "http://localhost:6333"
AXIOM_COLLECTION = "core_axioms"
FOUNDATION_COLLECTION = "foundational_core"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4"

client = OpenAI()

# ---------- helpers ----------
def embed_text(text):
    r = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return r.data[0].embedding

def qdrant_search(collection, vector, top=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{collection}/points/search",
        json={"vector": vector, "top": top, "with_payload": True}
    )
    return r.json().get("result", [])

# ---------- evaluator ----------
def evaluate_against_axioms(reflection):
    vec = embed_text(reflection)
    results = qdrant_search(AXIOM_COLLECTION, vec, top=5)
    if not results:
        return {"error": "No results from Qdrant."}

    # compute stance bands
    scores = [(r["payload"]["axiom_id"], r["score"]) for r in results]
    best = max(s[1] for s in scores)
    novelty = round(1 - best, 4)
    if best >= 0.7:
        stance, contradiction = "aligned", False
    elif best >= 0.5:
        stance, contradiction = "related", False
    elif best >= 0.3:
        stance, contradiction = "uncertain", False
    else:
        stance, contradiction = "potential_contradiction", True

    return {
        "reflection": reflection,
        "evaluation": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "coherence_score": round(best, 4),
            "novelty_score": novelty,
            "stance": stance,
            "contradiction": contradiction
        },
        "similar_axioms": [
            {
                "axiom_id": r["payload"]["axiom_id"],
                "score": round(r["score"], 4),
                "statement": r["payload"]["statement"]
            } for r in results
        ]
    }

# ---------- main ----------
if __name__ == "__main__":
    user_input = input("Enter a reflection or question: ").strip()
    is_question = user_input.endswith("?") or user_input.lower().split(" ")[0] in {"why","how","what","does","can","is"}

    # If it's a question, synthesize a reflection first
    if is_question:
        print("Synthesizing reflection from foundational memory...")
        q_vec = embed_text(user_input)
        ctx = qdrant_search(FOUNDATION_COLLECTION, q_vec, top=5)
        context_text = "\n".join(
            [f"Q{i+1}: {c['payload']['question']}\nA{i+1}: {c['payload']['answer']}" for i, c in enumerate(ctx)]
        )
        messages = [
            {"role":"system","content":"You are RESSE, a reflective intelligence grounded in Fractal Monism."},
            {"role":"user","content":f"Question: {user_input}\nContext:\n{context_text}\n\nWrite a concise reflection (5-7 sentences):"}
        ]
        reflection = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3).choices[0].message.content.strip()
        print("\n🧩 Synthesized Reflection:\n", reflection)
    else:
        reflection = user_input

    report = evaluate_against_axioms(reflection)

    print("\n📘 Evaluation Result")
    for k,v in report["evaluation"].items():
        print(f"  {k}: {v}")

    print("\nTop Similar Axioms:")
    for ax in report["similar_axioms"]:
        print(f"  [{ax['axiom_id']}] (score={ax['score']}): {ax['statement']}")


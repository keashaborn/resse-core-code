# evaluate_reflection.py

import os
import requests
import uuid
from openai import OpenAI
from datetime import datetime

QDRANT_URL = "http://localhost:6333"
AXIOM_COLLECTION = "core_axioms"
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()

def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def retrieve_axioms(vector, top_k=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{AXIOM_COLLECTION}/points/search",
        json={
            "vector": vector,
            "top": top_k,
            "with_payload": True
        }
    )
    return r.json()["result"]

if __name__ == "__main__":
    user_input = input("Enter a reflection to evaluate: ").strip()

    # Heuristic: if it ends with '?' or starts with 'why/how/what', treat as a question
    is_question = user_input.endswith("?") or user_input.lower().split(" ")[0] in {"why","how","what","does","can","is"}

    if is_question:
        # 1) retrieve context from foundational_core
        from openai import OpenAI
        import requests
        client = OpenAI()

        # embed question and search
        q_vec = embed_text(user_input)
        r = requests.post(
            f"{QDRANT_URL}/collections/foundational_core/points/search",
            json={"vector": q_vec, "top": 5, "with_payload": True}
        ).json()["result"]

        # format brief context
        ctx = ""
        for i, p in enumerate(r):
            ctx += f"\nQ{i+1}: {p['payload']['question']}\nA{i+1}: {p['payload']['answer']}\n"

        # 2) synthesize a short reflection FIRST, then evaluate that
        messages = [
            {"role":"system","content":"You are RESSE. Synthesize a clear 5–7 sentence reflection strictly from the context, in the Fractal Monism frame."},
            {"role":"user","content": f"Question: {user_input}\nContext:\n{ctx}\n\nWrite the reflection:"}
        ]
        reflection = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.3).choices[0].message.content.strip()
        text_to_score = reflection
        print("\n🧩 Synthesized reflection to evaluate:\n", reflection)
    else:
        text_to_score = user_input

    report = evaluate_against_axioms(text_to_score)

    print("\n📘 Evaluation Result:")
    print("Reflection:", report["reflection"])
    print("\nEvaluation:")
    for k, v in report["evaluation"].items():
        print(f"  {k}: {v}")
    print("\nTop Similar Axioms:")
    for ax in report["similar_axioms"]:
        print(f"  [{ax['axiom_id']}] (score={ax['score']}): {ax['statement']}")
if __name__ == "__main__":
    user_input = input("Enter a reflection to evaluate: ").strip()

    # Heuristic: if it ends with '?' or starts with 'why/how/what', treat as a question
    is_question = user_input.endswith("?") or user_input.lower().split(" ")[0] in {"why","how","what","does","can","is"}

    if is_question:
        # 1) retrieve context from foundational_core
        from openai import OpenAI
        import requests
        client = OpenAI()

        # embed question and search
        q_vec = embed_text(user_input)
        r = requests.post(
            f"{QDRANT_URL}/collections/foundational_core/points/search",
            json={"vector": q_vec, "top": 5, "with_payload": True}
        ).json()["result"]

        # format brief context
        ctx = ""
        for i, p in enumerate(r):
            ctx += f"\nQ{i+1}: {p['payload']['question']}\nA{i+1}: {p['payload']['answer']}\n"

        # 2) synthesize a short reflection FIRST, then evaluate that
        messages = [
            {"role":"system","content":"You are RESSE. Synthesize a clear 5–7 sentence reflection strictly from the context, in the Fractal Monism frame."},
            {"role":"user","content": f"Question: {user_input}\nContext:\n{ctx}\n\nWrite the reflection:"}
        ]
        reflection = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.3).choices[0].message.content.strip()
        text_to_score = reflection
        print("\n🧩 Synthesized reflection to evaluate:\n", reflection)
    else:
        text_to_score = user_input

    report = evaluate_against_axioms(text_to_score)

    print("\n📘 Evaluation Result:")
    print("Reflection:", report["reflection"])
    print("\nEvaluation:")
    for k, v in report["evaluation"].items():
        print(f"  {k}: {v}")
    print("\nTop Similar Axioms:")
    for ax in report["similar_axioms"]:
        print(f"  [{ax['axiom_id']}] (score={ax['score']}): {ax['statement']}")
import yaml, os
ev = report
out_file = "evaluations_log.yaml"
if os.path.exists(out_file):
    with open(out_file, "r") as f:
        data = yaml.safe_load(f) or []
else:
    data = []
data.append(ev)
with open(out_file, "w") as f:
    yaml.dump(data, f, allow_unicode=True)
print("\n📝 Saved evaluation to evaluations_log.yaml")

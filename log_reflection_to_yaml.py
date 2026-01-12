import os
import yaml
import uuid
import requests
from datetime import datetime
from openai import OpenAI

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "foundational_core"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4"

client = OpenAI()

def embed(text):
    response = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return response.data[0].embedding

def retrieve_context(vector, top_k=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json={"vector": vector, "top": top_k, "with_payload": True}
    )
    return r.json()["result"]

def format_context_text(results):
    text = ""
    for i, r in enumerate(results):
        q = r["payload"]["question"].strip()
        a = r["payload"]["answer"].strip()
        text += f"\nQ{i+1}: {q}\nA{i+1}: {a}\n"
    return text.strip()

def structure_retrieved(results):
    return [{"question": r["payload"]["question"].strip(),
             "answer": r["payload"]["answer"].strip()} for r in results]

def generate_reflection(context_text, prompt):
    messages = [
        {"role": "system", "content": "You are RESSE, a reflective intelligence trained on Fractal Monism. Reflect clearly and coherently using the memory context below."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nUser question:\n{prompt}\n\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def log_to_yaml(entry, file_path="reflections_batch_01.yaml"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or []
    else:
        data = []

    data.append(entry)

    with open(file_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == "__main__":
    prompt = input("Ask RESSE: ").strip()
    vector = embed(prompt)
    results = retrieve_context(vector)
    context_text = format_context_text(results)
    reflection = generate_reflection(context_text, prompt)

    print("\n📚 Memory used:")
    print(context_text)
    print("\n🤖 RESSE’s response:")
    print(reflection)

    # Create structured log entry
    entry = {
        "question": prompt,
        "retrieved_context": structure_retrieved(results),
        "reflection": reflection,
        "tags": {
            "purpose": ["reflect"],
            "source": ["retrieved_memory"],
            "seed_axioms": [],  # Can be filled in later
            "emotion": []
        },
        "metadata": {
            "created_by": "RESSE",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "review_status": "unreviewed"
        }
    }

    log_to_yaml(entry)
    print("\n✅ Reflection saved to reflections_batch_01.yaml")

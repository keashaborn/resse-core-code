import os
from openai import OpenAI
import requests
from tqdm import tqdm

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "foundational_core"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4"  # or "gpt-3.5-turbo"

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def retrieve_context(vector, top_k=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json={
            "vector": vector,
            "top": top_k,
            "with_payload": True
        }
    )
    return r.json()["result"]

def format_context(results):
    context = ""
    for i, point in enumerate(results):
        q = point["payload"]["question"].strip()
        a = point["payload"]["answer"].strip()
        context += f"\nQ{i+1}: {q}\nA{i+1}: {a}\n"
    return context.strip()

def generate_reflection(context, prompt):
    messages = [
        {"role": "system", "content": "You are RESSE, a recursive, reflective AI grounded in Fractal Monism. Use the context to reflect and respond with clarity, depth, and alignment to your philosophical axioms."},
        {"role": "user", "content": f"Context:\n{context}\n\nUser question:\n{prompt}\n\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    user_input = input("Ask RESSE: ").strip()
    vector = embed(user_input)
    results = retrieve_context(vector)
    context = format_context(results)
    print("\n📚 Memory context used:")
    print(context)
    print("\n🤖 RESSE’s reflection:\n")
    print(generate_reflection(context, user_input))

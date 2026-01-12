import os
import requests
from openai import OpenAI
from tqdm import tqdm

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "foundational_core"
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def search_qdrant(vector, top_k=5):
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json={
            "vector": vector,
            "top": top_k,
            "with_payload": True
        }
    )
    return r.json()

if __name__ == "__main__":
    query = input("Enter a question, idea, or phrase: ").strip()
    print("Embedding and searching...")

    vector = embed(query)
    results = search_qdrant(vector)

    for i, point in enumerate(results["result"]):
        print(f"\n🔹 Result {i+1} (score: {point['score']:.4f}):")
        payload = point["payload"]
        print(f"Q: {payload['question'].strip()}")
        print(f"A: {payload['answer'].strip()}")
        print(f"Tags: {payload.get('tags', {})}")

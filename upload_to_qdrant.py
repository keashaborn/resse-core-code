import yaml
import uuid
import requests
from tqdm import tqdm
from openai import OpenAI
import os

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "foundational_core"
EMBED_MODEL = "text-embedding-3-large"

# Create the OpenAI client once
client = OpenAI()

# Load YAML file
with open("foundational_core_batch_01.yaml.txt", "r") as f:
    items = yaml.safe_load(f)

points = []

for i, item in enumerate(tqdm(items)):
    q = item["question"].strip()
    a = item["answer"].strip()
    content = f"Q: {q}\nA: {a}"

    # Call OpenAI once per entry
    response = client.embeddings.create(
        input=[content],
        model=EMBED_MODEL
    )
    embedding = response.data[0].embedding

    points.append({
        "id": str(uuid.uuid4()),
        "vector": embedding,
        "payload": item
    })

# Upload to Qdrant
r = requests.put(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
    json={"points": points}
)

print("Qdrant response:", r.status_code, r.text)

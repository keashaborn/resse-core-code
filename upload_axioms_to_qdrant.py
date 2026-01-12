import os
import yaml
import uuid
import requests
from openai import OpenAI
from tqdm import tqdm

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "core_axioms"
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()

# Load YAML axioms
with open("core_axioms.yaml", "r") as f:
    axioms = yaml.safe_load(f)

points = []

for axiom in tqdm(axioms):
    axiom_id = axiom["id"]
    statement = axiom["statement"]
    title = axiom["title"]
    
    response = client.embeddings.create(
        input=[statement],
        model=EMBED_MODEL
    )
    vector = response.data[0].embedding

    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": {
            "axiom_id": axiom_id,
            "title": title,
            "statement": statement,
            "type": axiom.get("type", ""),
            "notes": axiom.get("notes", [])
        }
    }


    points.append(point)

# Upload to Qdrant
r = requests.put(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
    json={"points": points}
)

print("Qdrant response:", r.status_code, r.text)

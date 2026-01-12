#!/usr/bin/env python3
"""
upload_core_axioms_v2.py

Create a Qdrant collection 'core_axioms_v2' with:
- 15 canonical axioms (A1–A15)
- all paraphrase variants from core_axioms_paraphrases.yaml

Each point has:
- vector: embedding of the axiom/paraphrase text
- payload: { "axiom_id": "A1", "variant_id": "ax-A1" or "ax-A1-p1", "text": "..." }
"""

import json
import os
from pathlib import Path

import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

BASE_DIR = Path(__file__).parent

# Paths to your actual axiom files
AXIOMS_YAML = BASE_DIR / "core_axioms.yaml"
PARAPHRASES_YAML = BASE_DIR / "datasets/core_axioms_paraphrases.yaml"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # or None if no auth
COLLECTION_NAME = "core_axioms_v2"

EMBED_MODEL = "text-embedding-3-large"  # 3072-dim embeddings

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def load_axioms() -> list[dict]:
    """Load canonical axioms from core_axioms.yaml (15 axioms)."""
    with AXIOMS_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expect structure like: {"dataset": {...}, "items": [ {...}, ... ]}
    items = data.get("items", [])
    axioms = []
    for item in items:
        aid = item.get("axiom_id")
        stmt = item.get("statement") or ""
        if not aid or not stmt:
            continue
        axioms.append({
            "axiom_id": aid,
            "variant_id": item.get("id", aid),
            "text": stmt.strip(),
        })
    return axioms

def load_paraphrases() -> list[dict]:
    """Load paraphrased axioms from core_axioms_paraphrases.yaml."""
    with PARAPHRASES_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("items", [])
    variants = []
    for item in items:
        var_id = item.get("id")
        stmt = item.get("statement") or ""
        # infer axiom_id from id pattern like "ax-A4-p2"
        ax_id = None
        if isinstance(var_id, str):
            # e.g. "ax-A4-p2" -> "A4"
            parts = var_id.split("-")
            if len(parts) >= 2:
                ax_id = parts[1]
        if not ax_id or not stmt:
            continue
        variants.append({
            "axiom_id": ax_id,
            "variant_id": var_id,
            "text": stmt.strip(),
        })
    return variants


def main():
    # Load data
    axioms = load_axioms()
    paraphrases = load_paraphrases()
    all_points = axioms + paraphrases
    print(f"Loaded {len(axioms)} canonical axioms, {len(paraphrases)} paraphrases.")

    # Recreate collection
    print(f"Recreating Qdrant collection '{COLLECTION_NAME}'...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,         # text-embedding-3-large dimension
            distance=Distance.COSINE,
        ),
    )

    # Prepare points for upsert
    vectors = []
    payloads = []
    ids = []

    for idx, entry in enumerate(all_points):
        vec = embed_text(entry["text"])
        vectors.append(vec)
        payloads.append({
            "axiom_id": entry["axiom_id"],
            "variant_id": entry["variant_id"],
            "text": entry["text"],
        })
        ids.append(idx)

        if (idx + 1) % 20 == 0:
            print(f"  Embedded {idx + 1}/{len(all_points)} entries...")

    print("Uploading points to Qdrant...")
    points = [
        PointStruct(id=pid, vector=vec, payload=pl)
        for pid, (vec, pl) in enumerate(zip(vectors, payloads))
    ]
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    print("Done. Uploaded", len(points), "to", COLLECTION_NAME)
    

if __name__ == "__main__":
    main()

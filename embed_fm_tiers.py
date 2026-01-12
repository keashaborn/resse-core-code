#!/usr/bin/env python3
"""
embed_fm_tiers.py

Embed the three FM tier JSONL corpora into three new Qdrant collections:

- fm_canon_v1
- fm_context_v1
- external_context_v1

Each line in the JSONL is a full Q&A item from NTA with tags, axioms, metadata, dataset.

Embedding text includes:
- ROLE (based on which file it comes from)
- DATASET name
- QUESTION
- ANSWER
"""

import os
import json
from pathlib import Path
from typing import List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

ROOT = Path(__file__).parent
TIERS_DIR = ROOT / "fm_tiers"

# Files & target collections
TIERS = [
    {
        "path": TIERS_DIR / "fm_canon_v1.jsonl",
        "collection": "fm_canon_v1",
        "role": "FM_CANON",
    },
    {
        "path": TIERS_DIR / "fm_context_v1.jsonl",
        "collection": "fm_context_v1",
        "role": "FM_CONTEXT",
    },
    {
        "path": TIERS_DIR / "external_context_v1.jsonl",
        "collection": "external_context_v1",
        "role": "NON_FM_CONTEXT",
    },
]

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # usually None locally

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def recreate_collection(name: str):
    print(f"\nRecreating collection '{name}'...")
    qdrant.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE,
        ),
    )


def embed_tier(tier_conf: dict):
    path = tier_conf["path"]
    collection = tier_conf["collection"]
    role = tier_conf["role"]

    recreate_collection(collection)

    point_id = 1
    written = 0

    print(f"\nEmbedding tier: {path.name} → {collection} (role={role})")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            dataset = item.get("dataset", {})
            ds_name = dataset.get("name", "unknown_dataset")

            # Build embedding text with role + dataset
            emb_text = f"ROLE: {role}\nDATASET: {ds_name}\nQUESTION: {q}\nANSWER: {a}"

            vec = embed_text(emb_text)

            payload = {
                "id": item.get("id"),
                "role": role,
                "question": q,
                "answer": a,
                "tags": item.get("tags"),
                "axioms_primary": item.get("axioms_primary"),
                "axioms_secondary": item.get("axioms_secondary"),
                "metadata": item.get("metadata"),
                "dataset": dataset,
            }

            qdrant.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                ],
            )

            point_id += 1
            written += 1

            if written % 100 == 0:
                print(f"  {collection}: embedded {written} items...")

    print(f"\nFinished {collection}: total embedded = {written}")


def main():
    for tier in TIERS:
        embed_tier(tier)

    print("\nAll FM tier collections embedded.")


if __name__ == "__main__":
    main()
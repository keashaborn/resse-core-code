#!/usr/bin/env python3
"""
embed_nta_to_qdrant.py

Embed *.nta.jsonl files into a fresh Qdrant collection.

Usage example:
    python embed_nta_to_qdrant.py \
        --nta_dir resse_nta_clean/primary_core \
        --collection nta_primary_core_v1
"""

import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-large"
DIM = 3072

def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nta_dir", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--qdrant_api_key", default=None)
    args = parser.parse_args()

    qdrant = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key
    )

    # Recreate collection fresh
    print(f"Recreating Qdrant collection '{args.collection}'...")
    qdrant.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(
            size=DIM,
            distance=Distance.COSINE
        )
    )

    # Walk *.nta.jsonl
    nta_root = Path(args.nta_dir)
    files = sorted(nta_root.rglob("*.nta.jsonl"))

    point_id = 1
    for file in files:
        print(f"\nEmbedding file: {file.relative_to(nta_root)}")
        for line in file.open():
            obj = json.loads(line)

            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            text = f"{question}\n\n{answer}"

            vec = embed_text(text)

            payload = {
                "id": obj.get("id"),
                "question": question,
                "answer": answer,
                "tags": obj.get("tags"),
                "axioms_primary": obj.get("axioms_primary"),
                "axioms_secondary": obj.get("axioms_secondary"),
                "metadata": obj.get("metadata"),
                "dataset": obj.get("dataset"),
                "source_file": str(file),
            }

            qdrant.upsert(
                collection_name=args.collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                ]
            )

            point_id += 1

    print(f"\nDone embedding into '{args.collection}'. Total points={point_id-1}")

if __name__ == "__main__":
    main()

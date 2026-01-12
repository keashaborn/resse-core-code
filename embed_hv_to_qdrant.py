#!/usr/bin/env python3
"""
embed_hv_to_qdrant.py

Generic embedder for HV corpora into Qdrant.

Expected input format (JSONL), one object per line:

{
  "id": "hv_beh_v1_00001",
  "text": "Q: ...\\n\\nA: ...",
  "metadata": { ... }
}

Usage examples:

  python3 embed_hv_to_qdrant.py \
    --input hv_beh_phil/hv_behavioral_for_qdrant_v1.jsonl \
    --collection hv_behavioral_v1 \
    --recreate

  python3 embed_hv_to_qdrant.py \
    --input hv_beh_phil/hv_philosophical_for_qdrant_v1.jsonl \
    --collection hv_philosophical_v1 \
    --recreate

Environment:

  OPENAI_API_KEY   - your OpenAI key (already used elsewhere)
  QDRANT_URL       - e.g. http://127.0.0.1:6333 (default)
  QDRANT_API_KEY   - optional, if your Qdrant instance requires it
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_batch(texts, model="text-embedding-3-large"):
    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in resp.data]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file with id/text/metadata.")
    parser.add_argument("--collection", required=True, help="Target Qdrant collection name.")
    parser.add_argument("--recreate", action="store_true",
                        help="If set, recreate the collection before inserting.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = args.collection

    print(f"Embedding from: {input_path}")
    print(f"Target collection: {collection_name}")
    print(f"Qdrant URL: {qdrant_url}")

    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # First, load all records into memory (for simplicity).
    # If this gets too big later, we can stream instead.
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            text = obj.get("text")
            metadata = obj.get("metadata") or {}
            if not _id or not text:
                continue
            records.append((_id, text, metadata))

    # Reindex records with integer IDs for Qdrant; keep original id in metadata
    indexed_records = []
    for i, (_id, text, metadata) in enumerate(records):
        # Store original id under "doc_id" in metadata
        metadata = dict(metadata)
        metadata["doc_id"] = _id
        indexed_records.append((i, text, metadata))
    records = indexed_records

    if not records:
        print("No valid records found in input. Exiting.")
        return

    # Get vector dimension from a single embedding
    test_emb = embed_batch([records[0][1]])[0]
    dim = len(test_emb)
    print(f"Detected embedding dimension: {dim}")

    if args.recreate:
        print(f"Recreating collection '{collection_name}' with dim={dim}...")
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        print(f"Using existing collection '{collection_name}' (no recreate).")

    # Now embed all records in batches and upsert
    batch_size = args.batch_size
    total = len(records)
    print(f"Total records to embed: {total}")

    idx = 0
    while idx < total:
        batch = records[idx:idx+batch_size]
        texts = [t for (_id, t, _meta) in batch]
        ids = [_id for (_id, t, _meta) in batch]  # integer IDs
        metas = [_meta for (_id, t, _meta) in batch]

        embeddings = embed_batch(texts)

        points = []
        for pid, vec, meta in zip(ids, embeddings, metas):
            points.append(
                PointStruct(
                    id=pid,       # integer id acceptable to Qdrant
                    vector=vec,
                    payload=meta,  # contains "doc_id" for your original string id
                )
            )

        qdrant.upsert(
            collection_name=collection_name,
            points=points,
        )

        idx += batch_size
        print(f"Upserted {min(idx, total)}/{total} records...")

    print("Done embedding and upserting to Qdrant.")

if __name__ == "__main__":
    main()

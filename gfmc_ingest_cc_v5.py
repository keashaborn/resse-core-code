#!/usr/bin/env python3
"""
gfmc_ingest_cc_v5.py

Embed and upload the merged Fractal Monism canon+context corpus v5
into a new Qdrant collection called 'gfmc_cc_v5'.

Source file:
  fm_principles/results/fm_qna_merged_v5_deduped.jsonl
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from openai import OpenAI

# ---- CONFIG ----

QDRANT_URL = "http://127.0.0.1:6333"
EMBED_MODEL = "text-embedding-3-large"

BASE_DIR = Path("/home/ubuntu/resse-core")
QNA_PATH = BASE_DIR / "fm_principles" / "results" / "fm_qna_merged_v5_deduped.jsonl"

QNA_COLLECTION = "gfmc_cc_v5"

BATCH_SIZE = 128

client = OpenAI()  # uses OPENAI_API_KEY from environment


# ---- QDRANT HELPERS ----

def create_collection_if_not_exists(name: str, vector_size: int = 3072, distance: str = "Cosine") -> None:
    print(f"Ensuring collection '{name}' exists in Qdrant...")
    r = requests.get(f"{QDRANT_URL}/collections/{name}")
    if r.status_code == 200 and r.json().get("status") == "ok":
        print(f"  Collection '{name}' already exists.")
        return

    body = {
        "vectors": {
            "size": vector_size,
            "distance": distance,
        }
    }
    r = requests.put(f"{QDRANT_URL}/collections/{name}", json=body)
    r.raise_for_status()
    print(f"  Created collection '{name}'.")


def upsert_points(
    collection: str,
    ids: List[Any],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
) -> None:
    assert len(ids) == len(vectors) == len(payloads)
    points = []
    for pid, vec, payload in zip(ids, vectors, payloads):
        points.append(
            {
                "id": pid,
                "vector": vec,
                "payload": payload,
            }
        )

    body = {"points": points}
    r = requests.put(f"{QDRANT_URL}/collections/{collection}/points?wait=true", json=body)

    if r.status_code != 200:
        print(f"\nError upserting into collection '{collection}':")
        print("Status:", r.status_code)
        try:
            print("Body:", r.json())
        except Exception:
            print("Body (raw):", r.text[:500])
        raise SystemExit(1)


# ---- EMBEDDING + INGEST ----

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts with OpenAI, using text-embedding-3-large."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]


def main() -> None:
    if not QNA_PATH.exists():
        raise FileNotFoundError(f"QNA file not found: {QNA_PATH}")

    print(f"Loading Q&A from: {QNA_PATH}")
    create_collection_if_not_exists(QNA_COLLECTION)

    ids: List[int] = []
    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []

    next_id = 1
    batch_texts: List[str] = []
    batch_payloads: List[Dict[str, Any]] = []

    with QNA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a:
                continue

            text = f"Q: {q}\n\nA: {a}"
            batch_texts.append(text)

            # keep the full original object as payload
            payloads_obj = {
                "question": q,
                "answer": a,
                "source": "gfmc_cc_v5",
            }
            batch_payloads.append(payloads_obj)

            if len(batch_texts) >= BATCH_SIZE:
                print(f"Embedding and uploading batch of {len(batch_texts)}...")
                embs = embed_texts(batch_texts)
                start_id = next_id
                ids_batch = list(range(start_id, start_id + len(batch_texts)))
                upsert_points(QNA_COLLECTION, ids_batch, embs, batch_payloads)
                next_id += len(batch_texts)
                batch_texts = []
                batch_payloads = []

    # Final partial batch
    if batch_texts:
        print(f"Embedding and uploading final batch of {len(batch_texts)}...")
        embs = embed_texts(batch_texts)
        start_id = next_id
        ids_batch = list(range(start_id, start_id + len(batch_texts)))
        upsert_points(QNA_COLLECTION, ids_batch, embs, batch_payloads)

    print("Done uploading to Qdrant collection:", QNA_COLLECTION)


if __name__ == "__main__":
    main()

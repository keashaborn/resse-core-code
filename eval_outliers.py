#!/usr/bin/env python3
"""
eval_outliers.py

Find bottom-K outliers in each NTA collection by axiom alignment.

For each collection in Qdrant:
  - load all points (with vectors + payload)
  - compute best cosine similarity to any axiom in core_axioms_v2
  - keep the lowest-K scoring items
Write result to results/nta_outliers.tsv
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

# Qdrant setup
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # usually None locally

AXIOM_COLLECTION = "core_axioms_v2"

# NTA collections
NTA_COLLECTIONS = [
    "nta_primary_core_v1",
    "nta_secondary_core_ai_v1",
    "nta_secondary_core_psych_v1",
    "nta_secondary_core_metaphy_v1",
    "nta_secondary_core_science_v1",
    "nta_secondary_core_applied_v1",
    "nta_ref_psy_v1",
    "nta_ref_phi_v1",
    "nta_ref_misc_v1",
]

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "nta_outliers.tsv"

K = 20  # how many outliers per collection to keep


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_axiom_vectors(client: QdrantClient):
    """Load all axiom vectors from core_axioms_v2."""
    axioms = []
    points, _ = client.scroll(
        collection_name=AXIOM_COLLECTION,
        limit=100,
        with_vectors=True,
        with_payload=True,
    )
    for p in points:
        aid = p.payload.get("axiom_id")
        if aid is None:
            continue
        axioms.append((aid, np.array(p.vector, dtype=np.float32)))
    print(f"Loaded {len(axioms)} axiom vectors from {AXIOM_COLLECTION}")
    return axioms


def find_outliers_for_collection(client: QdrantClient, collection: str, axioms):
    """Return bottom-K items for a single collection as dicts."""
    print(f"\n=== Scanning collection: {collection} ===")

    points, _ = client.scroll(
        collection_name=collection,
        limit=50000,  # all in one go; total corpus < 50k points
        with_vectors=True,
        with_payload=True,
    )

    scored = []
    for p in points:
        v = np.array(p.vector, dtype=np.float32)

        best = 0.0
        for _ax_id, ax_v in axioms:
            sim = cosine(v, ax_v)
            if sim > best:
                best = sim

        payload = p.payload or {}
        scored.append(
            {
                "collection": collection,
                "id": payload.get("id"),
                "score": best,
                "question": (payload.get("question") or "").strip(),
                "answer": (payload.get("answer") or "").strip(),
            }
        )

    scored.sort(key=lambda x: x["score"])
    return scored[:K]


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    axioms = load_axiom_vectors(client)
    all_outliers = []

    for col in NTA_COLLECTIONS:
        outliers = find_outliers_for_collection(client, col, axioms)
        all_outliers.extend(outliers)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("collection\tid\tscore\tquestion\tanswer\n")
        for row in all_outliers:
            # compress whitespace a bit
            q = row["question"].replace("\n", " ")
            a = row["answer"].replace("\n", " ")
            f.write(
                f"{row['collection']}\t{row['id']}\t{row['score']:.4f}\t{q}\t{a}\n"
            )

    print(f"\nDone. Outliers written to {OUT_PATH}")


if __name__ == "__main__":
    main()
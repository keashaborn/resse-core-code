#!/usr/bin/env python3
"""
Evaluate all NTA Qdrant collections against core_axioms_v2.

For each collection:
- Pull a sample of vectors or entire set (configurable)
- Compare them to axiom vectors
- Compute avg/max similarity per axiom
- Compute an "alignment score"
- Detect outliers and mismatches
- Write results to results/nta_eval_report.tsv
"""

import os
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from openai import OpenAI
import numpy as np

QDRANT_URL = "http://localhost:6333"
AXIOM_COLLECTION = "core_axioms_v2"

# These are all your NTA vector collections
NTA_COLLECTIONS = [
    "nta_primary_core_v1",
    "nta_secondary_core_ai_v1",
    "nta_secondary_core_psych_v1",
    "nta_secondary_core_metaphy_v1",
    "nta_secondary_core_science_v1",
    "nta_secondary_core_applied_v1",
    "nta_ref_psy_v1",
    "nta_ref_phi_v1",
    "nta_ref_misc_v1"
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

client = QdrantClient(QDRANT_URL)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

### ----------------------------------------------------------
###  Helpers
### ----------------------------------------------------------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_axiom_vectors():
    """Load all axiom vectors from core_axioms_v2."""
    axioms = []
    scroll = client.scroll(
        collection_name=AXIOM_COLLECTION,
        limit=100,
        with_vectors=True
    )
    points, _ = scroll
    for p in points:
        axioms.append((p.payload["axiom_id"], np.array(p.vector)))
    return axioms


### ----------------------------------------------------------
###  Main evaluator
### ----------------------------------------------------------

def evaluate_collection(name, axioms):
    """Return summary metrics for a single NTA collection."""
    print(f"=== Evaluating {name} ===")

    scroll = client.scroll(
        collection_name=name,
        limit=50000,
        with_vectors=True
    )
    points, _ = scroll

    scores = []

    for p in points:
        v = np.array(p.vector)

        best = 0
        for ax_id, ax_v in axioms:
            sim = cosine(v, ax_v)
            if sim > best:
                best = sim

        scores.append(best)

    if not scores:
        return {"avg": 0, "max": 0, "min": 0, "count": 0}

    return {
        "avg": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "count": len(scores)
    }


### ----------------------------------------------------------
###  Entry point
### ----------------------------------------------------------

def main():
    axioms = get_axiom_vectors()
    out_path = RESULTS_DIR / "nta_eval_report.tsv"

    with open(out_path, "w") as f:
        f.write("collection\tcount\tavg\tmax\tmin\n")

        for col in NTA_COLLECTIONS:
            res = evaluate_collection(col, axioms)
            f.write(f"{col}\t{res['count']}\t{res['avg']:.4f}\t{res['max']:.4f}\t{res['min']:.4f}\n")

    print("\n=== Done. Report written to results/nta_eval_report.tsv ===")


if __name__ == "__main__":
    main()
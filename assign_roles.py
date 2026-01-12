#!/usr/bin/env python3
"""
assign_roles.py

Assigns each NTA point to one of:
- FM_CANON
- FM_CONTEXT
- NON_FM_CONTEXT

based on its best cosine similarity to any axiom vector in core_axioms_v2.

Outputs: results/nta_roles.tsv
"""

import os
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

AXIOM_COLLECTION = "core_axioms_v2"

# All NTA vector collections
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
OUT_PATH = RESULTS_DIR / "nta_roles.tsv"

# Thresholds (you can tune these later)
# Roughly: ≥0.43 = FM_CANON, 0.32–0.43 = FM_CONTEXT, <0.32 = NON_FM_CONTEXT
CANON_THR = 0.43
CONTEXT_THR = 0.32


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


def assign_role(score: float) -> str:
    """Return FM_CANON, FM_CONTEXT, or NON_FM_CONTEXT based on score."""
    if score >= CANON_THR:
        return "FM_CANON"
    elif score >= CONTEXT_THR:
        return "FM_CONTEXT"
    else:
        return "NON_FM_CONTEXT"


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    axioms = load_axiom_vectors(client)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("collection\tid\tscore\trole\tquestion\n")

        for col in NTA_COLLECTIONS:
            print(f"\n=== Scanning {col} ===")
            points, _ = client.scroll(
                collection_name=col,
                limit=50000,
                with_vectors=True,
                with_payload=True,
            )

            for p in points:
                v = np.array(p.vector, dtype=np.float32)
                payload = p.payload or {}
                q = (payload.get("question") or "").strip()

                # compute best similarity to axioms
                best = 0.0
                for _ax_id, ax_v in axioms:
                    sim = cosine(v, ax_v)
                    if sim > best:
                        best = sim

                role = assign_role(best)
                item_id = payload.get("id")

                # write one line
                f.write(
                    f"{col}\t{item_id}\t{best:.4f}\t{role}\t{q.replace(chr(9), ' ')}\n"
                )

    print(f"\nDone. Roles written to {OUT_PATH}")


if __name__ == "__main__":
    main()
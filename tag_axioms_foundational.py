#!/usr/bin/env python3
"""
tag_axioms_foundational.py

For each QA item in the foundational datasets:
- Build an embedding from question + answer text
- Query the core axioms Qdrant collection
- Fill axioms_primary and axioms_secondary based on top matches
- Writes *.axiomed.json next to the input file (does not overwrite existing files)
"""

import json
from copy import deepcopy
from pathlib import Path
import os

from openai import OpenAI
from qdrant_client import QdrantClient

# ---- CONFIGURE THESE TO MATCH YOUR ENVIRONMENT ----

DATASETS_DIR = Path(__file__).parent / "datasets"

# Use your actual Qdrant URL / API key (or host/port if local)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # or None if no auth

AXIOMS_COLLECTION = "core_axioms_v2"  # adjust if your collection has a different name
EMBED_MODEL = "text-embedding-3-large"  # or whatever you're using for axioms

# Foundational datasets to tag (use enriched if available, otherwise normalized)
FOUNDATIONAL_TARGETS = [
    "foundational/foundational_core.normalized.json",
    "foundational/foundational_clarifications.normalized.json",
    "foundational/foundational_clarifications_extended.normalized.enriched.json",
    "foundational/foundational_comprehension.normalized.json",
    "foundational/foundational_expansion.normalized.enriched.json",
    "foundational/foundational_definitions.normalized.json",
]

# How many nearest axioms to retrieve per item
TOP_K_AXIOMS = 6  # we'll split into primary (top 2) and secondary (next up to 4)


# ---- INITIALIZE CLIENTS ----

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---- HELPERS ----

def embed_text(text: str) -> list[float]:
    """Return embedding vector for given text using OpenAI embeddings."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def choose_axioms_for_item(text: str) -> tuple[list[str], list[str]]:
    """
    Query Qdrant for closest axioms to this text.
    Returns (primary_axioms, secondary_axioms) as lists of axiom_id strings.
    """
    vector = embed_text(text)
    results = qdrant.search(
        collection_name=AXIOMS_COLLECTION,
        query_vector=vector,
        limit=TOP_K_AXIOMS,
        with_vectors=False,
        with_payload=True,
    )

    # Expect each result.payload to contain something like {"axiom_id": "A1", "id": "ax-A1", ...}
    axiom_ids: list[str] = []
    for res in results:
        payload = res.payload or {}
        ax_id = payload.get("axiom_id") or payload.get("id")
        if isinstance(ax_id, str):
            axiom_ids.append(ax_id)

    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for aid in axiom_ids:
        if aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)

    # Split into primary / secondary
    primary = unique_ids[:2]
    secondary = unique_ids[2:6]

    return primary, secondary


def axioms_are_empty(item: dict) -> bool:
    """Return True if both axioms_primary and axioms_secondary are empty or missing."""
    ap = item.get("axioms_primary") or []
    asec = item.get("axioms_secondary") or []
    return not ap and not asec


def process_dataset(path: Path) -> None:
    print(f"\n=== Tagging axioms for: {path} ===")

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return

    dataset = data.get("dataset", {})
    items = data.get("items", [])

    # We'll write into a separate file to avoid overwriting normalized/enriched
    out_items = []
    updated = 0

    for item in items:
        new_item = deepcopy(item)

        # Only tag QA items
        if new_item.get("type") != "qa":
            out_items.append(new_item)
            continue

        if not axioms_are_empty(new_item):
            # Already tagged, skip for now
            out_items.append(new_item)
            continue

        q = (new_item.get("question") or "").strip()
        a = (new_item.get("answer") or "").strip()
        if not q and not a:
            out_items.append(new_item)
            continue

        text = f"{q}\n\n{a}"

        try:
            primary, secondary = choose_axioms_for_item(text)
            new_item["axioms_primary"] = primary
            new_item["axioms_secondary"] = secondary
            updated += 1
            print(f"  Tagged id={new_item.get('id')} → primary={primary}, secondary={secondary}")
        except Exception as e:
            print(f"  ERROR tagging id={new_item.get('id')}: {e}")

        out_items.append(new_item)

    out_data = {
        "dataset": dataset,
        "items": out_items,
    }

    out_path = path.with_suffix(".axiomed.json")
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"  Done. Updated {updated} items. Wrote: {out_path.name}")


def main():
    for rel in FOUNDATIONAL_TARGETS:
        base_path = DATASETS_DIR / rel
        if not base_path.exists():
            print(f"Skipping missing file: {base_path}")
            continue
        process_dataset(base_path)


if __name__ == "__main__":
    main()

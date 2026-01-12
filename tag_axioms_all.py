#!/usr/bin/env python3
"""
tag_axioms_all.py

For every dataset JSON under ./datasets:
  - For each item of type "qa" with empty axioms_primary/axioms_secondary:
    - Build an embedding from question + answer
    - Query Qdrant core_axioms_v2
    - Fill axioms_primary and axioms_secondary

Writes <original>.axiomed.json next to each input file.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

# ---- CONFIG ----

DATASETS_DIR = Path(__file__).parent / "datasets"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # or None if no auth
AXIOMS_COLLECTION = "core_axioms_v2"
EMBED_MODEL = "text-embedding-3-large"

TOP_K_AXIOMS = 6  # 2 primary + 4 secondary

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---- Helpers ----

def embed_text(text: str) -> List[float]:
    """Return embedding vector for a piece of text."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def choose_axioms_for_item(text: str) -> Tuple[List[str], List[str]]:
    """
    Query Qdrant for closest axioms to this text.
    Returns (primary_axioms, secondary_axioms) as lists of axiom_id strings.
    """
    vector = embed_text(text)

    # Using deprecated `search` is okay functionally; can be updated to `query_points` later.
    results: List[ScoredPoint] = qdrant.search(
        collection_name=AXIOMS_COLLECTION,
        query_vector=vector,
        limit=TOP_K_AXIOMS,
        with_vectors=False,
        with_payload=True,
    )

    axiom_ids: List[str] = []
    for res in results:
        payload = res.payload or {}
        ax_id = payload.get("axiom_id") or payload.get("id")
        if isinstance(ax_id, str):
            axiom_ids.append(ax_id)

    # Deduplicate while preserving order
    seen = set()
    unique_ids: List[str] = []
    for aid in axiom_ids:
        if aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)

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

    updated = 0
    out_items = []

    for item in items:
        new_item = deepcopy(item)

        if new_item.get("type") != "qa":
            out_items.append(new_item)
            continue

        if not axioms_are_empty(new_item):
            out_items.append(new_item)
            continue

        q = (new_item.get("question") or "").strip()
        a = (new_item.get("answer") or "").strip()
        if not q and not a:
            out_items.append(new_item)
            continue

        text = f"{q}\n\n{text_clean(a)}"

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

    out_path = path.with_suffix(path.suffix + ".axiomed.json")
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"  Done. Updated {updated} items. Wrote: {out_path.name}")


def text_clean(s: str) -> str:
    return s.strip()


def should_process_file(path: Path) -> bool:
    """Decide whether this file should be processed."""
    name = path.name

    # Only process normalized or normalized.enriched JSONs
    if not (name.endswith(".normalized.json") or name.endswith(".normalized.enriched.json")):
        return False

    # Skip already axiomed files (to avoid infinite suffixes)
    if name.endswith(".axiomed.json"):
        return False

    # Skip core axiom/paraphrase files or any other special files if needed
    if "core_axioms" in name:
        return false

    return True


def main() -> None:
    # Walk entire datasets tree
    for path in sorted(DATASETS_DIR.rglob("*.json")):
        if not should_process_file(path):
            continue
        process_dataset(path)


if __name__ == "__main__":
    main()

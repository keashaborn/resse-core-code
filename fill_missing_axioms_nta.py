#!/usr/bin/env python3
"""
fill_missing_axioms_nta.py

Walks all *.nta.json under resse_nta_clean/,
and for any item with empty axioms_primary & axioms_secondary:

- Builds an embedding from question + answer
- Queries Qdrant core_axioms_v2
- Fills axioms_primary (top 2) and axioms_secondary (next up to 4)

Requires:
- OPENAI_API_KEY in env
- Qdrant running at QDRANT_URL (default http://localhost:6333)
- collection core_axioms_v2 populated with axioms + paraphrases
"""

import os
import json
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

ROOT = Path(__file__).parent
NTA_ROOT = ROOT / "resse_nta_clean"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # or None
AXIOMS_COLLECTION = "core_axioms_v2"
EMBED_MODEL = "text-embedding-3-large"
TOP_K_AXIOMS = 6  # 2 primary + 4 secondary

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def choose_axioms_for_item(text: str) -> Tuple[List[str], List[str]]:
    vec = embed_text(text)

    results: List[ScoredPoint] = qdrant.search(
        collection_name=AXIOMS_COLLECTION,
        query_vector=vec,
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

    # Deduplicate, preserve order
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
    ap = item.get("axioms_primary") or []
    as2 = item.get("axioms_secondary") or []
    return not ap and not as2


def process_file(path: Path) -> int:
    data = json.loads(path.read_text())
    items = data.get("items", [])
    updated = 0

    new_items = []
    for item in items:
        itm = deepcopy(item)

        # Only QA items
        if itm.get("type") != "qa":
            new_items.append(itm)
            continue

        if not axioms_are_empty(itm):
            new_items.append(itm)
            continue

        q = (itm.get("question") or "").strip()
        a = (itm.get("answer") or "").strip()
        if not q and not a:
            new_items.append(itm)
            continue

        text = f"{q}\n\n{a}"
        try:
            primary, secondary = choose_axioms_for_item(text)
            itm["axioms_primary"] = primary
            itm["axioms_secondary"] = secondary
            updated += 1
            print(f"  {path.relative_to(NTA_ROOT)}  id={itm.get('id')}  →  primary={primary}, secondary={secondary}")
        except Exception as e:
            print(f"  ERROR tagging {path.relative_to(NTA_ROOT)} id={itm.get('id')}: {e}")

        new_items.append(itm)

    if updated:
        data["items"] = new_items
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return updated


def main():
    total_updated = 0
    for path in sorted(NTA_ROOT.rglob("*.nta.json")):
        print(f"\nProcessing: {path.relative_to(NTA_ROOT)}")
        updated = process_file(path)
        print(f"  Updated {updated} items in this file.")
        total_updated += updated

    print(f"\nDone. Filled axioms for {total_updated} items across all NTA files.")


if __name__ == "__main__":
    main()


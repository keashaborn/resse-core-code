#!/usr/bin/env python3
"""
check_qdrant_payload_text.py

Scan all Qdrant collections (except memory_raw) and report, for each:
  - total number of points
  - how many points do NOT have payload["text"]

This lets us see which collections follow the "payload.text" convention that
the chat/RAG layer expects, and which ones (like new HV/FM corpora) may not yet.
"""

import os
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
IGNORED_COLLECTIONS = {"memory_raw"}

def main():
    client = QdrantClient(url=QDRANT_URL)

    cols = client.get_collections()
    collections = getattr(cols, "collections", []) or []

    if not collections:
        print("No collections found.")
        return

    for c in collections:
        name = getattr(c, "name", None) or getattr(c, "name", None)
        if not name:
            continue
        if name in IGNORED_COLLECTIONS:
            continue

        print(f"\n=== Collection: {name} ===")

        total = 0
        missing_text = 0

        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            for p in points:
                total += 1
                payload = p.payload or {}
                text_val = payload.get("text")
                if not isinstance(text_val, str) or not text_val.strip():
                    missing_text += 1

            if next_offset is None:
                break
            offset = next_offset

        print(f"  total points         : {total}")
        print(f"  missing payload.text : {missing_text}")
        if total > 0:
            pct_missing = (missing_text / total) * 100.0
            print(f"  % missing            : {pct_missing:.2f}%")

if __name__ == "__main__":
    main()

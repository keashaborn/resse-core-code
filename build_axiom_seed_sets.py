#!/usr/bin/env python3
"""
build_axiom_seed_sets.py

From fm_tiers/fm_canon_v1.jsonl, build per-axiom seed sets:

fm_axiom_seeds/A1_seeds.jsonl
fm_axiom_seeds/A2_seeds.jsonl
...
fm_axiom_seeds/A15_seeds.jsonl

Each line in each file is a full canon Q&A that references that axiom
in axioms_primary or axioms_secondary.
"""

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
CANON_PATH = ROOT / "fm_tiers" / "fm_canon_v1.jsonl"
OUT_DIR = ROOT / "fm_axiom_seeds"

OUT_DIR.mkdir(exist_ok=True)

# Initialize buckets: A1..A15
axiom_buckets = defaultdict(list)

print("Loading fm_canon_v1.jsonl...")

count = 0
with CANON_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        count += 1

        primary = item.get("axioms_primary") or []
        secondary = item.get("axioms_secondary") or []

        for aid in primary + secondary:
            # only keep A1..A15, ignore weird IDs if any
            if aid.startswith("A") and len(aid) <= 3:
                axiom_buckets[aid].append(item)

print(f"Loaded {count} canon items.")
print("Building per-axiom seed files...")

for aid, items in sorted(axiom_buckets.items()):
    out_path = OUT_DIR / f"{aid}_seeds.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for itm in items:
            f.write(json.dumps(itm, ensure_ascii=False) + "\n")
    print(f"  {aid}: wrote {len(items)} items to {out_path}")

print("\nDone building axiom seed sets.")
print("Output dir:", OUT_DIR)
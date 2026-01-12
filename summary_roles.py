#!/usr/bin/env python3
"""
summarize_roles.py

Summarize how many items are FM_CANON / FM_CONTEXT / NON_FM_CONTEXT
per collection, based on results/nta_roles.tsv.
"""

from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).parent
ROLES_PATH = ROOT / "results" / "nta_roles.tsv"

def main():
    counts = defaultdict(lambda: {"FM_CANON": 0, "FM_CONTEXT": 0, "NON_FM_CONTEXT": 0, "TOTAL": 0})

    with ROLES_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            col = row["collection"]
            role = row["role"]
            counts[col]["TOTAL"] += 1
            if role in counts[col]:
                counts[col][role] += 1

    print("collection\tTOTAL\tFM_CANON\tFM_CONTEXT\tNON_FM_CONTEXT")
    for col, data in counts.items():
        print(
            f"{col}\t{data['TOTAL']}\t{data['FM_CANON']}\t{data['FM_CONTEXT']}\t{data['NON_FM_CONTEXT']}"
        )

if __name__ == "__main__":
    main()
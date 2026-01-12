#!/usr/bin/env python3
"""
rename_nta_corpus.py

Creates a clean copy of the NTA corpus under ./resse_nta_clean/
with simplified, stable filenames.

Rules:
- Remove bracket prefixes like [core], [applied], [pfi]
- Remove `_batch_XX`
- Remove double suffix chains like `.normalized.json`
- Normalize punctuation: replace spaces, unicode dashes, etc.
- Keep directory structure (foundational, core, applied, ref_psy ...)
- Output ends in `.nta.json`
"""

import re
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).parent
SRC = ROOT / "resse_nta"
DST = ROOT / "resse_nta_clean"

def clean_name(name: str) -> str:
    # Remove bracketed prefixes like [core], [applied]
    name = re.sub(r"\[[^]]*\]_", "", name)

    # Remove batch indicators
    name = re.sub(r"_batch_\d+", "", name)

    # Remove double .normalized.json or .normalized.enriched.json
    name = name.replace(".normalized.json", "")
    name = name.replace(".normalized.enriched.json", "")

    # Replace unicode punctuation with ASCII-friendly
    name = name.replace("–", "-")
    name = name.replace("—", "-")
    name = name.replace("’", "'")
    name = name.replace("“", "")
    name = name.replace("”", "")

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove parentheses
    name = name.replace("(", "").replace(")", "")

    # Collapse repeated underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing underscores or hyphens
    name = name.strip("_-")

    return name

def main():
    print("Building clean NTA corpus in resse_nta_clean/")

    for src in SRC.rglob("*.nta.json"):
        rel = src.relative_to(SRC)

        # clean the filename ONLY, not directory
        base = rel.name
        base = base.replace(".nta.json", "")
        cleaned = clean_name(base) + ".nta.json"

        dst = DST / rel.parent / cleaned
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)

        print(f"Copied {src} -> {dst}")

    print("\nDONE. Clean NTA corpus created in resse_nta_clean/")

if __name__ == "__main__":
    main()


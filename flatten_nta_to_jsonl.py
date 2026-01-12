#!/usr/bin/env python3
"""
flatten_nta_to_jsonl.py

Flatten all *.nta.json files under ./resse_nta_clean
into linewise *.nta.jsonl files (one Q&A per line),
preserving question, answer, tags, axioms, metadata.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
NTA_ROOT = ROOT / "resse_nta_clean"

def flatten_file(src: Path):
    dst = src.with_suffix(src.suffix + "l")  # .nta.json -> .nta.jsonl
    data = json.loads(src.read_text())
    items = data.get("items", [])
    dataset_meta = data.get("dataset", {})

    with dst.open("w", encoding="utf-8") as f:
        for itm in items:
            if itm.get("type") != "qa":
                continue
            row = {
                "id": itm.get("id"),
                "question": itm.get("question"),
                "answer": itm.get("answer"),
                "tags": itm.get("tags"),
                "axioms_primary": itm.get("axioms_primary"),
                "axioms_secondary": itm.get("axioms_secondary"),
                "metadata": itm.get("metadata"),
                "dataset": dataset_meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Flattened {src.relative_to(NTA_ROOT)} -> {dst.relative_to(NTA_ROOT)}")

def main():
    for path in sorted(NTA_ROOT.rglob("*.nta.json")):
        flatten_file(path)

if __name__ == "__main__":
    main()


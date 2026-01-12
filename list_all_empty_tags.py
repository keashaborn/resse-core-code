#!/usr/bin/env python3
"""
list_all_empty_tags.py

Scan the entire datasets tree and report any items whose tags are completely empty
(after normalization/enrichment).

- Prefers *.enriched.json if present.
- Otherwise uses *.normalized.json.
- Prints only files that still have items with fully empty tags.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "datasets"

# We already know foundational & kernel are special, but we still want to scan them.
# You can adjust SKIP_DIRS if there are folders you truly want to ignore.
SKIP_DIRS = set()  # e.g. {"some_folder_to_skip"}

def tags_are_empty(tags: dict) -> bool:
    """Return True if all tag fields (incl. intent subfields) are empty."""
    if not tags:
        return True
    # any non-empty array means it's not "empty"
    for key in ["authority", "category", "tone", "purpose", "frames", "emotion"]:
        if tags.get(key):
            return False
    intent = tags.get("intent") or {}
    if intent.get("question") or intent.get("answer"):
        return False
    return True

def scan_dataset_file(path: Path):
    """Return list of item IDs whose tags are completely empty."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return []

    items = data.get("items", [])
    empty_ids = []
    for item in items:
        tags = item.get("tags") or {}
        if tags_are_empty(tags):
            empty_ids.append(item.get("id", "<no-id>"))
    return empty_ids

def main():
    # Build a map: base dataset file -> chosen scan file (enriched or normalized)
    files_to_scan = []

    for json_path in DATA_DIR.rglob("*.normalized.json"):
        # Skip normalized-normalized, normalized-normalized-normalized, etc.
        if ".normalized.normalized" in json_path.name:
            continue

        # Respect skip dirs if needed
        parts = json_path.relative_to(DATA_DIR).parts
        if parts and parts[0] in SKIP_DIRS:
            continue

        # If there's an enriched version, prefer that instead
        enriched_path = json_path.with_suffix(".enriched.json")
        if enriched_path.exists():
            files_to_scan.append(enriched_path)
        else:
            files_to_scan.append(json_path)

    # Deduplicate in case of overlap
    files_to_scan = sorted(set(files_to_scan))

    print(f"Scanning {len(files_to_scan)} dataset files for empty tags...\n")

    total_empty = 0
    for path in files_to_scan:
        empty_ids = scan_dataset_file(path)
        if empty_ids:
            rel = path.relative_to(DATA_DIR)
            print(f"=== {rel} ===")
            print(f"  Items with completely empty tags ({len(empty_ids)}):")
            for eid in empty_ids:
                print(f"    - {eid}")
            print()
            total_empty += len(empty_ids)

    if total_empty == 0:
        print("No items with completely empty tags found anywhere.")
    else:
        print(f"\nTotal items with completely empty tags: {total_empty}")

if __name__ == "__main__":
    main()

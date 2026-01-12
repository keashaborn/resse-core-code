#!/usr/bin/env python3
"""
list_foundational_empty_tags.py

Helper: scan normalized foundational datasets and list items that have completely empty tags.
This tells us which items need GPT tag enrichment later.
"""

import json
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
FOUNDATIONAL_DIR = DATASETS_DIR / "foundational"


def tags_are_empty(tags: dict) -> bool:
    """Return True if all tag fields (incl. intent subfields) are empty."""
    if not tags:
        return True

    # If any of these have values, tags are not "empty"
    for key in ["authority", "category", "tone", "purpose", "frames", "emotion"]:
        if tags.get(key):
            return False

    intent = tags.get("intent") or {}
    if intent.get("question") or intent.get("answer"):
        return False

    return True


def scan_file(path: Path) -> None:
    data = json.loads(path.read_text())
    dataset = data.get("dataset", {})
    items = data.get("items", [])

    print(f"\n=== {path.name} ===")
    print(f"Dataset: {dataset.get('name', '<no name>')}")
    empty_ids = []

    for item in items:
        tags = item.get("tags") or {}
        if tags_are_empty(tags):
            empty_ids.append(item.get("id", "<no id>"))

    if not empty_ids:
        print("  No items with completely empty tags.")
    else:
        print(f"  Items with empty tags ({len(empty_ids)}):")
        for i in empty_ids:
            print(f"    - {i}")


def main() -> None:
    # Look at all normalized foundational files
    for path in sorted(FOUNDATIONAL_DIR.glob("*.normalized.json")):
        scan_file(path)


if __name__ == "__main__":
    main()

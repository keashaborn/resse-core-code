#!/usr/bin/env python3
"""
normalize_other_qa.py

Normalize all non-foundational, non-kernel datasets into canonical QA format.

This script:
- Walks ~/resse-core/datasets/* except foundational/ and kernel/
- Treats each dataset item as QA
- Verifies question+answer presence
- Normalizes tags, notes, metadata, axioms
- Flags "dataset description duplicated as first Q&A" cases
"""

import json
from pathlib import Path
from copy import deepcopy

DATASETS_DIR = Path(__file__).parent / "datasets"

SKIP_TOP = {"foundational", "kernel", "resse_json"}
SKIP_FILES = {"manifest_core.json", "manifest_applied.json", "manifest.json"}

#############################
# Shared helpers (copied/adapted from kernel/foundational normalizers)
#############################

def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def normalize_tags(raw_tags: dict | None) -> dict:
    raw_tags = raw_tags or {}

    authority = ensure_list(raw_tags.get("authority"))
    category = ensure_list(raw_tags.get("category"))
    tone = ensure_list(raw_tags.get("tone"))
    purpose = ensure_list(raw_tags.get("purpose"))
    frames = ensure_list(raw_tags.get("frames"))
    emotion = ensure_list(raw_tags.get("emotion"))

    raw_intent = raw_tags.get("intent") or {}
    if isinstance(raw_intent, dict):
        intent_q = ensure_list(raw_intent.get("question"))
        intent_a = ensure_list(raw_intent.get("answer"))
    else:
        intent_q = []
        intent_a = ensure_list(raw_intent)

    return {
        "authority": authority,
        "category": category,
        "tone": tone,
        "purpose": purpose,
        "intent": {
            "question": intent_q,
            "answer": intent_a,
        },
        "frames": frames,
        "emotion": emotion,
    }

def normalize_metadata(raw_meta: dict | None, source_file: str, pair_index: int,
                       item_type: str, scale: str, collection: str) -> dict:
    raw_meta = deepcopy(raw_meta) if raw_meta else {}
    raw_meta.setdefault("source_file", source_file)
    raw_meta.setdefault("pair_index", pair_index)
    raw_meta.setdefault("scale", scale)
    raw_meta.setdefault("form", item_type)
    raw_meta.setdefault("collection", collection)
    raw_meta.setdefault("version", "1.0")
    return raw_meta

def looks_like_qa(item: dict) -> bool:
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    return bool(q and a)

def looks_like_header_duplicate(desc: str, ans: str) -> bool:
    if not desc or not ans:
        return False
    d = desc.lower().replace("\n", " ").strip()
    a = ans.lower().replace("\n", " ").strip()
    if len(d) < 80 or len(a) < 80:
        return False
    # crude but effective for your dataset
    return d[:120] in a or a[:120] in d


#############################
# QA normalizer
#############################

def normalize_other_qa_item(item: dict,
                            rel_collection: str,
                            source_file: str,
                            idx: int,
                            scale: str) -> dict:
    """
    Convert arbitrary QA into canonical QA schema.
    """
    out = {}
    out["id"] = item.get("id", f"{rel_collection.replace('/', '-')}-qa-{idx:04d}")
    out["type"] = "qa"

    out["question"] = (item.get("question") or "").strip()
    out["answer"] = (item.get("answer") or "").strip()

    # tags + notes
    tags = normalize_tags(item.get("tags"))
    notes = ensure_list(item.get("notes"))

    # metadata
    raw_meta = item.get("metadata") or {}
    pair_index = raw_meta.get("pair_index", idx)
    metadata = normalize_metadata(raw_meta, source_file, pair_index,
                                  item_type="qa", scale=scale,
                                  collection=rel_collection)
    # Override any legacy form value: this script is QA-only
    metadata["form"] = "qa"

    # axioms
    axioms_primary = ensure_list(item.get("axioms_primary"))
    axioms_secondary = ensure_list(item.get("axioms_secondary"))

    out["tags"] = tags
    out["notes"] = notes
    out["metadata"] = metadata
    out["axioms_primary"] = axioms_primary
    out["axioms_secondary"] = axioms_secondary

    return out


#############################
# Dataset processor
#############################

def process_dataset_file(path: Path, scale: str, rel_collection: str) -> None:
    print(f"\n=== Normalizing: {path} ===")
    text = path.read_text()
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"  ERROR parsing file: {e}")
        return

    if "items" not in data:
        print("  Skipping: no 'items' field (not a dataset).")
        return

    dataset = data.get("dataset", {})
    items = data.get("items", [])

    normalized_items = []

    for idx, item in enumerate(items):
        if looks_like_qa(item):
            normalized = normalize_other_qa_item(item, rel_collection,
                                                 path.name, idx, scale)
        else:
            print(f"  WARNING: Non-QA item id={item.get('id')} in {path.name}")
            # Keep raw but mark
            normalized = deepcopy(item)
            md = normalized.setdefault("metadata", {})
            md["form"] = "unknown"
            normalized["type"] = "unknown"

        normalized_items.append(normalized)

    # Detect header-duplicate Q&A (item 0)
    desc = (dataset.get("description") or "").strip()
    if normalized_items:
        ans0 = (normalized_items[0].get("answer") or "").strip()
        if looks_like_header_duplicate(desc, ans0):
            md0 = normalized_items[0].setdefault("metadata", {})
            md0["is_dataset_header_like"] = True
            print("  FLAG: first item looks like dataset header duplicate.")

    out = {
        "dataset": dataset,
        "items": normalized_items,
    }

    out_path = path.with_suffix(".normalized.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  Wrote: {out_path}")


#############################
# Main Walker
#############################

def main():
    for top in sorted(DATASETS_DIR.iterdir()):
        if not top.is_dir() or top.name in SKIP_TOP:
            continue

        scale = top.name
        print(f"\n=== Entering top-level: {top} (scale={scale}) ===")

        # Walk recursively for *.json
        for path in top.rglob("*.json"):
            if path.name in SKIP_FILES:
                continue

            # Skip already-normalized files
            if "normalized" in path.name:
                continue

            # Only process dataset-style JSONs
            try:
                temp = json.loads(path.read_text())
            except:
                continue

            if "items" not in temp:
                continue

            # relativized collection, e.g. core/psych
            rel_collection = str(path.relative_to(DATASETS_DIR).parent)

            process_dataset_file(path, scale, rel_collection)


if __name__ == "__main__":
    main()

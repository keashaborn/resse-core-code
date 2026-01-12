#!/usr/bin/env python3
"""
normalize_foundational.py

Phase 0: structural normalization ONLY (no GPT yet).

- Discover foundational JSON datasets
- Detect type (qa vs definition) from dataset.name
- Normalize each item to the canonical spine:
  - ensure tags.* are arrays
  - ensure notes, axioms_* exist
  - ensure metadata fields exist (scale, form, collection, version)
- Write out a *.normalized.json file next to the original
"""

import json
from copy import deepcopy
from pathlib import Path


DATASETS_DIR = Path(__file__).parent / "datasets"
FOUNDATIONAL_DIR = DATASETS_DIR / "foundational"

FOUNDATIONAL_COLLECTION = "foundational"
FOUNDATIONAL_SCALE = "core"
FOUNDATIONAL_VERSION = "1.0"


def detect_type_from_name(dataset_name: str) -> str:
    """Decide whether this foundational dataset is qa or definition."""
    name = (dataset_name or "").lower()
    if "definition" in name:
        return "definition"
    # For Phase 0, everything else in foundational is treated as qa
    return "qa"


def ensure_list(value):
    """Ensure value is a list. Scalars -> [scalar], None -> []."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_tags(raw_tags: dict | None) -> dict:
    """Normalize the tags block to the canonical structure."""
    raw_tags = raw_tags or {}

    authority = ensure_list(raw_tags.get("authority"))
    category = ensure_list(raw_tags.get("category"))
    tone = ensure_list(raw_tags.get("tone"))
    purpose = ensure_list(raw_tags.get("purpose"))
    frames = ensure_list(raw_tags.get("frames"))
    emotion = ensure_list(raw_tags.get("emotion"))

    # intent can be dict or list or missing
    raw_intent = raw_tags.get("intent") or {}
    if isinstance(raw_intent, dict):
        intent_q = ensure_list(raw_intent.get("question"))
        intent_a = ensure_list(raw_intent.get("answer"))
    else:
        # Fallback: if intent is a list or string, treat as answer intents
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

def normalize_metadata(raw_meta: dict | None, source_file: str, pair_index: int, item_type: str) -> dict:
    """Normalize metadata with defaults for foundational datasets."""
    raw_meta = deepcopy(raw_meta) if raw_meta else {}
    raw_meta.setdefault("source_file", source_file)
    raw_meta.setdefault("pair_index", pair_index)
    raw_meta.setdefault("scale", FOUNDATIONAL_SCALE)
    raw_meta.setdefault("form", item_type)  # qa | definition
    raw_meta.setdefault("collection", FOUNDATIONAL_COLLECTION)
    raw_meta.setdefault("version", FOUNDATIONAL_VERSION)
    return raw_meta


def normalize_qa_item(item: dict, dataset_name: str, source_file: str, idx: int) -> dict:
    """Normalize a QA item to the canonical schema."""
    out = {}

    out["id"] = item.get("id", f"qa-{idx:04d}")
    out["type"] = "qa"

    # Question & answer
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    # If question is missing but answer looks like a description, we leave as-is for now
    # In later phases we can use GPT to repair.
    out["question"] = question
    out["answer"] = answer

    # Tags / notes / metadata
    out["tags"] = normalize_tags(item.get("tags"))
    out["notes"] = ensure_list(item.get("notes"))

    pair_index = item.get("metadata", {}).get("pair_index", idx)
    out["metadata"] = normalize_metadata(item.get("metadata"), source_file, pair_index, "qa")

    # Axiom linkage placeholders
    axioms_primary = item.get("axioms_primary")
    axioms_secondary = item.get("axioms_secondary")
    out["axioms_primary"] = ensure_list(axioms_primary)
    out["axioms_secondary"] = ensure_list(axioms_secondary)

    return out


def normalize_definition_item(item: dict, dataset_name: str, source_file: str, idx: int) -> dict:
    """Normalize a Definition item to the canonical schema."""
    out = {}

    out["id"] = item.get("id", f"defs-{idx:04d}")
    out["type"] = "definition"

    term = item.get("term", "").strip()
    defs = item.get("definitions") or []

    out["term"] = term
    out["definitions"] = [d.strip() for d in defs if isinstance(d, str) and d.strip()]

    out["tags"] = normalize_tags(item.get("tags"))
    out["notes"] = ensure_list(item.get("notes"))

    pair_index = item.get("metadata", {}).get("pair_index", idx)
    out["metadata"] = normalize_metadata(item.get("metadata"), source_file, pair_index, "definition")

    axioms_primary = item.get("axioms_primary")
    axioms_secondary = item.get("axioms_secondary")
    out["axioms_primary"] = ensure_list(axioms_primary)
    out["axioms_secondary"] = ensure_list(axioms_secondary)

    return out


def normalize_dataset_file(path: Path) -> None:
    """Normalize a single foundational dataset file and write a *.normalized.json beside it."""
    print(f"\n=== Normalizing file: {path.name} ===")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  ERROR: could not parse JSON: {e}")
        return

    dataset = data.get("dataset", {})
    items = data.get("items", [])

    ds_name = dataset.get("name", "")
    ds_type = detect_type_from_name(ds_name)

    print(f"  Dataset name  : {ds_name or '<no name>'}")
    print(f"  Detected type : {ds_type}")
    print(f"  Item count    : {len(items)}")

    normalized_items = []
    source_file = path.name

    for idx, item in enumerate(items):
        if ds_type == "qa":
            normalized = normalize_qa_item(item, ds_name, source_file, idx)
        elif ds_type == "definition":
            normalized = normalize_definition_item(item, ds_name, source_file, idx)
        else:
            # For Phase 0 we only expect qa/definition in foundational
            print(f"  WARNING: unexpected type '{ds_type}' for item {idx}, copying raw item")
            normalized = item
        normalized_items.append(normalized)

    # Build new dataset object
    normalized_data = {
        "dataset": dataset,
        "items": normalized_items,
    }

    out_path = path.with_suffix(".normalized.json")
    out_path.write_text(json.dumps(normalized_data, indent=2, ensure_ascii=False))
    print(f"  Wrote normalized file: {out_path.name}")


def main() -> None:
    if not FOUNDATIONAL_DIR.exists():
        print(f"Foundational dir not found: {FOUNDATIONAL_DIR}")
        return

    json_files = sorted(FOUNDATIONAL_DIR.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {FOUNDATIONAL_DIR}")
        return

    print(f"Found {len(json_files)} foundational JSON file(s) in {FOUNDATIONAL_DIR}:")

    for path in json_files:
        # Skip manifest file; it has no items
        if "manifest" in path.name:
            print(f"\n=== Skipping manifest file: {path.name} ===")
            continue
        normalize_dataset_file(path)


if __name__ == "__main__":
    main()

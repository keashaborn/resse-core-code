#!/usr/bin/env python3
"""
normalize_kernel.py

Phase 0: structural normalization for kernel datasets (no GPT yet).

- Walk datasets/kernel/* subdirectories
- Detect canonical type from subfolder name
- Normalize each item to the shared spine:

  - Ensure tags.* are arrays
  - Move tag-like fields out of metadata when needed (qa_identity, etc.)
  - Ensure notes, axioms_* exist
  - Ensure metadata fields exist (scale, form, collection, version)

- Write out a *.normalized.json file next to the original.
"""

import json
from copy import deepcopy
from pathlib import Path


DATASETS_DIR = Path(__file__).parent / "datasets"
KERNEL_DIR = DATASETS_DIR / "kernel"

# Map subdirectory names to canonical type + default scale
KERNEL_SUBDIR_CONFIG = {
    # QA variants
    "qa_identity": {
        "folder_type": "qa",
        "scale": "identity",
    },
    "qa_archetype": {
        "folder_type": "qa",
        "scale": "archetype",
    },
    "qa_com_style": {
        "folder_type": "qa",
        "scale": "style",
    },

    # Stimulus/response
    "sr_archetype": {
        "folder_type": "stimulus_response",
        "scale": "archetype",
    },

    # Imperative / reasoning
    "imp_reasoning": {
        "folder_type": "method_step",  # default; individual items may specify 'directive'
        "scale": "reasoning",
    },
    "imp_archetype": {
        "folder_type": "directive",  # default; or method_step depending on items
        "scale": "archetype",
    },

    # Sarcasm / tone
    "ps_sarcasm": {
        "folder_type": "sarcasm_pair",
        "scale": "sarcasm",
    },

    # Declarative identity
    # Identity recursive framework (these are method steps/directives, not bare statements)
    "d_identity": {
        "folder_type": "method_step",
        "scale": "identity",
    },
}


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


def normalize_metadata(
    raw_meta: dict | None,
    source_file: str,
    pair_index: int,
    item_type: str,
    scale: str,
    collection: str,
) -> dict:
    """Normalize metadata with defaults for kernel datasets."""
    raw_meta = deepcopy(raw_meta) if raw_meta else {}
    raw_meta.setdefault("source_file", source_file)
    raw_meta.setdefault("pair_index", pair_index)
    raw_meta.setdefault("scale", scale)
    raw_meta.setdefault("form", item_type)
    raw_meta.setdefault("collection", collection)
    raw_meta.setdefault("version", "1.0")
    return raw_meta


def extract_tags_from_metadata(raw_meta: dict) -> tuple[dict, list, dict]:
    """
    Some kernel QA items (e.g. qa_identity) store tag-like fields inside metadata.
    This pulls them out into a tags dict + notes list, and returns the cleaned metadata.
    """
    raw_meta = deepcopy(raw_meta) if raw_meta else {}

    # Fields that belong in tags
    tag_fields = ["authority", "category", "tone", "purpose", "frames", "emotion"]
    tag_src = {key: raw_meta.pop(key, None) for key in tag_fields}

    # Notes may also live in metadata
    notes_src = raw_meta.pop("notes", None)

    tags = normalize_tags(tag_src)
    notes = ensure_list(notes_src)

    return tags, notes, raw_meta


def normalize_common_spine(
    out: dict,
    raw_item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    item_type: str,
    scale: str,
) -> dict:
    """
    Apply the shared spine fields:
    - tags
    - notes
    - metadata
    - axioms_primary / axioms_secondary
    """
    collection = f"kernel/{folder_name}"

    # Decide where tags/notes come from:
    if "tags" in raw_item:
        tags = normalize_tags(raw_item.get("tags"))
        # notes might be in raw_item or in metadata
        notes = ensure_list(raw_item.get("notes"))
        cleaned_meta = raw_item.get("metadata", {})
    else:
        # Some QA items (e.g. qa_identity) store tags-like fields in metadata
        tags, notes, cleaned_meta = extract_tags_from_metadata(raw_item.get("metadata") or {})

    pair_index = cleaned_meta.get("pair_index", idx)
    metadata = normalize_metadata(
        cleaned_meta,
        source_file=source_file,
        pair_index=pair_index,
        item_type=item_type,
        scale=scale,
        collection=collection,
    )

    axioms_primary = ensure_list(raw_item.get("axioms_primary"))
    axioms_secondary = ensure_list(raw_item.get("axioms_secondary"))

    out["tags"] = tags
    out["notes"] = notes
    out["metadata"] = metadata
    out["axioms_primary"] = axioms_primary
    out["axioms_secondary"] = axioms_secondary

    return out


def normalize_kernel_qa_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    scale: str,
) -> dict:
    """Normalize a kernel QA item (qa_identity, qa_archetype, qa_com_style)."""
    out = {}
    out["id"] = item.get("id", f"{folder_name}-qa-{idx:04d}")
    out["type"] = "qa"

    question = (item.get("question") or "").strip()
    answer = (item.get("answer") or "").strip()

    out["question"] = question
    out["answer"] = answer

    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type="qa",
        scale=scale,
    )
    return out


def normalize_kernel_method_or_directive_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    default_type: str,
    scale: str,
) -> dict:
    """
    Normalize imperatives / reasoning items.
    We honor an existing item['type'] if it's 'method_step' or 'directive';
    otherwise we use default_type from subdir config.
    """
    out = {}
    raw_type = (item.get("type") or "").strip()
    if raw_type in ("method_step", "directive"):
        item_type = raw_type
    else:
        item_type = default_type

    out["id"] = item.get("id", f"{folder_name}-{item_type}-{idx:04d}")
    out["type"] = item_type

    directive = (item.get("directive") or "").strip()
    context = (item.get("context") or "").strip()

    out["directive"] = directive
    if context:
        out["context"] = context

    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type=item_type,
        scale=scale,
    )
    return out


def normalize_kernel_stimulus_response_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    scale: str,
) -> dict:
    """Normalize archetype SR items."""
    out = {}
    out["id"] = item.get("id", f"{folder_name}-sr-{idx:04d}")
    out["type"] = "stimulus_response"

    out["stimulus"] = (item.get("stimulus") or "").strip()
    out["response"] = (item.get("response") or "").strip()

    context = (item.get("context") or "").strip()
    style_ref = (item.get("style_reference") or "").strip()

    if context:
        out["context"] = context
    if style_ref:
        out["style_reference"] = style_ref

    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type="stimulus_response",
        scale=scale,
    )
    return out


def normalize_kernel_sarcasm_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    scale: str,
) -> dict:
    """Normalize sarcasm pairs."""
    out = {}
    out["id"] = item.get("id", f"{folder_name}-sarcasm-{idx:04d}")
    out["type"] = "sarcasm_pair"

    out["prompt"] = (item.get("prompt") or "").strip()
    out["sarcastic"] = (item.get("sarcastic") or "").strip()
    context = (item.get("context") or "").strip()
    if context:
        out["context"] = context

    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type="sarcasm_pair",
        scale=scale,
    )
    return out


def normalize_kernel_persona_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    scale: str,
) -> dict:
    """Normalize persona archetype items (Keeper, Lucifer core, etc.)."""
    out = {}
    out["id"] = item.get("id", f"{folder_name}-persona-{idx:04d}")
    out["type"] = "persona"

    # Front persona fields
    out["name"] = (item.get("name") or "").strip()
    out["description"] = (item.get("description") or "").strip()
    primary_directive = (item.get("primary_directive") or "").strip()
    if primary_directive:
        out["primary_directive"] = primary_directive

    imperatives = item.get("imperatives") or []
    out["imperatives"] = [s.strip() for s in imperatives if isinstance(s, str) and s.strip()]

    example_prompt = (item.get("example_prompt") or "").strip()
    if example_prompt:
        out["example_prompt"] = example_prompt

    # Shared spine (tags, notes, metadata, axioms)
    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type="persona",
        scale=scale,
    )
    return out



def normalize_kernel_declarative_item(
    item: dict,
    folder_name: str,
    source_file: str,
    idx: int,
    scale: str,
) -> dict:
    """Normalize declarative identity items (approximate; refine later if needed)."""
    out = {}
    out["id"] = item.get("id", f"{folder_name}-decl-{idx:04d}")
    out["type"] = "declarative"

    statement = (item.get("statement") or item.get("content") or "").strip()
    explanation = (item.get("explanation") or "").strip()

    out["statement"] = statement
    if explanation:
        out["explanation"] = explanation

    out = normalize_common_spine(
        out=out,
        raw_item=item,
        folder_name=folder_name,
        source_file=source_file,
        idx=idx,
        item_type="declarative",
        scale=scale,
    )
    return out


def normalize_kernel_dataset_file(folder: Path, path: Path, folder_type: str, scale: str) -> None:
    """Normalize a single kernel dataset file and write a *.normalized.json beside it."""
    print(f"\n=== Normalizing file: {folder.name}/{path.name} ===")

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  ERROR: could not parse JSON: {e}")
        return

    dataset = data.get("dataset", {})
    items = data.get("items", [])

    ds_name = dataset.get("name", "")
    print(f"  Dataset name  : {ds_name or '<no name>'}")
    print(f"  Folder type   : {folder_type}")
    print(f"  Item count    : {len(items)}")

    normalized_items = []
    source_file = path.name

    for idx, item in enumerate(items):
        # If this item is explicitly a persona, normalize as persona regardless of folder_type
        raw_type = (item.get("type") or "").strip()
        if raw_type == "persona":
            normalized = normalize_kernel_persona_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                scale=scale,
            )
            normalized_items.append(normalized)
            continue

        # Special case: stray QA items inside imperative folders (e.g. eva-rgi-0001)
        if folder_type in ("method_step", "directive"):
            has_q = "question" in item
            has_a = "answer" in item
            directive_text = (item.get("directive") or "").strip()
            if has_q and has_a and not directive_text:
                normalized = normalize_kernel_qa_item(
                    item=item,
                    folder_name=folder.name,
                    source_file=source_file,
                    idx=idx,
                    scale=scale,
                )
                normalized_items.append(normalized)
                continue
        if folder_type == "qa":
            normalized = normalize_kernel_qa_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                scale=scale,
            )
        elif folder_type in ("method_step", "directive"):
            normalized = normalize_kernel_method_or_directive_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                default_type=folder_type,
                scale=scale,
            )
        elif folder_type == "stimulus_response":
            normalized = normalize_kernel_stimulus_response_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                scale=scale,
            )
        elif folder_type == "sarcasm_pair":
            normalized = normalize_kernel_sarcasm_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                scale=scale,
            )
        elif folder_type == "declarative":
            normalized = normalize_kernel_declarative_item(
                item=item,
                folder_name=folder.name,
                source_file=source_file,
                idx=idx,
                scale=scale,
            )
        else:
            print(f"  WARNING: unsupported folder_type '{folder_type}' for {folder.name}; copying raw item")
            normalized = item

        normalized_items.append(normalized)

    normalized_data = {
        "dataset": dataset,
        "items": normalized_items,
    }

    out_path = path.with_suffix(".normalized.json")
    out_path.write_text(json.dumps(normalized_data, indent=2, ensure_ascii=False))
    print(f"  Wrote normalized file: {folder.name}/{out_path.name}")


def main() -> None:
    if not KERNEL_DIR.exists():
        print(f"Kernel dir not found: {KERNEL_DIR}")
        return

    print(f"Kernel root: {KERNEL_DIR}")

    for subdir in sorted(KERNEL_DIR.iterdir()):
        if not subdir.is_dir():
            # Skip manifest_kernel.json and any stray files
            continue

        folder_name = subdir.name
        cfg = KERNEL_SUBDIR_CONFIG.get(folder_name)
        if cfg is None:
            print(f"\n=== Skipping unknown kernel subdir: {folder_name} ===")
            continue

        folder_type = cfg["folder_type"]
        scale = cfg["scale"]

        all_json = sorted(subdir.glob("*.json"))
        # Only normalize base files, not already-normalized ones
        json_files = [p for p in all_json if not p.name.endswith(".normalized.json")]

        if not json_files:
            print(f"\n=== No base .json files in kernel/{folder_name} ===")
            continue
        print(f"\n=== Processing kernel subdir: {folder_name} (type={folder_type}, scale={scale}) ===")

        for path in json_files:
            normalize_kernel_dataset_file(
                folder=subdir,
                path=path,
                folder_type=folder_type,
                scale=scale,
            )


if __name__ == "__main__":
    main()

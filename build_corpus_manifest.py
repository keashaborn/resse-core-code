#!/usr/bin/env python3
"""
build_corpus_manifest.py

Builds a manifest of all Q&A content datasets that should be part of the main corpus.
- Includes foundational/core/applied/ref_* JSONs
- Excludes kernel (personas/behavior) and foundational_definitions
- Excludes manifest files and any non-dataset JSONs (no "items" key)
Writes: corpus_manifest.json
"""

from pathlib import Path
import json

DATASETS_DIR = Path(__file__).parent / "datasets"

def is_content_dataset(path: Path) -> bool:
    rel = path.relative_to(DATASETS_DIR)
    parts = rel.parts
    name = path.name

    # Skip kernel entirely
    if parts and parts[0] == "kernel":
        return False

    # Skip foundational definitions from main Q&A corpus
    if "foundational_definitions" in str(rel):
        return False

    # Skip manifest files
    if name.startswith("manifest_") or name == "manifest_kernel.json":
        return False

    # Must be a JSON with "items"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False

    if not isinstance(data, dict) or "items" not in data:
        return False

    # Only include datasets where items are Q&A-like (have "question" and "answer" in first item)
    items = data.get("items") or []
    if not items:
        return False
    first = items[0]
    if "question" in first and "answer" in first:
        return True

    return False

def main():
    manifest = []
    for path in sorted(DATASETS_DIR.rglob("*.json")):
        rel = path.relative_to(DATASETS_DIR)
        if not is_content_dataset(path):
            continue
        manifest.append(str(rel))

    out_path = DATASETS_DIR.parent / "corpus_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Written {len(manifest)} entries to {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_nta_corpus.py

Use corpus_manifest.json to build a clean NTA corpus under ./resse_nta:

For each path in corpus_manifest.json, e.g.
  "foundational/foundational_core.normalized.axiomed.json"

We create:
  ./resse_nta/foundational/foundational_core.nta.json

Where "nta" means:
  - normalized
  - tagged
  - axiomed
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"
MANIFEST_PATH = ROOT / "corpus_manifest.json"
NTA_ROOT = ROOT / "resse_nta"

# Suffixes we know how to strip to get the base name
PREF_SUFFIXES = [
    ".normalized.enriched.axiomed.json",
    ".normalized.axiomed.json",
    ".normalized.enriched.json",
    ".normalized.json",
    ".json",
]

def strip_suffixes(rel: Path) -> Path:
    """
    Given a relative path like
      foundational/foundational_core.normalized.axiomed.json

    Return:
      foundational/foundational_core
    """
    s = str(rel)
    for suf in PREF_SUFFIXES:
        if s.endswith(suf):
            return Path(s[: -len(suf)])
    # If no known suffix matched, return rel without change
    return rel.with_suffix("")


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run build_corpus_manifest.py first.")

    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"Loaded manifest with {len(manifest)} entries")

    count = 0

    for rel_str in manifest:
        src_rel = Path(rel_str)
        src = DATASETS_DIR / src_rel

        if not src.exists():
            print(f"SKIP (missing source): {src}")
            continue

        # Compute the base path (no suffixes) relative to datasets/
        base_rel = strip_suffixes(src_rel)

        # Destination path in resse_nta with .nta.json extension
        dst = NTA_ROOT / base_rel
        dst = dst.with_suffix(".nta.json")

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Read source and write as-is to NTA path
        data = json.loads(src.read_text())
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        count += 1
        print(f"Copied {src} -> {dst}")

    print(f"\nDone. Wrote {count} NTA files under {NTA_ROOT}")

if __name__ == "__main__":
    main()

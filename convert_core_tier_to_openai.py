"""
convert_core_tier_to_openai.py

Convert Tier 1 core JSON files to OpenAI fine-tune format.

Reads from:
  datasets/resse_json/core/metaphy/*.json
  datasets/resse_json/core/psych/*.json

Writes:
  datasets/core_metaphy_openai.jsonl
  datasets/core_psych_openai.jsonl

Each input object is expected to have at least:
  "input": "<user text>",
  "output": "<assistant text>"

Each output line is:
  {"messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ]}
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "datasets" / "resse_json"  # top-level corpus folder
CORE_DIR = BASE / "core"

TARGETS = {
    "metaphy": ROOT / "datasets" / "core_metaphy_openai.jsonl",
    "psych":  ROOT / "datasets" / "core_psych_openai.jsonl",
}


def convert_folder(name: str, out_path: Path):
    src_dir = CORE_DIR / name
    if not src_dir.is_dir():
        print(f"[!] Source folder does not exist: {src_dir}")
        return

    print(f"\n[+] Converting core/{name} from {src_dir}")
    count_files = 0
    count_examples = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for path in sorted(src_dir.glob("*.json")):
            count_files += 1
            print(f"  - {path.name}")
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"    ! Skipping {path.name} (JSON parse error: {e})")
                continue

            # Handle either list-of-objects or single-object JSON
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                print(f"    ! Unexpected top-level type in {path.name}: {type(data)}")
                continue

            for obj in items:
                # We expect your schema uses 'input' and 'output' per TRAINING_SCHEMA_NOTES
                q = (obj.get("input") or "").strip()
                a = (obj.get("output") or "").strip()
                if not q or not a:
                    continue

                record = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_examples += 1

    print(f"  => Read {count_files} JSON files, wrote {count_examples} Q/A examples to {out_path}")


def main():
    for folder, out_path in TARGETS.items():
        convert_folder(folder, out_path)


if __name__ == "__main__":
    main()

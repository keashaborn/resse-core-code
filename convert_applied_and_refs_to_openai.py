"""
convert_applied_and_refs_to_openai.py

Converts Tier 2/3 non-kernel corpora to OpenAI fine-tune format.

Buckets:
  - applied      -> datasets/applied_openai.jsonl
  - ref_psy      -> datasets/ref_psy_openai.jsonl
  - ref_phi      -> datasets/ref_phi_openai.jsonl
  - ref_misc     -> datasets/ref_misc_openai.jsonl

Source layout (adjust if needed):
  datasets/resse_json/applied/*.json or train_*_train.jsonl
  datasets/resse_json/ref_psy/*.json or train_*_train.jsonl
  datasets/resse_json/ref_phi/*.json or train_*_train.jsonl
  datasets/resse_json/ref_misc/*.json or train_*_train.jsonl

Each example must have at least:
  "input": "<user text>",
  "output": "<assistant text>"

We emit:
  {"messages": [
      {"role": "user", "content": "<input>"},
      {"role": "assistant", "content": "<output>"}
  ]}
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "datasets" / "resse_json"

# Adjust folder names here if your dirs are named slightly differently
BUCKETS = {
    "applied":  ROOT / "datasets" / "applied_openai.jsonl",
    "ref_psy":  ROOT / "datasets" / "ref_psy_openai.jsonl",
    "ref_phi":  ROOT / "datasets" / "ref_phi_openai.jsonl",
    "ref_misc": ROOT / "datasets" / "ref_misc_openai.jsonl",
}


def convert_bucket(bucket_name: str, out_path: Path):
    src_dir = BASE / bucket_name
    if not src_dir.exists():
        print(f"[!] Skipping '{bucket_name}': directory not found at {src_dir}")
        return

    print(f"\n[+] Converting bucket '{bucket_name}' from {src_dir}")

    # Prefer train_*_train.jsonl if present; otherwise use *.json
    train_files = sorted(src_dir.glob("train_*_train.jsonl"))
    use_jsonl = bool(train_files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_in = 0
    total_out = 0

    with out_path.open("w", encoding="utf-8") as fout:
        if use_jsonl:
            print(f"  Using train_*_train.jsonl files: {[p.name for p in train_files]}")
            for tf in train_files:
                total_in, total_out = _convert_from_jsonl(tf, fout, total_in, total_out)
        else:
            json_files = [
                p for p in sorted(src_dir.glob("*.json"))
                if not p.name.startswith("manifest_")
            ]
            print(f"  Using JSON batch files: {[p.name for p in json_files]}")
            for jf in json_files:
                total_in, total_out = _convert_from_json(jf, fout, total_in, total_out)

    print(
        f"  => Finished bucket '{bucket_name}': "
        f"{total_in} raw items, {total_out} examples written to {out_path}"
    )


def _convert_from_jsonl(path: Path, fout, total_in: int, total_out: int):
    """Read a train_*.jsonl file with one JSON object per line."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total_in += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"    ! Skipping malformed line {line_no} in {path.name}: {e}")
                    continue

                q, a = _extract_qa(obj)
                if not q or not a:
                    continue

                record = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_out += 1

    except Exception as e:
        print(f"    ! Error reading {path}: {e}")

    return total_in, total_out


def _convert_from_json(path: Path, fout, total_in: int, total_out: int):
    """Read a batch JSON file and extract items (supports top-level dict with 'items')."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    ! Error parsing {path.name}: {e}")
        return total_in, total_out

    # Case: {"dataset": {...}, "items": [...]}
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            items = data["items"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        print(f"    ! Unexpected top-level type in {path.name}: {type(data)}")
        return total_in, total_out

    for idx, obj in enumerate(items, start=1):
        total_in += 1
        q, a = _extract_qa(obj)
        if not q or not a:
            continue

        record = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_out += 1

    return total_in, total_out


def _extract_qa(obj: dict):
    q = (
        (obj.get("input") or "")
        or (obj.get("question") or "")
        or (obj.get("prompt") or "")
        or (obj.get("title") or "")
    ).strip()

    a = (
        (obj.get("output") or "")
        or (obj.get("answer") or "")
        or (obj.get("response") or "")
        or (obj.get("text") or "")
        or (obj.get("definition") or "")
        or (obj.get("content") or "")
    ).strip()

    # Skip clearly non-question artifacts (like YAML '---' front-matter)
    if q in ("", "---"):
        return "", ""

    return q, a

def main():
    for bucket_name, out_path in BUCKETS.items():
        convert_bucket(bucket_name, out_path)


if __name__ == "__main__":
    main()

"""
convert_foundational_to_openai.py

Reads:
  datasets/foundational/train_foundational_train.jsonl

Writes:
  datasets/foundational/fm_foundational_openai.jsonl

Each input line like:
  {"input": "...", "output": "...", ...}

Becomes:
  {"messages": [
      {"role": "user", "content": "<input>"},
      {"role": "assistant", "content": "<output>"}
  ]}
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "datasets" / "foundational" / "train_foundational_train.jsonl"
OUT_PATH = ROOT / "datasets" / "foundational" / "fm_foundational_openai.jsonl"

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    count_in = 0
    count_out = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            count_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line {count_in}")
                continue

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
            count_out += 1

    print(f"Read {count_in} lines from {IN_PATH.name}")
    print(f"Wrote {count_out} examples to {OUT_PATH.name}")

if __name__ == "__main__":
    main()

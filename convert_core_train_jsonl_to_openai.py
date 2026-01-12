"""
convert_core_train_jsonl_to_openai.py

Convert Tier 1 core *train* JSONL files to OpenAI fine-tune format.

Reads:
  datasets/resse_json/core/metaphy/train_core_metaphy_train.jsonl
  datasets/resse_json/core/psych/train_core_psych_train.jsonl

Writes:
  datasets/core_metaphy_openai.jsonl
  datasets/core_psych_openai.jsonl

Each input line is expected to be a JSON object with at least:
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

# Adjust these if your train files have slightly different names:
IN_METAPHY = ROOT / "datasets" / "resse_json" / "core" / "metaphy" / "train_core_metaphy_train.jsonl"
IN_PSYCH   = ROOT / "datasets" / "resse_json" / "core" / "psych"   / "train_core_psych_train.jsonl"

OUT_METAPHY = ROOT / "datasets" / "core_metaphy_openai.jsonl"
OUT_PSYCH   = ROOT / "datasets" / "core_psych_openai.jsonl"

IN_AI      = ROOT / "datasets" / "resse_json" / "core" / "ai"      / "train_core_ai_train.jsonl"
IN_SCIENCE = ROOT / "datasets" / "resse_json" / "core" / "science" / "train_core_science_train.jsonl"

OUT_AI      = ROOT / "datasets" / "core_ai_openai.jsonl"
OUT_SCIENCE = ROOT / "datasets" / "core_science_openai.jsonl"



def convert_train_jsonl(in_path: Path, out_path: Path, label: str):
    if not in_path.exists():
        print(f"[!] Missing input file for {label}: {in_path}")
        return

    print(f"\n[+] Converting {label} from {in_path}")
    n_in = 0
    n_out = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ! Skipping malformed line {n_in} in {in_path.name}: {e}")
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
            n_out += 1

    print(f"  => Read {n_in} lines from {in_path.name}, wrote {n_out} examples to {out_path.name}")


def main():
    convert_train_jsonl(IN_METAPHY, OUT_METAPHY, "core/metaphy")
    convert_train_jsonl(IN_PSYCH,   OUT_PSYCH,   "core/psych")
    convert_train_jsonl(IN_AI,      OUT_AI,      "core/ai")
    convert_train_jsonl(IN_SCIENCE, OUT_SCIENCE, "core/science")

if __name__ == "__main__":
    main()

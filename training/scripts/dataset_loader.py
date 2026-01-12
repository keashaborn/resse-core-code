"""
dataset_loader.py

Minimal dataset loader for RESSE JSONL training files.
Right now it:
  - opens a given JSONL file
  - converts each line to a (sequence, raw) pair
  - prints a few examples for sanity
Later, we'll wrap this in a proper Dataset class.
"""

import json
from pathlib import Path
from sequence_builder import build_qa_sequence

ROOT = Path(__file__).resolve().parents[2]  # /home/ubuntu/resse-core
DATASETS = ROOT / "datasets"

def iter_jsonl(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj

def demo_load_foundational(n_examples: int = 3):
    src = DATASETS / "foundational" / "train_foundational_train.jsonl"
    print("ROOT:", ROOT)
    print("Loading:", src)
    if not src.exists():
        print("❌ File not found:", src)
        return

    for i, obj in enumerate(iter_jsonl(src)):
        seq = build_qa_sequence(obj.get("input", ""), obj.get("output", ""))
        print(f"\n--- Example {i+1} ---")
        print("source_file:", obj.get("source_file"))
        print("id:", obj.get("id"))
        print("sequence:\n", seq[:400])  # truncate for printing
        if i + 1 >= n_examples:
            break

if __name__ == "__main__":
    demo_load_foundational()

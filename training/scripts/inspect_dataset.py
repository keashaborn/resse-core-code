import json
from pathlib import Path

# ROOT should be /home/ubuntu/resse-core
ROOT = Path(__file__).resolve().parents[2]

# DATASETS should be /home/ubuntu/resse-core/datasets
DATASETS = ROOT / "datasets"

# Choose one dataset to inspect
FILE = DATASETS / "foundational" / "train_foundational_train.jsonl"

def main():
    print("ROOT:", ROOT)
    print("Inspecting file:", FILE)

    if not FILE.exists():
        print("❌ File not found:", FILE)
        return

    with open(FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"⚠️ JSON parse error on line {i+1}:", e)
                break

            print(f"\n--- Example {i+1} ---")
            print("source_file:", obj.get("source_file"))
            print("id:", obj.get("id"))
            print("input:", obj.get("input", "")[:200])
            print("output:", obj.get("output", "")[:200])
            print("tags keys:", list(obj.get("tags", {}).keys()))
            print("metadata keys:", list(obj.get("metadata", {}).keys()))

            if i >= 2:  # just show first 3 examples
                break

if __name__ == "__main__":
    main()

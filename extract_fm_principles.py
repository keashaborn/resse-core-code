#!/usr/bin/env python3
"""
extract_fm_principles.py

Step 1 of the FM growth engine:
- Read fm_tiers/fm_canon_v1.jsonl
- For each Q&A, ask GPT-5.1 to extract a single core principle
- Save to results/fm_principles.jsonl

Each output line looks like:
{
  "id": "...",
  "source_file": "...",
  "question": "...",
  "answer": "...",
  "axioms_primary": [...],
  "axioms_secondary": [...],
  "principle": "...",
  "notes": "..."
}
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI

ROOT = Path(__file__).parent
CANON_PATH = ROOT / "fm_tiers" / "fm_canon_v1.jsonl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "fm_principles.jsonl"

MODEL = "gpt-5.1"
MAX_OUTPUT_TOKENS = 400
TEMPERATURE = 0.3

# to keep costs sane at first, you can limit how many items you process
MAX_ITEMS = None  # set to an int (e.g. 200) if you want to test on a subset

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def build_prompt(item: Dict[str, Any]) -> (str, str):
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    axioms_primary = item.get("axioms_primary") or []
    axioms_secondary = item.get("axioms_secondary") or []
    dataset_meta = item.get("dataset") or {}
    ds_name = dataset_meta.get("name", "unknown_dataset")

    axioms_text = ", ".join(axioms_primary + axioms_secondary) or "(none)"

    system_msg = (
        "You are a Fractal Monism reasoning assistant.\n"
        "Given a single Q&A from the FM canon, plus its linked axioms, "
        "extract ONE clear, concise principle that the answer is really expressing.\n"
        "The principle should be:\n"
        "- monistic and non-dual (no separate substances)\n"
        "- relational and recursive (distinctions within one whole)\n"
        "- 1–2 sentences, not an essay\n"
        "Do NOT drift into CBT or pop psychology language.\n"
    )

    user_msg = (
        f"DATASET: {ds_name}\n"
        f"AXIOMS LINKED: {axioms_text}\n\n"
        f"QUESTION:\n{q}\n\n"
        f"ANSWER:\n{a}\n\n"
        "Now extract ONE core principle this answer expresses.\n"
        "Return ONLY JSON with this structure:\n"
        "{\n"
        '  \"principle\": \"1-2 sentence distilled principle\",\n'
        '  \"notes\": \"optional nuance or clarification\"\n'
        "}\n"
    )

    return system_msg, user_msg


def call_model(system_msg: str, user_msg: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    output = resp.output_text.strip()

    # First try: parse directly
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        # Try to salvage JSON inside { ... }
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = output[start:end + 1]
            return json.loads(inner)
        else:
            print("  [ERROR] Could not parse JSON from model output:")
            print("  ", output[:400].replace("\n", " "), "...")
            raise


def main():
    if not CANON_PATH.exists():
        print(f"Canon file not found: {CANON_PATH}")
        return

    print(f"Reading canon from: {CANON_PATH}")
    print(f"Writing principles to: {OUT_PATH}")

    out_f = OUT_PATH.open("w", encoding="utf-8")

    count = 0
    with CANON_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            count += 1
            if MAX_ITEMS is not None and count > MAX_ITEMS:
                break

            item_id = item.get("id")
            source_file = item.get("metadata", {}).get("source_file")

            print(f"[{count}] Extracting principle for id={item_id}...")

            system_msg, user_msg = build_prompt(item)
            try:
                res = call_model(system_msg, user_msg)
            except Exception as e:
                print(f"  [ERROR] Failed for id={item_id}: {e}")
                continue

            principle = res.get("principle", "").strip()
            notes = res.get("notes", "").strip()

            if not principle:
                print(f"  [WARN] No principle returned for id={item_id}, skipping.")
                continue

            out_obj = {
                "id": item_id,
                "source_file": source_file,
                "question": (item.get("question") or "").strip(),
                "answer": (item.get("answer") or "").strip(),
                "axioms_primary": item.get("axioms_primary") or [],
                "axioms_secondary": item.get("axioms_secondary") or [],
                "principle": principle,
                "notes": notes,
            }

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"\nDone. Wrote principles for {count} canon items to {OUT_PATH}")


if __name__ == "__main__":
    main()
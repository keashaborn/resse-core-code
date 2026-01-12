#!/usr/bin/env python3
"""
generate_axiom_concepts_all.py

Robust generator for conceptual children for all axioms A1–A15.

For each axiom A1..A15:
  - Load the axiom text from core_axioms.yaml
  - Load a few canon seed Q&A from fm_axiom_seeds/AX_seeds.jsonl
  - Call GPT (gpt-5.1) to produce ~6 conceptual children
  - Parse JSON robustly with retry
  - Write results to fm_axiom_concepts/AX_concepts.jsonl

Safe to re-run:
  - If a concepts file already exists for an axiom, it is skipped
    (you can change SKIP_EXISTING to False to overwrite).
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
from openai import OpenAI
from openai import OpenAIError

ROOT = Path(__file__).parent
AXIOMS_YAML = ROOT / "core_axioms.yaml"
SEEDS_DIR = ROOT / "fm_axiom_seeds"
OUT_DIR = ROOT / "fm_axiom_concepts"

OUT_DIR.mkdir(exist_ok=True)

AXIOM_IDS = [f"A{i}" for i in range(1, 16)]

MODEL = "gpt-5.1"  # modern, strong reasoning model

MAX_CONCEPTS_PER_AXIOM = 6
MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.5
RETRIES = 3
SKIP_EXISTING = True  # set to False if you want to overwrite existing files

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_axioms_map() -> Dict[str, str]:
    """Return dict axiom_id -> statement from core_axioms.yaml."""
    data = yaml.safe_load(AXIOMS_YAML.read_text())
    axioms = {}
    for item in data.get("items", []):
        aid = item.get("axiom_id")
        stmt = item.get("statement")
        if aid and stmt:
            axioms[aid] = stmt.strip()
    return axioms


def load_seeds_for_axiom(axiom_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Load up to `limit` seeds for a given axiom from fm_axiom_seeds/AX_seeds.jsonl."""
    path = SEEDS_DIR / f"{axiom_id}_seeds.jsonl"
    if not path.exists():
        print(f"  [WARN] No seeds file for {axiom_id}: {path}")
        return []

    seeds = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seeds.append(obj)
            if len(seeds) >= limit:
                break
    return seeds


def build_prompt(axiom_id: str, axiom_text: str, seeds: List[Dict[str, Any]]) -> (str, str):
    """Build system and user messages for the concept generator."""
    examples = []
    for s in seeds:
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()
        if q or a:
            examples.append(f"Q: {q}\nA: {a}")
    examples_block = "\n\n---\n\n".join(examples) if examples else "(no seeds provided)"

    system_msg = (
        "You are helping define conceptual sub-principles for Fractal Monism.\n"
        f"For a given axiom ({axiom_id}), you will propose "
        f"{MAX_CONCEPTS_PER_AXIOM} core conceptual children.\n"
        "Each child is NOT a Q&A, but a named concept that expresses one facet "
        "of the axiom in clear, precise, non-dual language.\n"
        "Keep each 'summary' to at most 2 sentences.\n"
        "Keep 'notes' to at most 1 sentence.\n"
        "Do NOT talk about CBT, pop psychology, or generic self-help.\n"
        "Stay inside Fractal Monism's monistic, recursive, relational frame."
    )

    user_msg = (
        f"AXIOM {axiom_id} (seed statement):\n{axiom_text}\n\n"
        "Here are a few existing canon Q&A that express this axiom:\n\n"
        f"{examples_block}\n\n"
        f"Now propose {MAX_CONCEPTS_PER_AXIOM} conceptual children of {axiom_id}.\n"
        "Return ONLY JSON with this structure:\n"
        "{\n"
        f'  "axiom_id": "{axiom_id}",\n'
        '  "concepts": [\n'
        "    {\n"
        '      "id": "AX.Y",\n'
        '      "title": "short name",\n'
        '      "summary": "1–2 sentences explaining the idea",\n'
        '      "notes": "optional extra nuance"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return system_msg, user_msg


def call_model(system_msg: str, user_msg: str) -> Dict[str, Any]:
    """Call the OpenAI model and parse JSON robustly."""
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

    # First attempt: direct JSON parse
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("  [DEBUG] First JSON parse failed, trying to salvage inner JSON...")
        # Print limited portion for debugging if needed
        print("  [DEBUG] Raw model output (truncated to 500 chars):")
        print("  ", output[:500].replace("\n", " "), "...")
        # Try inner { ... }
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = output[start:end + 1]
            try:
                return json.loads(inner)
            except json.JSONDecodeError as e:
                print("  [ERROR] Inner JSON parse still failed:", e)
                raise
        else:
            print("  [ERROR] Could not find JSON braces in output.")
            raise


def write_concepts(axiom_id: str, concepts: List[Dict[str, Any]]):
    """Write concept list to fm_axiom_concepts/AX_concepts.jsonl."""
    out_path = OUT_DIR / f"{axiom_id}_concepts.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in concepts:
            # ensure axiom_id is set
            c["axiom_id"] = axiom_id
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  [OK] Wrote {len(concepts)} concepts → {out_path}")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main():
    axioms_map = load_axioms_map()
    print(f"Loaded {len(axioms_map)} axioms from {AXIOMS_YAML}")

    for ax in AXIOM_IDS:
        if ax not in axioms_map:
            print(f"\n[WARN] {ax} not found in core_axioms.yaml, skipping.")
            continue

        out_path = OUT_DIR / f"{ax}_concepts.jsonl"
        if SKIP_EXISTING and out_path.exists():
            print(f"\n[SKIP] {ax}: {out_path} already exists.")
            continue

        print(f"\n=== Generating concepts for {ax} ===")
        axiom_text = axioms_map[ax]

        seeds = load_seeds_for_axiom(ax, limit=5)
        print(f"  Loaded {len(seeds)} seeds for {ax}")

        system_msg, user_msg = build_prompt(ax, axiom_text, seeds)

        # retry loop
        attempt = 0
        while attempt < RETRIES:
            attempt += 1
            try:
                print(f"  [TRY {attempt}/{RETRIES}] Calling model for {ax}...")
                data = call_model(system_msg, user_msg)
                concepts = data.get("concepts", [])
                if not concepts:
                    print("  [WARN] No 'concepts' field returned; will retry.")
                    raise ValueError("No concepts returned")
                # truncate to desired count if more
                if len(concepts) > MAX_CONCEPTS_PER_AXIOM:
                    concepts = concepts[:MAX_CONCEPTS_PER_AXIOM]
                write_concepts(ax, concepts)
                break  # success
            except (OpenAIError, json.JSONDecodeError, ValueError) as e:
                print(f"  [ERROR] Attempt {attempt} for {ax} failed:", e)
                if attempt < RETRIES:
                    wait = 3 * attempt
                    print(f"  [INFO] Waiting {wait} seconds before retry...")
                    time.sleep(wait)
                else:
                    print(f"  [FAIL] Giving up on {ax} after {RETRIES} attempts.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
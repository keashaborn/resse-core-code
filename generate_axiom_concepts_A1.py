#!/usr/bin/env python3
"""
generate_axiom_concepts_A1.py

Use:
- A1 axiom text from core_axioms.yaml
- A few A1 canon seed Q&A from fm_axiom_seeds/A1_seeds.jsonl

to generate ~8 conceptual “children” for A1 (not Q&A yet, just core concepts).

Output:
fm_axiom_concepts/A1_concepts.jsonl
"""

import os
import json
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).parent
AXIOMS_YAML = ROOT / "core_axioms.yaml"
A1_SEEDS = ROOT / "fm_axiom_seeds" / "A1_seeds.jsonl"
OUT_DIR = ROOT / "fm_axiom_concepts"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "A1_concepts.jsonl"

MODEL = "gpt-5.1"   # or "gpt-5.1"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_axiom_A1():
    data = yaml.safe_load(AXIOMS_YAML.read_text())
    for item in data.get("items", []):
        if item.get("axiom_id") == "A1":
            return item.get("statement", "").strip()
    raise RuntimeError("A1 not found in core_axioms.yaml")


def load_sample_seeds(limit=5):
    seeds = []
    with A1_SEEDS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seeds.append(obj)
            if len(seeds) >= limit:
                break
    return seeds


def build_prompt(axiom_text, seeds):
    # Build a compact view of a few seeds
    examples = []
    for s in seeds:
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()
        examples.append(f"Q: {q}\nA: {a}")
    examples_block = "\n\n---\n\n".join(examples)

    system_msg = (
        "You are helping define conceptual sub-principles for Fractal Monism.\n"
        "For a given axiom, you will propose 6 core conceptual children.\n"
        "Each child is NOT a Q&A, but a named concept that expresses one facet "
        "of the axiom in clear, precise, non-dual language.\n"
        "Keep each 'summary' to at most 2 sentences.\n"
        "Keep 'notes' to at most 1 sentence.\n"
        "Do NOT talk about CBT, pop psychology, or generic self-help. "
        "Stay inside Fractal Monism's monistic, recursive, relational frame."
    )

    user_msg = (
        f"AXIOM A1 (seed statement):\n{axiom_text}\n\n"
        "Here are a few existing canon Q&A that express A1:\n\n"
        f"{examples_block}\n\n"
        "Now propose 6 conceptual children of A1.\n"
        "Return ONLY JSON with this structure:\n"
        "{\n"
        '  "axiom_id": "A1",\n'
        '  "concepts": [\n'
        "    {\n"
        '      "id": "A1.1",\n'
        '      "title": "short name",\n'
        '      "summary": "2–4 sentences explaining the idea",\n'
        '      "notes": "optional extra notes or nuances"\n'
        "    },\n"
        "    ... up to ~8 items ...\n"
        "  ]\n"
        "}\n"
    )
    return system_msg, user_msg


def call_model(system_msg, user_msg):
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.65,
        max_output_tokens=1200
    )

    output = resp.output_text.strip()

    # First try: the model followed instructions perfectly
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("---- RAW MODEL OUTPUT (first parse failed) ----")
        print(output)
        print("---- END RAW OUTPUT ----")

        # Try to salvage JSON inside the first { ... } pair
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = output[start:end + 1]
            try:
                return json.loads(inner)
            except json.JSONDecodeError as e:
                print("---- INNER JSON ALSO FAILED ----")
                print(inner)
                print("---- JSON ERROR:", e, "----")
                raise
        else:
            # No braces to salvage; we can only give up
            raise



def main():
    print("Loading A1 axiom...")
    axiom = load_axiom_A1()
    print("A1:", axiom)

    print("Loading A1 seeds...")
    seeds = load_sample_seeds(limit=5)
    print(f"Loaded {len(seeds)} example seeds for A1.")

    system_msg, user_msg = build_prompt(axiom, seeds)

    print("Calling model to generate A1 concepts...")
    data = call_model(system_msg, user_msg)

    concepts = data.get("concepts", [])
    print(f"Model returned {len(concepts)} concepts.")

    # Write each concept as one JSONL line
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in concepts:
            c["axiom_id"] = "A1"
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Done. Wrote A1 concepts to {OUT_FILE}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
evaluate_generated_hv_qna.py

Run the Human Vantage evaluator over a JSONL file of generated Q&A, e.g.:

  results/generated_hv_qna_tuned3.jsonl

Each line is expected to look like:
  {
    "question": "...",
    "answer": "...",
    "question_archetype": "...",
    "answer_archetype": "...",
    "hv_eval": null or {...}
  }

This script:
  - builds a text block "Q: ...\\n\\nA: ..."
  - calls the HV axioms evaluator
  - attaches hv_eval to each item
  - writes a new *.hv.jsonl file

Usage:

  python3 human_vantage/evaluate_generated_hv_qna.py \
    --input results/generated_hv_qna_tuned3.jsonl \
    --output results/generated_hv_qna_tuned3.hv.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent  # ~/resse-core
HV_YAML = BASE_DIR / "human_vantage" / "hv_axioms.yaml"

client = OpenAI()


def load_hv_axioms():
    if not HV_YAML.exists():
        raise FileNotFoundError(f"Human vantage YAML not found at {HV_YAML}")
    with HV_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("items", [])
    return items


def build_hv_axiom_summary(axioms) -> str:
    lines = []
    for ax in axioms:
        ax_id = ax.get("id", "")
        title = ax.get("title", "")
        stmt = (ax.get("statement", "") or "").strip().replace("\n", " ")
        if len(stmt) > 240:
            stmt = stmt[:240].rstrip() + "..."
        lines.append(f"- {ax_id}: {title} — {stmt}")
    return "\n".join(lines)


def build_eval_prompt(text: str, metadata: Dict[str, Any], hv_summary: str) -> str:
    return f"""
You are evaluating a piece of Q&A at the HUMAN VANTAGE layer.

You are given:

1. The Human Vantage axioms (HV-A1–HV-A15):
{hv_summary}

2. A question and answer pair.
3. Some metadata about how it was generated.

Your task:

- Evaluate how well this answer aligns with the Human Vantage axioms.
- Focus especially on:
  • Does it treat identity, traits, and emotion as labels for patterns, not causes?
  • Does it honor vantage, constraints, perception choice, and next acts?
  • Does it avoid inner essences, dualism, and CBT-style inner-fixing?
  • Is the reasoning grounded in loops (cue → perception → behavior → consequence)?
  • Does it give small, realistic next acts when appropriate?

Return a strict JSON object with these fields:

{{
  "hv_alignment_score": float between 0 and 1,
  "vantage_fit_score": float between 0 and 1,
  "emotion_handling": "aligned" | "mixed" | "misframed" | "not_relevant",
  "attention_handling": "aligned" | "mixed" | "misframed" | "not_relevant",
  "constraint_handling": "aligned" | "mixed" | "misframed" | "not_relevant",
  "meaning_handling": "aligned" | "mixed" | "misframed" | "not_relevant",
  "flags": [list of short snake_case flags, e.g. "trait_essentialism", "emotion_as_cause"],
  "bucket": "A" | "B" | "C" | "D",
  "notes": "1–3 sentences explaining your reasoning in plain language."
}}

Definition of buckets:

- "A": Strongly aligned with Human Vantage axioms, clear loop/vantage framing, no major flags.
- "B": Mostly aligned, minor issues or omissions; okay for training but could be tightened.
- "C": Conceptually interesting but has notable Human Vantage issues (e.g., inner-essence talk,
       misframed emotion, unclear constraints).
- "D": Fundamentally misaligned with Human Vantage axioms or very confusing; should be quarantined
       or rewritten.

Here is the Q&A to evaluate:

---
TEXT:
{text}

---
METADATA:
{json.dumps(metadata, ensure_ascii=False)}
"""


def call_hv_eval_api(prompt: str) -> Optional[Dict[str, Any]]:
    resp = client.responses.create(
        model="gpt-5.1",  # or set HV_EVAL_MODEL in env if you prefer
        input=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous evaluator of Human Vantage axioms. "
                    "Always respond with STRICT JSON, no extra text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )
    try:
        content = resp.output[0].content[0].text
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing model response as JSON: {e}")
        try:
            print("Raw output:", resp.output[0].content[0].text)
        except Exception:
            pass
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input generated Q&A JSONL file.")
    parser.add_argument("--output", required=True, help="Output .hv.jsonl file.")
    args = parser.parse_args()

    in_path = BASE_DIR / args.input
    out_path = BASE_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    hv_axioms = load_hv_axioms()
    hv_summary = build_hv_axiom_summary(hv_axioms)

    total = 0
    evaluated = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            obj = json.loads(line)
            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()

            if not question or not answer:
                obj["hv_eval"] = {
                    "hv_alignment_score": None,
                    "vantage_fit_score": None,
                    "flags": ["missing_question_or_answer"],
                    "bucket": "D",
                    "notes": "Question or answer missing; cannot evaluate."
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            text = f"Q: {question}\n\nA: {answer}"
            metadata = {
                "question_archetype": obj.get("question_archetype"),
                "answer_archetype": obj.get("answer_archetype"),
            }

            prompt = build_eval_prompt(text=text, metadata=metadata, hv_summary=hv_summary)
            hv_eval = call_hv_eval_api(prompt)

            if hv_eval is None:
                obj["hv_eval"] = {
                    "hv_alignment_score": None,
                    "vantage_fit_score": None,
                    "flags": ["api_error_or_parse_error"],
                    "bucket": "D",
                    "notes": "HV evaluator failed to produce a valid JSON response."
                }
            else:
                obj["hv_eval"] = hv_eval

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            evaluated += 1

            if evaluated % 10 == 0:
                print(f"  Evaluated {evaluated} items...")

    print(f"Done. Read {total} items, evaluated {evaluated} items.")
    print(f"Wrote HV-evaluated Q&A to: {out_path}")


if __name__ == "__main__":
    main()
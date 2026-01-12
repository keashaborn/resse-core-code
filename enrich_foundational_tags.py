#!/usr/bin/env python3
"""
enrich_foundational_tags.py

Use GPT to fill in tags for foundational items whose tags are completely empty:
- foundational_clarifications_extended.normalized.json
- foundational_expansion.normalized.json

One GPT call per item. Writes *.enriched.json alongside the normalized files.
"""

import json
from copy import deepcopy
from pathlib import Path
import os
from openai import OpenAI

DATASETS_DIR = Path(__file__).parent / "datasets"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TARGET_FILES = [
    # already-enriched foundational sets
    "foundational/foundational_clarifications_extended.normalized.json",
    "foundational/foundational_expansion.normalized.json",
    # new: core AI guidance
    "core/ai/core_functional_ai_guidance.normalized.json",
    # new: core scientific commentary
    "core/science/core_fractal_monism_scientific_commentary.normalized.json",
]

def tags_are_empty(tags: dict) -> bool:
    if not tags:
        return True
    if any(tags.get(k) for k in ["authority", "category", "tone", "purpose", "frames", "emotion"]):
        return False
    intent = tags.get("intent") or {}
    if intent.get("question") or intent.get("answer"):
        return False
    return True


SYSTEM_PROMPT = """You are helping normalize a corpus for a model called RESSE that uses the Fractal Monism worldview.
Your job is to assign high-level semantic tags to a single question–answer pair.

You MUST be conservative:
- Only assign tags when they are clearly supported by the text.
- If you are not sure about a field, leave it as an empty array.
- Do NOT invent new keys. Only use the fields provided in the schema.
- Do NOT change the question or answer.
- Do NOT talk about "Fractal Monism" unless it is clearly relevant to the specific Q&A.
- Output ONLY a single JSON object that fits the schema below. No explanations, no extra text.
"""

USER_TEMPLATE = """Here is the tag schema you must fill:

"tags": {{
  "authority": [],
  "category": [],
  "tone": [],
  "purpose": [],
  "intent": {{
    "question": [],
    "answer": []
  }},
  "frames": [],
  "emotion": []
}}

Given this question and answer:

QUESTION:
{question}

ANSWER:
{answer}

Fill the tag schema above. Remember:
- "category": 1–3 short labels about the conceptual domain (e.g. "metaphysics", "emotion/regulation", "time/experience", "identity/core").
- "tone": a few words about the voice (e.g. "clarifying", "analytic", "gentle", "confrontational", "reassuring").
- "purpose": what the answer is trying to accomplish (e.g. "explain", "clarify", "reframe", "orient", "differentiate", "motivate").
- "intent.question": what the question is doing (e.g. "inquire", "challenge", "probe", or leave empty).
- "intent.answer": what the answer is doing (e.g. "explain", "reframe", "reassure", "confront", "normalize").
- "frames": which relational/structural frames are active (e.g. "distinction", "temporal", "causal", "coordination", "scaling", "comparison").
- "emotion": optional emotional tones in the content (e.g. "relief", "curiosity", "tension", "hope"); leave empty if not obvious.

Return ONLY a single JSON object for the "tags" value, like:

{{
  "authority": [...],
  "category": [...],
  "tone": [...],
  "purpose": [...],
  "intent": {{ "question": [...], "answer": [...] }},
  "frames": [...],
  "emotion": [...]
}}"""


def call_gpt_for_tags(question: str, answer: str) -> dict:
    """Call the Chat Completions API to get a tags JSON object."""
    user_prompt = USER_TEMPLATE.format(question=question, answer=answer)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # or another chat model you have access to
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature = 0.0,
    )

    content = resp.choices[0].message.content
    # We told the model to output ONLY a JSON object, so we can parse it directly
    try:
        tags = json.loads(content)
    except json.JSONDecodeError as e:
        # If something goes wrong, print the raw content for debugging
        print("    JSON parse error, raw content was:")
        print("    ", repr(content))
        raise

    return tags

def enrich_file(path: Path) -> None:
    print(f"\n=== Enriching: {path.name} ===")
    data = json.loads(path.read_text())
    dataset = data.get("dataset", {})
    items = data.get("items", [])

    enriched_items = []
    updated_count = 0

    for item in items:
        new_item = deepcopy(item)
        tags = new_item.get("tags") or {}
        if tags_are_empty(tags):
            q = (new_item.get("question") or "").strip()
            a = (new_item.get("answer") or "").strip()
            if not q or not a:
                print(f"  SKIP (no Q/A) id={new_item.get('id')}")
            else:
                print(f"  Enriching tags for id={new_item.get('id')}")
                try:
                    new_tags = call_gpt_for_tags(q, a)
                    new_item["tags"] = new_tags
                    updated_count += 1
                except Exception as e:
                    print(f"    ERROR calling GPT for id={new_item.get('id')}: {e}")
        enriched_items.append(new_item)

    out = {
        "dataset": dataset,
        "items": enriched_items,
    }

    out_path = path.with_suffix(".enriched.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  Done. Updated {updated_count} items. Wrote: {out_path.name}")


def main() -> None:
    for rel in TARGET_FILES:
        path = DATASETS_DIR / rel
        if not path.exists():
            print(f"Missing file: {path}")
            continue
        enrich_file(path)

if __name__ == "__main__":
    main()


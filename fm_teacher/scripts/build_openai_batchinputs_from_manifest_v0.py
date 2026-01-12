#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def slurp(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(slurp(path))


def iter_jsonl_range(path: Path, start: int, end: int) -> Iterable[Tuple[int, dict]]:
    """
    Stream JSONL objects for line indices [start, end).
    Uses sequential reads (no indexing), but total work ~ total lines processed.
    """
    if start < 0 or end < start:
        raise ValueError(f"bad range start={start} end={end}")
    with path.open("r", encoding="utf-8") as f:
        # fast skip
        for _ in range(start):
            if f.readline() == "":
                return
        i = start
        while i < end:
            line = f.readline()
            if line == "":
                return
            line = line.strip()
            if line:
                yield i, json.loads(line)
            i += 1


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def cluster_id_of(obj: dict) -> str:
    return (
        obj.get("cluster_id")
        or obj.get("seed_fact_id")
        or (obj.get("seed_payload") or {}).get("fact_id")
        or (obj.get("seed_payload") or {}).get("id")
        or obj.get("id")
        or ""
    )


def domain_of(obj: dict, fallback: str) -> str:
    return (
        (obj.get("seed_payload") or {}).get("domain")
        or obj.get("domain")
        or fallback
    )


def topic_of(obj: dict) -> Tuple[Optional[str], Optional[str]]:
    sp = obj.get("seed_payload") or {}
    return sp.get("topic"), sp.get("subtopic")


def extract_facts(obj: dict, max_facts: int, max_fact_chars: int) -> List[str]:
    """
    Convert a cluster json object (seed_payload + members[]) -> list[str] facts.
    Dedup by normalized text.
    """
    out: List[str] = []
    seen = set()

    def add(txt: str):
        t = norm_ws(txt)
        if not t:
            return
        if max_fact_chars and len(t) > max_fact_chars:
            t = t[:max_fact_chars].rstrip() + "…"
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    sp = obj.get("seed_payload") or {}
    add(sp.get("text") or sp.get("canonical_text") or "")

    for m in (obj.get("members") or []):
        add(m.get("text") or m.get("canonical_text") or "")

    # hard cap
    if max_facts and len(out) > max_facts:
        out = out[:max_facts]
    return out


def make_user_prompt(cluster_id: str, domain: str, topic: Optional[str], subtopic: Optional[str], facts: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"cluster_id: {cluster_id}")
    lines.append(f"domain: {domain}")
    if topic:
        lines.append(f"topic: {topic}")
    if subtopic:
        lines.append(f"subtopic: {subtopic}")
    lines.append("facts:")
    for i, f in enumerate(facts, 1):
        lines.append(f"{i}. {f}")
    return "\n".join(lines)


def clamp_prompt(prompt: str, max_chars: int) -> str:
    if max_chars <= 0:
        return prompt
    if len(prompt) <= max_chars:
        return prompt
    # Keep header; trim tail.
    return prompt[:max_chars].rstrip() + "…"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build OpenAI Batch input JSONL from an existing manifest.csv (line ranges).")
    ap.add_argument("--manifest", required=True, help="CSV with src_path,start_line,end_line,suggested_batchinput_name")
    ap.add_argument("--run_order", default="", help="Optional run_order.txt listing suggested_batchinput_name in desired order")
    ap.add_argument("--out_dir", required=True, help="Directory to write *.batchinput.jsonl files")

    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--system_prompt_file", required=True)
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--max_output_tokens", type=int, default=900)

    ap.add_argument("--url", default="/v1/responses", help="Batch request URL")
    ap.add_argument("--mode_tag", default="passR", help="Prefix used in custom_id (does not change request semantics)")

    ap.add_argument("--max_facts", type=int, default=24)
    ap.add_argument("--max_fact_chars", type=int, default=420)
    ap.add_argument("--max_prompt_chars", type=int, default=9000)

    ap.add_argument("--max_requests_per_part", type=int, default=0, help="0 = all; else emit only first N requests per part (smoke)")
    ap.add_argument("--limit_parts", type=int, default=0, help="0 = all; else only process first N parts (smoke)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files instead of erroring")

    return ap.parse_args()


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_doc = load_json(a.schema_file)
    sys_prompt = slurp(a.system_prompt_file)

    # Load manifest
    manifest_rows: List[Dict[str, str]] = []
    with open(a.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manifest_rows.append(r)

    # Index by suggested name
    by_name: Dict[str, Dict[str, str]] = {}
    for r in manifest_rows:
        name = r.get("suggested_batchinput_name") or ""
        if not name:
            raise ValueError("manifest row missing suggested_batchinput_name")
        by_name[name] = r

    # Determine processing order
    order: List[str]
    if a.run_order:
        order = [ln.strip() for ln in Path(a.run_order).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        order = [r["suggested_batchinput_name"] for r in manifest_rows]

    if a.limit_parts and a.limit_parts > 0:
        order = order[: a.limit_parts]

    total_parts = 0
    total_reqs = 0

    for part_name in order:
        r = by_name.get(part_name)
        if not r:
            raise ValueError(f"run_order references unknown part: {part_name}")

        domain = r["domain"]
        src_path = Path(r["src_path"])
        start = int(r["start_line"])
        end = int(r["end_line"])
        part_index = int(r["part_index"])

        out_path = out_dir / part_name
        if out_path.exists() and not a.overwrite:
            raise FileExistsError(f"refusing to overwrite existing {out_path} (use --overwrite)")
        if out_path.exists() and a.overwrite:
            out_path.unlink()

        n_written = 0
        with out_path.open("w", encoding="utf-8") as out_f:
            for line_idx, obj in iter_jsonl_range(src_path, start, end):
                cid = cluster_id_of(obj) or f"{domain}:{part_index}:{line_idx}"
                dom = domain_of(obj, domain)
                topic, subtopic = topic_of(obj)
                facts = extract_facts(obj, a.max_facts, a.max_fact_chars)
                user_prompt = clamp_prompt(make_user_prompt(cid, dom, topic, subtopic, facts), a.max_prompt_chars)

                # keep custom_id short + deterministic
                short = (cid.replace(":", "_").replace("/", "_"))[:24]
                custom_id = f"{a.mode_tag}.{dom}.p{part_index:04d}.l{line_idx:06d}.{short}"

                body = {
                    "model": a.model,
                    "input": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema_doc["name"],
                            "schema": schema_doc["schema"],
                        }
                    },
                    "max_output_tokens": a.max_output_tokens,
                }

                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": a.url,
                    "body": body,
                }
                out_f.write(json.dumps(req, ensure_ascii=False) + "\n")
                n_written += 1

                if a.max_requests_per_part and n_written >= a.max_requests_per_part:
                    break

        total_parts += 1
        total_reqs += n_written
        print(f"WROTE {out_path} requests={n_written} src={src_path} range=[{start},{end})")

    print(f"DONE parts={total_parts} total_requests={total_reqs} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

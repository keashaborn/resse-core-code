#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

from openai import OpenAI


def slurp(p: str | Path) -> str:
    return Path(p).read_text(encoding="utf-8")

def _get(o, k: str):
    if o is None:
        return None
    if isinstance(o, dict):
        return o.get(k)
    return getattr(o, k, None)

def load_json(p: str | Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def select_facts(cluster: dict, k: int) -> List[Tuple[int, str]]:
    """
    Mirrors the PassC pipeline convention: choose up to k facts.
    Prefer seed text + top-scoring members; fall back to first k member texts.
    """
    members = cluster.get("members") or []
    out: List[Tuple[int, str]] = []

    # seed text first if present
    seed_txt = ((cluster.get("seed_payload") or {}).get("text") or "").strip()
    if seed_txt:
        out.append((0, seed_txt))

    # collect member texts, keep stable order but prefer higher score if present
    scored = []
    for m in members:
        if not isinstance(m, dict):
            continue
        t = (m.get("text") or "").strip()
        if not t:
            continue
        scored.append((float(m.get("score", 0.0)), t, m))

    # sort desc score, stable tie by appearance
    scored.sort(key=lambda x: x[0], reverse=True)

    for _, t, _m in scored:
        if len(out) >= k:
            break
        # avoid duplicate exact text
        if any(t == tt for _, tt in out):
            continue
        out.append((len(out), t))

    # if still short, add remaining raw member texts in original order
    if len(out) < k:
        for m in members:
            if not isinstance(m, dict):
                continue
            t = (m.get("text") or "").strip()
            if not t:
                continue
            if any(t == tt for _, tt in out):
                continue
            out.append((len(out), t))
            if len(out) >= k:
                break

    return out[:k]


def make_user_text(cluster_id: str, domain: str, topic: str, subtopic: str, facts: List[Tuple[int, str]]) -> str:
    lines = []
    lines.append(f"cluster_id: {cluster_id}")
    if domain:
        lines.append(f"domain: {domain}")
    if topic:
        lines.append(f"topic: {topic}")
    if subtopic:
        lines.append(f"subtopic: {subtopic}")
    lines.append("facts:")
    for i, t in facts:
        # PassC schema uses 0..k-1 indices; your prompt shows 1..K.
        # We keep the human-friendly 1..K numbering here, but the model output is still 0-based indices.
        lines.append(f"{i+1}. {t}")
    return "\n".join(lines)


def build_request(
    *,
    custom_id: str,
    model: str,
    sys_prompt: str,
    schema_doc: dict,
    user_text: str,
    max_output_tokens: int,
) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": "low"},
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_doc["name"],
                    "schema": schema_doc["schema"],
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
        },
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_dir", required=True, help="dir of domain.jsonl shards")
    ap.add_argument("--domains", required=True, help="comma-separated domain list")
    ap.add_argument("--n_per_domain", type=int, default=2000)
    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--system_prompt_file", required=True)
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max_output_tokens", type=int, default=2000)

    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/batch_runs")
    ap.add_argument("--run_tag", default="passC_batch")
    ap.add_argument("--completion_window", default="24h")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{a.run_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = slurp(a.system_prompt_file)
    schema_doc = load_json(a.schema_file)

    input_jsonl = run_dir / "batchinput.jsonl"
    map_jsonl = run_dir / "map.jsonl"

    doms = [d.strip() for d in a.domains.split(",") if d.strip()]
    total_req = 0

    with input_jsonl.open("w", encoding="utf-8") as out_f, map_jsonl.open("w", encoding="utf-8") as map_f:
        for dom in doms:
            src = Path(a.clusters_dir) / f"{dom}.jsonl"
            if not src.exists():
                raise FileNotFoundError(str(src))

            for idx, cluster in enumerate(iter_jsonl(src)):
                if idx >= a.n_per_domain:
                    break

                cluster_id = (cluster.get("seed_fact_id") or "").strip()
                payload = cluster.get("seed_payload") or {}
                topic = (payload.get("topic") or "").strip()
                subtopic = (payload.get("subtopic") or "").strip()

                facts = select_facts(cluster, k=a.k)
                user_text = make_user_text(cluster_id, dom, topic, subtopic, facts)

                # custom_id: stable, short, unique
                custom_id = f"passC.{dom}.{idx:07d}.{cluster_id[:20]}"

                req = build_request(
                    custom_id=custom_id,
                    model=a.model,
                    sys_prompt=sys_prompt,
                    schema_doc=schema_doc,
                    user_text=user_text,
                    max_output_tokens=a.max_output_tokens,
                )
                out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

                # sidecar map for collector validation + provenance (best-effort)
                members = cluster.get("members") or []
                # map facts by exact text back to member meta (first match)
                buckets: Dict[str, List[Tuple[Any, Any]]] = {}
                for m in members:
                    if not isinstance(m, dict):
                        continue
                    t = (m.get("text") or "").strip()
                    if not t:
                        continue
                    buckets.setdefault(t, []).append((m.get("fact_id"), m.get("dup_group_id")))

                facts_for_log = []
                for i, txt in facts:
                    lst = buckets.get(txt) or []
                    fid, dg = (lst.pop(0) if lst else (None, None))
                    facts_for_log.append({"i": i, "text": txt, "fact_id": fid, "dup_group_id": dg})

                map_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "domain": dom,
                            "cluster_id": cluster_id,
                            "facts": facts_for_log,
                            "k": a.k,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_req += 1

    print("WROTE_RUN_DIR", str(run_dir))
    print("input_jsonl", str(input_jsonl))
    print("map_jsonl", str(map_jsonl))
    print("requests", total_req)

    if a.dry_run:
        print("DRY_RUN: not uploading/creating batch")
        return

    client = OpenAI()
    # upload file for batch
    with input_jsonl.open("rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    input_file_id = getattr(up, "id", None)
    print("UPLOADED input_file_id", input_file_id)

    input_file_id = getattr(up, "id", None)
    print("UPLOADED input_file_id", input_file_id)
    if not input_file_id:
        raise RuntimeError("File upload returned no id")


    b = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window=a.completion_window,
        metadata={"run_dir": str(run_dir), "run_tag": a.run_tag},
    )
    batch_id = getattr(b, "id", None)
    if not batch_id:
      raise RuntimeError("Batch create returned no id")
    (run_dir / "batch_id.txt").write_text(batch_id + "\n", encoding="utf-8")
    print("CREATED batch_id", batch_id)
    print("batch_id_file", str(run_dir / "batch_id.txt"))


if __name__ == "__main__":
    main()

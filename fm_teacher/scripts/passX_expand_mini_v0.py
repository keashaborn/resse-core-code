#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI

QDRANT = "http://127.0.0.1:6333"
COLL = "sc_fractal_field_facts_v1"
EMB_MODEL = "text-embedding-3-large"

# Optional coverage ledger (only used if RFF_LEDGER_DB is set)
# Load by file path so it works regardless of how this script is invoked.
import importlib.util as _importlib_util


def _load_ledger_module():
    led_path = Path(__file__).resolve().with_name("rff_coverage_sqlite_v0.py")
    if not led_path.exists():
        return None
    spec = _importlib_util.spec_from_file_location("rff_coverage_sqlite_v0", str(led_path))
    if not spec or not spec.loader:
        return None
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


led = _load_ledger_module()


def _load_allowlist(path: str) -> Set[str]:
    p = Path(path)
    doms: Set[str] = set()
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            doms.add(ln)
    return doms


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def write_jsonl_append(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def qdrant_search(vec: List[float], domain: str, limit: int) -> List[dict]:
    body = {
        "vector": vec,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must": [{"key": "domain", "match": {"value": domain}}]},
    }
    out = subprocess.check_output(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{QDRANT}/collections/{COLL}/points/search",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(body),
        ],
        text=True,
    )
    data = json.loads(out)
    return (data.get("result") or [])


def _ledger_expansion_stats(conn) -> Dict[str, Tuple[int, int]]:
    """
    Returns {concept_id: (n_expansions, last_seen_ts)} from the ledger.
    """
    try:
        cur = conn.execute("SELECT concept_id, n_expansions, last_seen_ts FROM expansions;")
        return {str(cid): (int(n), int(ts)) for (cid, n, ts) in cur.fetchall()}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True, help="Merged field dir containing concept_nodes.jsonl + retrieval_seeds.jsonl")
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/runs")
    ap.add_argument("--n_concepts", type=int, default=100)
    ap.add_argument("--queries_per_concept", type=int, default=3)
    ap.add_argument("--hits_per_query", type=int, default=15)
    ap.add_argument("--max_members", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    # Optional domain allowlist (restrict sampling universe)
    # Must be exported by the caller if using run_field_iter_override.sh
    allowlist_file = os.environ.get("PASSX_DOMAIN_ALLOWLIST", "").strip()
    allow_doms = None
    if allowlist_file:
        allow_doms = _load_allowlist(allowlist_file)
        print(f"PASSX_DOMAIN_ALLOWLIST loaded n={len(allow_doms)} path={allowlist_file}")

    # Optional coverage ledger (coverage-first + fairness sampling)
    ledger_db = os.environ.get("RFF_LEDGER_DB", "").strip()
    led_ok = (
        led is not None
        and all(hasattr(led, fn) for fn in ("open_db", "get_seen_set", "mark_expanded", "get_expansion_stats"))
    )
    use_ledger = bool(ledger_db) and led_ok

    conn = None
    seen: Set[str] = set()

    if use_ledger:
        conn = led.open_db(ledger_db)  # type: ignore[attr-defined]
        seen = led.get_seen_set(conn)  # type: ignore[attr-defined]
        print(f"RFF_LEDGER enabled db={ledger_db} seen={len(seen)}")

    field_dir = Path(args.field_dir)
    nodes_path = field_dir / "concept_nodes.jsonl"
    seeds_path = field_dir / "retrieval_seeds.jsonl"
    if not nodes_path.exists() or not seeds_path.exists():
        raise SystemExit("field_dir must contain concept_nodes.jsonl and retrieval_seeds.jsonl")

    # concept_id -> canonical_text, domain
    concept: Dict[str, dict] = {}
    for n in iter_jsonl(nodes_path):
        concept[n["concept_id"]] = {"canonical_text": n["canonical_text"], "domain": n["domain"]}

    # read seeds
    seed_rows = [
        r for r in iter_jsonl(seeds_path)
        if r.get("concept_id") in concept and r.get("retrieval_queries")
    ]
    if not seed_rows:
        raise SystemExit("No usable retrieval_seeds rows found.")

    # sample concepts (dedupe by concept_id)
    by_cid: Dict[str, dict] = {}
    for r in seed_rows:
        cid = r["concept_id"]
        if cid not in by_cid:
            by_cid[cid] = r

    cids_all = list(by_cid.keys())
    random.shuffle(cids_all)

    # apply allowlist AFTER we have concept map (domain lives there)
    if allow_doms is not None:
        before = len(cids_all)
        cids_all = [cid for cid in cids_all if concept.get(cid, {}).get("domain") in allow_doms]
        after = len(cids_all)
        print(f"PASSX_DOMAIN_ALLOWLIST filtered concepts {before}->{after}")

    # Choose cids
    if use_ledger and conn is not None:
        exp_stats = led.get_expansion_stats(conn)  # type: ignore[attr-defined]
        # exp_stats: {cid: (n_expansions, last_seen_ts)}
        scored = []
        for cid in cids_all:
            n, ts = exp_stats.get(cid, (0, 0))  # unseen => (0,0)
            # sort: unseen first, then fewer expansions, then older last_seen, then random
            scored.append((n, ts, random.random(), cid))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        cids = [cid for (_n, _ts, _r, cid) in scored[: args.n_concepts]]

        unseen_used = sum(1 for cid in cids if exp_stats.get(cid, (0, 0))[0] == 0)
        min_n = min((exp_stats.get(cid, (0, 0))[0] for cid in cids), default=0)
        max_n = max((exp_stats.get(cid, (0, 0))[0] for cid in cids), default=0)
        min_ts = min((exp_stats.get(cid, (0, 0))[1] for cid in cids), default=0)
        max_ts = max((exp_stats.get(cid, (0, 0))[1] for cid in cids), default=0)
        print(f"RFF_LEDGER sampling picked={len(cids)} unseen_used={unseen_used} min_n={min_n} max_n={max_n} min_ts={min_ts} max_ts={max_ts}")
    else:
        cids = cids_all[: args.n_concepts]

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"passX_mini_v0_{ts}"
    clusters_dir = run_dir / "clusters_by_domain_passX"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(timeout=90.0)

    made = 0
    expanded_n = 0

    emb_model = os.environ.get("PASSX_EMB_MODEL", EMB_MODEL)
    fallback_model = os.environ.get("PASSX_EMB_FALLBACK", "text-embedding-3-small")

    def _embed(model_name: str, inputs: List[str]):
        return client.embeddings.create(model=model_name, input=inputs)

    for cid in cids:
        info = concept.get(cid)
        if not info:
            continue
        dom = info["domain"]
        canon_text = info["canonical_text"]
        rq = (by_cid.get(cid, {}).get("retrieval_queries") or [])[: args.queries_per_concept]
        if not rq:
            continue

        # embed queries (resilient): fall back on 403 instead of killing the iteration
        try:
            emb = _embed(emb_model, rq)
        except Exception as e:
            code = getattr(e, "status_code", None)
            msg = str(e).lower()
            if code == 403 or "permission" in msg:
                try:
                    emb = _embed(fallback_model, rq)
                    print(
                        f"PASSX_EMB_FALLBACK_OK concept_id={cid} domain={dom} "
                        f"primary={emb_model} fallback={fallback_model}"
                    )
                except Exception as e2:
                    code2 = getattr(e2, "status_code", None)
                    print(
                        f"PASSX_EMB_SKIP concept_id={cid} domain={dom} "
                        f"primary={emb_model} fallback={fallback_model} "
                        f"err2={type(e2).__name__} code2={code2}"
                    )
                    continue
            else:
                transient = code in (429, 500, 502, 503, 504) or any(
                    s in msg for s in ("timeout", "timed out", "readtimeout", "overloaded", "rate limit", "connection")
                )
                if transient:
                    ok = False
                    for attempt in range(1, 7):
                        time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
                        try:
                            emb = _embed(emb_model, rq)
                            ok = True
                            break
                        except Exception:
                            continue
                    if not ok:
                        print(f"PASSX_EMB_TRANSIENT_GIVEUP concept_id={cid} domain={dom} model={emb_model} rq_n={len(rq)}")
                        continue
                else:
                    raise

        vecs = [d.embedding for d in emb.data]

        # search each query, dedupe by fact_id
        seen_fact = set()
        members = []
        for v in vecs:
            hits = qdrant_search(v, dom, args.hits_per_query)
            for h in hits:
                payload = h.get("payload") or {}
                fid = payload.get("fact_id") or h.get("id")
                if not fid or fid in seen_fact:
                    continue
                txt = payload.get("text") or ""
                if not txt:
                    continue
                seen_fact.add(fid)
                members.append(
                    {
                        "text": txt,
                        "fact_id": fid,
                        "dup_group_id": payload.get("dup_group_id") or payload.get("fingerprint"),
                    }
                )
                if len(members) >= args.max_members:
                    break
            if len(members) >= args.max_members:
                break

        # Mark ledger as "expanded" once we've actually attempted retrieval for this concept.
        if use_ledger and conn is not None:
            led.mark_expanded(conn, cid, dom, str(field_dir))  # type: ignore[attr-defined]
            expanded_n += 1
            if expanded_n % 200 == 0:
                conn.commit()

        # write cluster record into domain file
        if members:
            out_path = clusters_dir / f"{dom}.jsonl"
            rec = {
                "seed_fact_id": f"passx::{cid}",
                "seed_payload": {"text": canon_text},
                "members": members,
                "meta": {
                    "source_field_dir": str(field_dir),
                    "concept_id": cid,
                    "domain": dom,
                    "retrieval_queries": rq,
                },
            }
            write_jsonl_append(out_path, rec)
            made += 1

    if use_ledger and conn is not None:
        conn.commit()
        conn.close()

    print("WROTE_RUN_DIR", str(run_dir))
    print("CLUSTERS_DIR", str(clusters_dir))
    print("clusters_written", made)
    print("\nNext step (run PassC on expansion clusters):")
    print(
        f'python3 /home/ubuntu/resse-core/fm_teacher/scripts/eval_passC_prompt_v0.py '
        f'--clusters_dir "{clusters_dir}" '
        f'--domains "$(ls -1 {clusters_dir}/*.jsonl | sed \'s#.*/##; s#\\.jsonl$##\' | paste -sd, -)" '
        f'--n_per_domain 999999 '
        f'--system_prompt_file /home/ubuntu/resse-core/fm_teacher/prompts/passC_system_v0.txt '
        f'--schema_file /home/ubuntu/resse-core/fm_teacher/schemas/passC_schema_v0.json'
    )


if __name__ == "__main__":
    main()

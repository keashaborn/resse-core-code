#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, time, hashlib
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

WS = re.compile(r"\s+")

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = WS.sub(" ", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def find_latest_passc_eval_dir(evals_root: Path) -> Path:
    dirs = sorted([p for p in evals_root.glob("passC_prompt_v0_*") if p.is_dir()], reverse=True)
    if not dirs:
        raise SystemExit(f"No passC_prompt_v0_* dirs under {evals_root}")
    return dirs[0]

def member_id(domain: str, text: str) -> str:
    return sha1(f"MEMBER||{domain}||{norm_text(text)}")

def concept_id(domain: str, canonical_text: str) -> str:
    return sha1(f"CONCEPT||{domain}||{norm_text(canonical_text)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passc_eval_dir", default="", help="Path to passC eval dir containing ok.jsonl")
    ap.add_argument("--evals_root", default="/home/ubuntu/resse-core/fm_teacher/evals")
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/field")
    ap.add_argument("--run_tag", default="passM_v1")
    args = ap.parse_args()

    evals_root = Path(args.evals_root)
    passc_dir = Path(args.passc_eval_dir) if args.passc_eval_dir else find_latest_passc_eval_dir(evals_root)
    ok_path = passc_dir / "ok.jsonl"
    if not ok_path.exists():
        raise SystemExit(f"Missing ok.jsonl in {passc_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_out = out_dir / "concept_nodes.jsonl"
    members_out = out_dir / "concept_members.jsonl"
    edges_out = out_dir / "concept_edges.jsonl"
    seeds_out = out_dir / "retrieval_seeds.jsonl"
    summary_out = out_dir / "summary.md"

    rel_counts = Counter()
    role_counts = Counter()
    keep_len_counts = Counter()
    drop_reason_counts = Counter()

    # De-dupe nodes by concept_id; merge provenance lists
    node_by_id: Dict[str, dict] = {}

    member_rows: List[dict] = []
    edge_rows: List[dict] = []
    seed_rows: List[dict] = []

    for rec in iter_jsonl(ok_path):
        domain = rec.get("domain") or ""
        cluster_id = rec.get("cluster_id") or ""
        obj = rec.get("obj") or {}
        facts_list = rec.get("facts") or []
        facts = {f.get("i"): f.get("text") for f in facts_list if isinstance(f, dict) and "i" in f and "text" in f}

        keep = obj.get("keep_fact_i") or []
        canonical_i = obj.get("canonical_i")
        roles_list = obj.get("fact_roles") or []
        roles = {it.get("i"): it.get("role") for it in roles_list if isinstance(it, dict) and "i" in it and "role" in it}

        if not isinstance(canonical_i, int) or canonical_i not in facts:
            continue

        canonical_text = facts.get(canonical_i, "") or ""
        cid = concept_id(domain, canonical_text)

        keep_len_counts[len(keep)] += 1
        for it in (obj.get("drop_facts") or []):
            if isinstance(it, dict) and it.get("reason"):
                drop_reason_counts[it["reason"]] += 1

        if cid not in node_by_id:
            node_by_id[cid] = {
                "concept_id": cid,
                "domain": domain,
                "canonical_text": canonical_text,
                "canonical_member_id": member_id(domain, canonical_text),
                "source_cluster_ids": [cluster_id],
            }
        else:
            if cluster_id not in node_by_id[cid]["source_cluster_ids"]:
                node_by_id[cid]["source_cluster_ids"].append(cluster_id)

        # Build local map from cluster index -> member_id for kept facts
        idx_to_mid: Dict[int, str] = {}
        for i in keep:
            if not isinstance(i, int):
                continue
            txt = facts.get(i, "")
            if not txt:
                continue
            idx_to_mid[i] = member_id(domain, txt)

        # Members (stable member_id)
        for i in keep:
            if not isinstance(i, int):
                continue
            txt = facts.get(i, "")
            if not txt:
                continue
            mid = idx_to_mid.get(i)
            role = roles.get(i, "unknown")
            role_counts[role] += 1
            member_rows.append({
                "concept_id": cid,
                "cluster_id": cluster_id,
                "domain": domain,
                "member_id": mid,
                "is_canonical": (i == canonical_i),
                "role": role,
                "text": txt,
                "fact_i": i,  # cluster-local index for debugging only
            })

        # Edges (write stable endpoints + keep local indices for debugging)
        for e in (obj.get("edge_candidates") or []):
            if not isinstance(e, dict):
                continue
            rt = e.get("rel_type")
            si = e.get("src_i")
            di = e.get("dst_i")
            sup = e.get("support_i_list") or []
            if rt:
                rel_counts[rt] += 1

            src_txt = facts.get(si, "") if isinstance(si, int) else ""
            dst_txt = facts.get(di, "") if isinstance(di, int) else ""

            src_mid = idx_to_mid.get(si) if isinstance(si, int) else None
            dst_mid = idx_to_mid.get(di) if isinstance(di, int) else None
            sup_mids = []
            if isinstance(sup, list):
                for x in sup:
                    if isinstance(x, int) and x in idx_to_mid:
                        sup_mids.append(idx_to_mid[x])

            edge_rows.append({
                "concept_id": cid,
                "cluster_id": cluster_id,
                "domain": domain,
                "rel_type": rt,
                "src_member_id": src_mid,
                "dst_member_id": dst_mid,
                "support_member_ids": sup_mids,
                # debug fields
                "src_i": si,
                "dst_i": di,
                "support_i_list": sup,
                "src_text": src_txt,
                "dst_text": dst_txt,
            })

        # Retrieval seeds
        rq = obj.get("retrieval_queries") or []
        if rq:
            seed_rows.append({
                "concept_id": cid,
                "cluster_id": cluster_id,
                "domain": domain,
                "retrieval_queries": rq,
            })

    node_rows = list(node_by_id.values())

    n_nodes = write_jsonl(nodes_out, node_rows)
    n_members = write_jsonl(members_out, member_rows)
    n_edges = write_jsonl(edges_out, edge_rows)
    n_seeds = write_jsonl(seeds_out, seed_rows)

    with summary_out.open("w", encoding="utf-8") as f:
        f.write("# PassM materialization summary (v1)\n\n")
        f.write(f"- passC_source_dir: {str(passc_dir)}\n")
        f.write(f"- out_dir: {str(out_dir)}\n\n")
        f.write("## Counts\n\n")
        f.write(f"- concept_nodes: {n_nodes}\n")
        f.write(f"- concept_members: {n_members}\n")
        f.write(f"- concept_edges: {n_edges}\n")
        f.write(f"- retrieval_seeds: {n_seeds}\n\n")
        f.write("## keep_fact_i size distribution\n\n")
        f.write(json.dumps(dict(sorted(keep_len_counts.items())), indent=2) + "\n\n")
        f.write("## role distribution\n\n")
        f.write(json.dumps(dict(role_counts.most_common()), indent=2) + "\n\n")
        f.write("## rel_type distribution\n\n")
        f.write(json.dumps(dict(rel_counts.most_common()), indent=2) + "\n\n")
        f.write("## drop reason distribution\n\n")
        f.write(json.dumps(dict(drop_reason_counts.most_common()), indent=2) + "\n\n")

    print("PASSC_DIR", str(passc_dir))
    print("WROTE", str(out_dir))
    print("nodes", n_nodes, "members", n_members, "edges", n_edges, "seeds", n_seeds)
    print("summary", str(summary_out))

if __name__ == "__main__":
    main()

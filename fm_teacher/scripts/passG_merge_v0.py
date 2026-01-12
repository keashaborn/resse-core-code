#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Tuple, Any

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

def load_all(path: Path) -> List[dict]:
    return list(iter_jsonl(path)) if path.exists() else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--cross_edges", required=True)  # concept_edges_cross.jsonl
    ap.add_argument("--cross_edges_scored", default="", help="Optional scored cross edges file (concept_edges_cross_scored.jsonl). If set, only keep keep==true.")
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/field")
    ap.add_argument("--run_tag", default="passG_v0")
    args = ap.parse_args()

    base = Path(args.base_dir)
    exp  = Path(args.exp_dir)
    cross = Path(args.cross_edges)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    b_nodes = load_all(base/"concept_nodes.jsonl")
    e_nodes = load_all(exp/"concept_nodes.jsonl")
    b_mems  = load_all(base/"concept_members.jsonl")
    e_mems  = load_all(exp/"concept_members.jsonl")
    b_edges = load_all(base/"concept_edges.jsonl")
    e_edges = load_all(exp/"concept_edges.jsonl")
    b_seeds = load_all(base/"retrieval_seeds.jsonl")
    e_seeds = load_all(exp/"retrieval_seeds.jsonl")
    b_alias = load_all(base/"concept_aliases.jsonl")
    e_alias = load_all(exp/"concept_aliases.jsonl")
    x_edges = load_all(cross)
    # If a scored cross-edges file is provided, use it instead and filter keep==true
    if args.cross_edges_scored:
        scored = Path(args.cross_edges_scored)
        x_edges = load_all(scored)
        x_edges = [e for e in x_edges if (e.get("keep") is True)]

    # Merge nodes by concept_id (union provenance)
    node_by: Dict[str, dict] = {}
    def upsert_node(n: dict):
        cid = n.get("concept_id")
        if not cid:
            return
        if cid not in node_by:
            node_by[cid] = dict(n)
            # normalize list field
            if not isinstance(node_by[cid].get("source_cluster_ids"), list):
                node_by[cid]["source_cluster_ids"] = []
        else:
            # merge source_cluster_ids
            sc = node_by[cid].get("source_cluster_ids") or []
            add = n.get("source_cluster_ids") or []
            if isinstance(sc, list) and isinstance(add, list):
                sset = set(sc)
                for x in add:
                    if x not in sset:
                        sc.append(x); sset.add(x)
                node_by[cid]["source_cluster_ids"] = sc

    for n in b_nodes: upsert_node(n)
    for n in e_nodes: upsert_node(n)

    # Merge members (dedupe by (concept_id, member_id, role, text))
    mem_seen = set()
    mem_out = []
    for m in b_mems + e_mems:
        cid = m.get("concept_id"); mid = m.get("member_id"); role = m.get("role"); txt = m.get("text")
        key = (cid, mid, role, txt)
        if key in mem_seen: 
            continue
        mem_seen.add(key)
        mem_out.append(m)

    # Merge intra-concept edges (dedupe by key)
    edge_seen = set()
    edge_out = []
    for e in b_edges + e_edges:
        key = (
            e.get("concept_id"),
            e.get("rel_type"),
            e.get("src_member_id"),
            e.get("dst_member_id"),
            tuple(e.get("support_member_ids") or []),
        )
        if key in edge_seen:
            continue
        edge_seen.add(key)
        edge_out.append(e)

    # Merge retrieval seeds (dedupe by (concept_id, cluster_id, domain, tuple(queries)))
    seed_seen = set()
    seed_out = []
    for s in b_seeds + e_seeds:
        key = (s.get("concept_id"), s.get("cluster_id"), s.get("domain"), tuple(s.get("retrieval_queries") or []))
        if key in seed_seen:
            continue
        seed_seen.add(key)
        seed_out.append(s)

    # Merge aliases (dedupe)
    alias_seen = set()
    alias_out = []
    for a in b_alias + e_alias:
        key = (a.get("concept_id"), a.get("alias_member_id"), a.get("rep_member_id"))
        if key in alias_seen:
            continue
        alias_seen.add(key)
        alias_out.append(a)

    # Cross edges: drop self-loops, dedupe
    x_seen = set()
    x_out = []
    self_loops = 0
    for x in x_edges:
        s = x.get("src_concept_id"); d = x.get("dst_concept_id")
        if not s or not d:
            continue
        if s == d:
            self_loops += 1
            continue
        key = (x.get("rel_type"), s, d, tuple(x.get("retrieval_queries") or []))
        if key in x_seen:
            continue
        x_seen.add(key)
        x_out.append(x)

    # Write
    n_nodes = write_jsonl(out_dir/"concept_nodes.jsonl", node_by.values())
    n_mems  = write_jsonl(out_dir/"concept_members.jsonl", mem_out)
    n_edges = write_jsonl(out_dir/"concept_edges.jsonl", edge_out)
    n_seeds = write_jsonl(out_dir/"retrieval_seeds.jsonl", seed_out)
    n_alias = write_jsonl(out_dir/"concept_aliases.jsonl", alias_out)
    n_x     = write_jsonl(out_dir/"concept_edges_cross.jsonl", x_out)

    # Summary
    rel_counts = Counter(e.get("rel_type") for e in edge_out)
    summary = out_dir/"summary.md"
    summary.write_text(
        "# PassG-lite merge summary\n\n"
        f"- base_dir: {base}\n"
        f"- exp_dir: {exp}\n"
        f"- cross_edges_in: {cross}\n"
        f"- out_dir: {out_dir}\n\n"
        "## Counts\n\n"
        f"- concept_nodes: {n_nodes}\n"
        f"- concept_members: {n_mems}\n"
        f"- concept_edges: {n_edges}\n"
        f"- retrieval_seeds: {n_seeds}\n"
        f"- concept_aliases: {n_alias}\n"
        f"- concept_edges_cross: {n_x}\n"
        f"- cross_self_loops_dropped: {self_loops}\n\n"
        "## rel_type distribution (intra-concept edges)\n\n"
        + json.dumps(dict(rel_counts.most_common()), indent=2) + "\n",
        encoding="utf-8"
    )

    print("WROTE", str(out_dir))
    print("concept_nodes", n_nodes, "members", n_mems, "edges", n_edges, "seeds", n_seeds, "aliases", n_alias, "cross_edges", n_x, "self_loops_dropped", self_loops)
    print("summary", str(summary))

if __name__ == "__main__":
    main()

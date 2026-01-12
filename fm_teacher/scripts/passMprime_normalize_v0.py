#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Iterable, Tuple, Set

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/field")
    ap.add_argument("--run_tag", default="passMprime_v0")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = list(iter_jsonl(in_dir/"concept_nodes.jsonl"))
    members = list(iter_jsonl(in_dir/"concept_members.jsonl"))
    edges = list(iter_jsonl(in_dir/"concept_edges.jsonl"))
    seeds = list(iter_jsonl(in_dir/"retrieval_seeds.jsonl"))

    # Group members and edges by concept_id
    mem_by_c: Dict[str, List[dict]] = defaultdict(list)
    for m in members:
        mem_by_c[m["concept_id"]].append(m)

    edge_by_c: Dict[str, List[dict]] = defaultdict(list)
    for e in edges:
        edge_by_c[e["concept_id"]].append(e)

    # Build alias mapping per concept from same_as edges
    # Union-find over member_ids
    def uf_make(ids: Set[str]):
        parent = {x: x for x in ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        return parent, find, union

    alias_rows = []
    new_members = []
    new_edges = []
    removed_same_as = 0
    dedup_edges = 0

    rel_counts_before = Counter(e.get("rel_type") for e in edges)
    rel_counts_after = Counter()

    for n in nodes:
        cid = n["concept_id"]
        ms = mem_by_c.get(cid, [])
        es = edge_by_c.get(cid, [])

        # collect member_ids for this concept
        mids = {m["member_id"] for m in ms if m.get("member_id")}
        if not mids:
            continue

        parent, find, union = uf_make(mids)

        # union member_ids connected by same_as
        for e in es:
            if e.get("rel_type") != "same_as":
                continue
            a = e.get("src_member_id")
            b = e.get("dst_member_id")
            if a in mids and b in mids:
                union(a,b)

        # group by root
        groups: Dict[str, List[str]] = defaultdict(list)
        for mid in mids:
            groups[find(mid)].append(mid)

        # choose representative for each group
        # preference: canonical_member_id if present in group, else smallest member_id
        canon_mid = n.get("canonical_member_id")
        rep_for: Dict[str, str] = {}
        for root, members_in_group in groups.items():
            if canon_mid in members_in_group:
                rep = canon_mid
            else:
                rep = sorted(members_in_group)[0]
            for m_id in members_in_group:
                rep_for[m_id] = rep

        # write alias map rows (only for non-reps)
        for root, members_in_group in groups.items():
            rep = rep_for[members_in_group[0]]
            for m_id in members_in_group:
                if m_id != rep:
                    alias_rows.append({
                        "concept_id": cid,
                        "alias_member_id": m_id,
                        "rep_member_id": rep,
                    })

        # rewrite members: add alias_of field if not rep
        for m in ms:
            m = dict(m)
            mid = m.get("member_id")
            rep = rep_for.get(mid, mid)
            if mid != rep:
                m["alias_of"] = rep
            new_members.append(m)

        # rewrite edges to use reps; drop same_as edges
        seen_edge = set()
        for e in es:
            rt = e.get("rel_type")
            if rt == "same_as":
                removed_same_as += 1
                continue

            e2 = dict(e)
            sm = e2.get("src_member_id")
            dm = e2.get("dst_member_id")
            if sm in rep_for: e2["src_member_id"] = rep_for[sm]
            if dm in rep_for: e2["dst_member_id"] = rep_for[dm]

            sup = e2.get("support_member_ids") or []
            if isinstance(sup, list):
                e2["support_member_ids"] = [rep_for.get(x, x) for x in sup]

            # dedupe key
            key = (cid, e2.get("rel_type"), e2.get("src_member_id"), e2.get("dst_member_id"), tuple(e2.get("support_member_ids") or []))
            if key in seen_edge:
                dedup_edges += 1
                continue
            seen_edge.add(key)

            new_edges.append(e2)
            rel_counts_after[e2.get("rel_type")] += 1

    # write outputs
    out_nodes = out_dir/"concept_nodes.jsonl"
    out_members = out_dir/"concept_members.jsonl"
    out_edges = out_dir/"concept_edges.jsonl"
    out_seeds = out_dir/"retrieval_seeds.jsonl"
    out_aliases = out_dir/"concept_aliases.jsonl"
    out_summary = out_dir/"summary.md"

    n_nodes = write_jsonl(out_nodes, nodes)
    n_members = write_jsonl(out_members, new_members)
    n_edges = write_jsonl(out_edges, new_edges)
    n_seeds = write_jsonl(out_seeds, seeds)
    n_alias = write_jsonl(out_aliases, alias_rows)

    with out_summary.open("w", encoding="utf-8") as f:
        f.write("# PassM′ normalization summary\n\n")
        f.write(f"- in_dir: {str(in_dir)}\n")
        f.write(f"- out_dir: {str(out_dir)}\n\n")
        f.write("## Counts\n\n")
        f.write(f"- nodes: {n_nodes}\n")
        f.write(f"- members: {n_members}\n")
        f.write(f"- edges: {n_edges}\n")
        f.write(f"- seeds: {n_seeds}\n")
        f.write(f"- aliases: {n_alias}\n\n")
        f.write("## Edge rel_type counts (before)\n\n")
        f.write(json.dumps(dict(rel_counts_before.most_common()), indent=2) + "\n\n")
        f.write("## Edge rel_type counts (after)\n\n")
        f.write(json.dumps(dict(rel_counts_after.most_common()), indent=2) + "\n\n")
        f.write(f"removed_same_as_edges: {removed_same_as}\n")
        f.write(f"dedup_edges_removed: {dedup_edges}\n")

    print("IN_DIR", str(in_dir))
    print("WROTE", str(out_dir))
    print("nodes", n_nodes, "members", n_members, "edges", n_edges, "seeds", n_seeds, "aliases", n_alias)
    print("summary", str(out_summary))

if __name__ == "__main__":
    main()

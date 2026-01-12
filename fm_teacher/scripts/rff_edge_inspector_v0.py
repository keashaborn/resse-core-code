#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from collections import defaultdict, Counter

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True)
    ap.add_argument("--n_per_rel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rels", default="contradicts,depends_on,condition_for,mechanism_for,entails,refines")
    args = ap.parse_args()

    random.seed(args.seed)
    field_dir = Path(args.field_dir)

    nodes_path = field_dir / "concept_nodes.jsonl"
    members_path = field_dir / "concept_members.jsonl"
    edges_path = field_dir / "concept_edges.jsonl"

    for p in (nodes_path, members_path, edges_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    # concept_id -> node
    nodes = {}
    for r in iter_jsonl(nodes_path):
        cid = r.get("concept_id")
        if cid:
            nodes[str(cid)] = r

    # concept_id -> fact_i -> text
    facts = defaultdict(dict)
    for r in iter_jsonl(members_path):
        cid = str(r.get("concept_id") or "")
        fi = r.get("fact_i")
        txt = r.get("text")
        if cid and isinstance(fi, int) and isinstance(txt, str):
            facts[cid][fi] = txt

    want = [x.strip() for x in args.rels.split(",") if x.strip()]
    picked = {k: [] for k in want}
    rel_counts = Counter()

    # first pass: gather candidates
    candidates_by_rel = defaultdict(list)
    for e in iter_jsonl(edges_path):
        rt = e.get("rel_type")
        if not rt:
            continue
        rt = str(rt)
        rel_counts[rt] += 1
        if rt not in picked:
            continue
        cid = str(e.get("concept_id") or "")
        si = e.get("src_i")
        di = e.get("dst_i")
        if not cid or not isinstance(si, int) or not isinstance(di, int):
            continue
        candidates_by_rel[rt].append((cid, si, di))

    print("field_dir=", str(field_dir))
    print("rel_type_counts_top12=", rel_counts.most_common(12))

    for rt in want:
        cands = candidates_by_rel.get(rt, [])
        random.shuffle(cands)
        cands = cands[: args.n_per_rel]
        print(f"\n== {rt} sample_n={len(cands)}")
        for cid, si, di in cands:
            node = nodes.get(cid) or {}
            canon = (node.get("canonical_text") or "")[:120].replace("\n"," ")
            dom = node.get("domain")
            s_txt = facts.get(cid, {}).get(si, "")
            d_txt = facts.get(cid, {}).get(di, "")
            print(f"\n- concept_id={cid} domain={dom}")
            print(f"  canonical: {canon}")
            print(f"  EDGE: {si} --{rt}--> {di}")
            print(f"  src: {s_txt}")
            print(f"  dst: {d_txt}")

if __name__ == "__main__":
    main()

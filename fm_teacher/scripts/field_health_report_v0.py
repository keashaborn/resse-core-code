#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, collections
from pathlib import Path

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True)
    ap.add_argument("--sample_cross", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.field_dir)
    nodes_p = d/"concept_nodes.jsonl"
    edges_p = d/"concept_edges.jsonl"
    cross_p = d/"concept_edges_cross.jsonl"

    nodes = list(iter_jsonl(nodes_p))
    node_by = {n["concept_id"]: n for n in nodes if "concept_id" in n}
    rel = collections.Counter(e.get("rel_type") for e in iter_jsonl(edges_p))
    total_edges = sum(rel.values())

    print("FIELD_DIR", str(d))
    print("nodes", len(nodes))
    print("edges", total_edges)
    print("rel_type_distribution:")
    for k,v in rel.most_common():
        print(f"  {k:14s} {v:7d} share={v/max(1,total_edges):.3f}")

    if cross_p.exists():
        cross = list(iter_jsonl(cross_p))
        print("\ncross_edges", len(cross))
        # sample a few cross links
        random.shuffle(cross)
        for e in cross[:args.sample_cross]:
            s = e.get("src_concept_id"); t = e.get("dst_concept_id")
            if not s or not t:
                continue
            sn = node_by.get(s, {})
            tn = node_by.get(t, {})
            print("\n---")
            print("queries:", e.get("retrieval_queries", []))
            print("SRC:", sn.get("domain"), "|", (sn.get("canonical_text","")[:160] if sn else ""))
            print("DST:", tn.get("domain"), "|", (tn.get("canonical_text","")[:160] if tn else ""))
    else:
        print("\n(no concept_edges_cross.jsonl found)")

if __name__ == "__main__":
    main()

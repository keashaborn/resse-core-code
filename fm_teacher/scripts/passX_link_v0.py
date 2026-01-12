#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Dict, Iterable, List

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
    ap.add_argument("--passx_run_dir", required=True, help="passX_mini_v0_<ts> run dir containing clusters_by_domain_passX/")
    ap.add_argument("--exp_field_dir", required=True, help="PassM′ output dir for expansion slice (passMprime_passX100_rerun_...)")
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/field")
    ap.add_argument("--run_tag", default="passXlink_v0")
    args = ap.parse_args()

    run_dir = Path(args.passx_run_dir)
    clusters_dir = run_dir / "clusters_by_domain_passX"
    if not clusters_dir.exists():
        raise SystemExit(f"missing clusters_by_domain_passX under {run_dir}")

    exp_dir = Path(args.exp_field_dir)
    nodes_path = exp_dir / "concept_nodes.jsonl"
    if not nodes_path.exists():
        raise SystemExit(f"missing concept_nodes.jsonl in {exp_dir}")

    # Map seed_fact_id -> meta (retrieval_queries, domain, concept_id source)
    seed_meta: Dict[str, dict] = {}
    for p in sorted(clusters_dir.glob("*.jsonl")):
        for r in iter_jsonl(p):
            sid = r.get("seed_fact_id") or ""
            meta = r.get("meta") or {}
            seed_meta[sid] = {
                "source_concept_id": meta.get("concept_id"),
                "domain": meta.get("domain"),
                "retrieval_queries": meta.get("retrieval_queries", []),
                "source_field_dir": meta.get("source_field_dir"),
            }

    edges_cross: List[dict] = []
    for n in iter_jsonl(nodes_path):
        dst_cid = n.get("concept_id")
        dom = n.get("domain")
        for cluster_id in (n.get("source_cluster_ids") or []):
            if not isinstance(cluster_id, str):
                continue
            if not cluster_id.startswith("passx::"):
                continue
            src_cid = cluster_id.split("passx::", 1)[1]
            m = seed_meta.get(cluster_id, {})
            edges_cross.append({
                "rel_type": "retrieved_neighbor",
                "src_concept_id": src_cid,
                "dst_concept_id": dst_cid,
                "domain": dom,
                "source_cluster_id": cluster_id,
                "passx_run_dir": str(run_dir),
                "retrieval_queries": m.get("retrieval_queries", []),
            })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_edges = out_dir / "concept_edges_cross.jsonl"
    n_edges = write_jsonl(out_edges, edges_cross)

    summary = out_dir / "summary.md"
    summary.write_text(
        "# PassX-link summary\n\n"
        f"- passx_run_dir: {run_dir}\n"
        f"- exp_field_dir: {exp_dir}\n"
        f"- out_dir: {out_dir}\n\n"
        f"- cross_edges: {n_edges}\n",
        encoding="utf-8"
    )

    print("WROTE", str(out_dir))
    print("cross_edges", n_edges)
    print("edges_file", str(out_edges))
    print("summary", str(summary))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

def load_domain_order(yaml_path: Path) -> List[str]:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    domains: List[str] = []

    for g in doc.get("groups", []) or []:
        ds = g.get("domains", []) or []
        domains.extend([d for d in ds if isinstance(d, str)])

    extras = doc.get("extras_assigned_by_level", {}) or {}
    extras_norm: Dict[int, List[str]] = {}
    for k, ds in extras.items():
        try:
            lvl = int(k)
        except Exception:
            continue
        if isinstance(ds, list):
            extras_norm[lvl] = [d for d in ds if isinstance(d, str)]
    for lvl in sorted(extras_norm.keys()):
        domains.extend(extras_norm[lvl])

    seen = set()
    out: List[str] = []
    for d in domains:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out

def parse_done_marker(done_path: Path) -> int:
    wrote = 0
    for line in done_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "wrote":
                try:
                    wrote = int(v.strip())
                except Exception:
                    wrote = 0
    return wrote

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--clusters_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include_missing", action="store_true",
                    help="Include clusters_missing_v0.jsonl as a separate pseudo-domain '_missing'")
    ap.add_argument("--missing_file", default="/home/ubuntu/resse-core/sc_fractal_field/results/clusters_missing_v0.jsonl")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    clusters_dir = Path(args.clusters_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = load_domain_order(yaml_path)

    # domain -> cluster count (prefer .DONE wrote)
    dom_counts: Dict[str, int] = {}
    dom_files: Dict[str, str] = {}

    for d in domains:
        done = clusters_dir / f"{d}.DONE"
        js = clusters_dir / f"{d}.jsonl"
        if done.exists():
            dom_counts[d] = parse_done_marker(done)
        elif js.exists():
            # fallback (should be rare)
            dom_counts[d] = sum(1 for _ in js.open("r", encoding="utf-8", errors="ignore"))
        else:
            dom_counts[d] = 0

        if js.exists():
            dom_files[d] = str(js)

    # optional: include missing clusters file as its own processing unit
    if args.include_missing:
        mf = Path(args.missing_file)
        if mf.exists():
            n = sum(1 for _ in mf.open("r", encoding="utf-8", errors="ignore"))
            dom_counts["_missing"] = n
            dom_files["_missing"] = str(mf)

    # pack domains into workers (greedy bin packing by cluster count)
    items = sorted(dom_counts.items(), key=lambda x: x[1], reverse=True)
    bins: List[Dict[str, Any]] = [{"worker": i, "clusters": 0, "domains": [], "files": []} for i in range(args.workers)]

    for dom, cnt in items:
        # ignore empty domains (shouldn't happen in practice, but safe)
        if cnt <= 0:
            continue
        b = min(bins, key=lambda z: z["clusters"])
        b["domains"].append(dom)
        b["clusters"] += cnt
        if dom in dom_files:
            b["files"].append(dom_files[dom])

    # write per-worker domain lists and file lists
    for b in bins:
        i = b["worker"]
        (out_dir / f"worker_{i:02d}.domains.txt").write_text("\n".join(b["domains"]) + "\n", encoding="utf-8")
        (out_dir / f"worker_{i:02d}.files.txt").write_text("\n".join(b["files"]) + "\n", encoding="utf-8")

    manifest = {
        "yaml": str(yaml_path),
        "clusters_dir": str(clusters_dir),
        "workers": args.workers,
        "bins": bins,
        "total_clusters": sum(b["clusters"] for b in bins),
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (out_dir / "gen_shards.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # quick stdout summary
    print("out_dir", str(out_dir))
    for b in bins:
        print(f"worker_{b['worker']:02d} clusters={b['clusters']} domains={len(b['domains'])} files={len(b['files'])}")
    print("total_clusters", manifest["total_clusters"])

if __name__ == "__main__":
    main()

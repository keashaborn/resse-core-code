#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

# Load ledger module by file path so this script works regardless of CWD.
import importlib.util as _importlib_util

def _load_ledger_module():
    led_path = Path(__file__).resolve().with_name("rff_coverage_sqlite_v0.py")
    if not led_path.exists():
        raise SystemExit(f"missing ledger module: {led_path}")
    spec = _importlib_util.spec_from_file_location("rff_coverage_sqlite_v0", str(led_path))
    if not spec or not spec.loader:
        raise SystemExit("failed to load ledger module spec")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

led = _load_ledger_module()

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True, help="Merged field dir containing retrieval_seeds.jsonl")
    ap.add_argument("--ledger_db", required=True, help="SQLite ledger db path")
    args = ap.parse_args()

    field_dir = Path(args.field_dir)
    seeds_path = field_dir / "retrieval_seeds.jsonl"
    if not seeds_path.exists():
        raise SystemExit(f"missing: {seeds_path}")

    # candidate universe from retrieval_seeds
    cand: Dict[str, str] = {}  # concept_id -> domain
    for r in iter_jsonl(seeds_path):
        cid = r.get("concept_id")
        dom = r.get("domain")
        if cid and dom:
            cand[str(cid)] = str(dom)

    conn = led.open_db(args.ledger_db)
    seen = led.get_seen_set(conn)

    total = len(cand)
    expanded = sum(1 for cid in cand.keys() if cid in seen)
    remaining = total - expanded

    print(f"field_dir={field_dir}")
    print(f"ledger_db={args.ledger_db}")
    print(f"candidates_total={total}")
    print(f"expanded_unique={expanded}")
    print(f"remaining_unique={remaining}")

    # per-domain expanded counts (intersection)
    per_dom: Dict[str, int] = {}
    for cid, dom in cand.items():
        if cid in seen:
            per_dom[dom] = per_dom.get(dom, 0) + 1

    if per_dom:
        print("\nexpanded_by_domain (top 20):")
        for dom, n in sorted(per_dom.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {dom}\t{n}")


if __name__ == "__main__":
    main()

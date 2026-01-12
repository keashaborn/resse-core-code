#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from collections import defaultdict

NEG = re.compile(r"\b(not|no|never|cannot|can't|does not|do not|is not|isn't|aren't|without)\b", re.I)
DEP = re.compile(r"\b(require|requires|required|depend|depends|need|needs|must|assume|assumes|only if)\b", re.I)

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    field = Path(args.field_dir).resolve()
    outd  = Path(args.out_dir).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    members_p = field / "concept_members.jsonl"
    edges_p   = field / "concept_edges.jsonl"
    if not members_p.exists() or not edges_p.exists():
        raise SystemExit("field_dir missing concept_members.jsonl or concept_edges.jsonl")

    # concept_id -> fact_i -> text
    facts = defaultdict(dict)
    for r in iter_jsonl(members_p):
        cid = r.get("concept_id")
        fi  = r.get("fact_i")
        txt = r.get("text")
        if cid and isinstance(fi, int) and isinstance(txt, str):
            facts[str(cid)][fi] = txt

    changed = 0
    con_total = con_downgraded = 0
    dep_total = dep_downgraded = 0

    out_edges = outd / "concept_edges.jsonl"
    with out_edges.open("w", encoding="utf-8") as w:
        for e in iter_jsonl(edges_p):
            rt = e.get("rel_type")
            cid = e.get("concept_id")
            si, di = e.get("src_i"), e.get("dst_i")
            if not cid or not isinstance(si, int) or not isinstance(di, int) or not isinstance(rt, str):
                w.write(json.dumps(e, ensure_ascii=False) + "\n")
                continue

            cid = str(cid)
            s_txt = facts.get(cid, {}).get(si, "") or ""
            d_txt = facts.get(cid, {}).get(di, "") or ""

            if rt == "contradicts":
                con_total += 1
                if not (NEG.search(s_txt) or NEG.search(d_txt)):
                    e["rel_type"] = "refines"
                    con_downgraded += 1
                    changed += 1

            elif rt == "depends_on":
                dep_total += 1
                if not DEP.search(d_txt):
                    e["rel_type"] = "refines"
                    dep_downgraded += 1
                    changed += 1

            w.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Copy-through other files (symlinks are fine too, but keep explicit here)
    for fn in ["concept_nodes.jsonl", "concept_members.jsonl", "retrieval_seeds.jsonl", "concept_aliases.jsonl"]:
        src = field / fn
        if src.exists():
            (outd / fn).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    report = {
        "field_dir": str(field),
        "out_dir": str(outd),
        "changed_total": changed,
        "contradicts_total": con_total,
        "contradicts_downgraded": con_downgraded,
        "depends_on_total": dep_total,
        "depends_on_downgraded": dep_downgraded,
    }
    (outd / "hygiene_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", str(outd))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

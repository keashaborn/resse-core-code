#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time, re, unicodedata
from pathlib import Path
from typing import Iterable, List

WORD = re.compile(r"\b[\w'’-]+\b")

STOP = {
    "the","a","an","and","or","of","to","in","on","by","for","with","as","is","are","was","were",
    "be","been","being","that","which","this","these","those","it","its","into","from","at",
    "if","then","there","exists","such"
}

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def toks(s: str) -> list[str]:
    s = nfkc(s).lower()
    return [w for w in WORD.findall(s) if w]

def content_set(s: str) -> set[str]:
    out: set[str] = set()
    for w in toks(s):
        # keep mathy short tokens like ln, dx, dy; drop 1-char noise
        if w in STOP:
            continue
        if len(w) == 1:
            continue
        # drop pure digits
        if w.isdigit():
            continue
        out.add(w)
    return out

def jacc(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

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
    ap.add_argument("--merged_field_dir", required=True, help="Field dir containing concept_nodes.jsonl (used for canonical texts)")
    ap.add_argument("--cross_in", required=True, help="concept_edges_cross.jsonl from passX_link_v0")
    ap.add_argument("--out_root", default="/home/ubuntu/resse-core/fm_teacher/field")
    ap.add_argument("--run_tag", default="passXlink_v1")

    ap.add_argument("--min_src_dst", type=float, default=0.06)
    ap.add_argument("--min_query_dst", type=float, default=0.08)
    ap.add_argument("--min_query_src", type=float, default=0.06)

    ap.add_argument("--min_query_tokens_hit", type=int, default=2)
    ap.add_argument("--min_votes", type=int, default=3, help="Require at least this many independent DST-side signals to keep a cross-link")

    args = ap.parse_args()

    field = Path(args.merged_field_dir)
    nodes = {n["concept_id"]: n for n in iter_jsonl(field / "concept_nodes.jsonl") if n.get("concept_id")}

    cross = list(iter_jsonl(Path(args.cross_in)))

    out_rows: List[dict] = []
    kept = 0
    dropped = 0
    self_loops = 0
    missing_nodes = 0
    missing_ids = 0

    for e in cross:
        s = e.get("src_concept_id")
        d = e.get("dst_concept_id")
        if not s or not d:
            missing_ids += 1
            continue
        if s == d:
            self_loops += 1
            continue

        sn = nodes.get(s)
        dn = nodes.get(d)
        if not sn or not dn:
            missing_nodes += 1
            continue

        src_txt = sn.get("canonical_text", "")
        dst_txt = dn.get("canonical_text", "")
        qlist = e.get("retrieval_queries") or []
        qblob = " ".join(qlist)

        A = content_set(src_txt)
        B = content_set(dst_txt)
        Q = content_set(qblob)

        score_src_dst = jacc(A, B)
        score_query_dst = jacc(Q, B)
        score_query_src = jacc(Q, A)

        hits_dst = len(Q & B)
        hits_src = len(Q & A)

        votes = 0
        vote_reasons: list[str] = []

        # Votes must be DST-evidenced (to avoid “free votes” from src/query overlap)
        if score_src_dst >= args.min_src_dst:
            votes += 1
            vote_reasons.append("src_dst")
        if score_query_dst >= args.min_query_dst:
            votes += 1
            vote_reasons.append("query_dst")
        if hits_dst >= args.min_query_tokens_hit:
            votes += 1
            vote_reasons.append("hits_dst")

        # Sanity gate: if the query keywords don’t even resemble the SRC, treat the link as suspect
        src_query_ok = score_query_src >= args.min_query_src

        keep = bool(src_query_ok and (votes >= args.min_votes))

        out = dict(e)
        out["score_src_dst"] = round(score_src_dst, 4)
        out["score_query_dst"] = round(score_query_dst, 4)
        out["score_query_src"] = round(score_query_src, 4)
        out["query_token_hits_dst"] = hits_dst
        out["query_token_hits_src"] = hits_src
        out["votes"] = votes
        out["vote_reasons"] = vote_reasons
        out["src_query_ok"] = bool(src_query_ok)
        out["keep"] = keep

        if not keep:
            if not src_query_ok:
                out["drop_reason"] = "query_src_low"
            else:
                out["drop_reason"] = f"votes<{args.min_votes}"

        # Always record; PassG will use keep==True rows only
        out_rows.append(out)
        if keep:
            kept += 1
        else:
            dropped += 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "concept_edges_cross_scored.jsonl"
    n = write_jsonl(out_path, out_rows)

    summary = out_dir / "summary.md"
    summary.write_text(
        "# PassX-link v1 (scored) summary\n\n"
        f"- merged_field_dir: {field}\n"
        f"- cross_in: {args.cross_in}\n"
        f"- out_dir: {out_dir}\n\n"
        f"- total_rows_written: {n}\n"
        f"- kept: {kept}\n"
        f"- dropped: {dropped}\n"
        f"- self_loops_skipped: {self_loops}\n"
        f"- missing_ids_skipped: {missing_ids}\n"
        f"- missing_nodes_skipped: {missing_nodes}\n"
        f"- thresholds: min_src_dst={args.min_src_dst} min_query_dst={args.min_query_dst} "
        f"min_query_src={args.min_query_src} min_query_tokens_hit={args.min_query_tokens_hit} min_votes={args.min_votes}\n",
        encoding="utf-8"
    )

    print("WROTE", str(out_dir))
    print("rows", n, "kept", kept, "dropped", dropped, "self_loops_skipped", self_loops, "missing_ids", missing_ids, "missing_nodes", missing_nodes)
    print("edges_file", str(out_path))
    print("summary", str(summary))

if __name__ == "__main__":
    main()

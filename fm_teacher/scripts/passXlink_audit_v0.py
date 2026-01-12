#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, collections, unicodedata
from pathlib import Path

# Normalize unicode math so token overlap isn't destroyed by symbols/Greek/subscripts.
# Goal: a robust *audit* signal, not perfect NLP.

# Map common Greek letters to names (keeps α-divergence etc. meaningful).
GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}

STOP = {
    "the","a","an","and","or","of","to","in","on","by","for","with","as","is","are","was","were",
    "be","been","being","that","which","this","these","those","it","its","into","from","at"
}

# Token pattern: keep underscores so F_z survives, keep digits, keep compact math IDs.
TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)

# Keep single-letter variables that matter in math.
VAR_KEEP = set("xyztnmkpqrabc")

# Unicode subscripts/superscripts -> ascii (minimal set that matters for audit)
_SUBSUP_MAP = str.maketrans({
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9",
    "₊":"+","₋":"-","₌":"=",
})

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.translate(_SUBSUP_MAP)

    # Greek letters -> names (α-divergence etc.)
    for ch, name in GREEK_MAP.items():
        s = s.replace(ch, f" {name} ")

    # Preserve partial-derivative patterns before stripping symbols.
    # ∂z/∂x -> dz_dx, d/dx -> d_dx
    s = s.replace("∂", "d")
    s = re.sub(r"\bd\s*([a-z])\s*/\s*d\s*([a-z])\b", r"d\1_d\2", s)
    s = re.sub(r"\bd\s*/\s*d\s*([a-z])\b", r"d_d\1", s)

    # Turn common relation symbols into tokens rather than deleting signal.
    s = s.replace("≤", " le ").replace("≥", " ge ").replace("≠", " ne ").replace("=", " eq ")
    s = s.replace("→", " to ").replace("⇒", " implies ").replace("↔", " iff ")
    s = s.replace("∈", " in ").replace("⊆", " subset ").replace("⊂", " psubset ")

    # Keep underscores; convert everything else non-alnum/_ to spaces.
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    return s

def toks(s: str) -> list[str]:
    return TOKEN.findall(_norm(s))

def content_set(s: str) -> set[str]:
    out = set()
    for w in toks(s):
        if w in STOP:
            continue
        if w.isdigit():
            continue
        # Keep 1-char tokens only if they are common math variables.
        if len(w) == 1 and w not in VAR_KEEP:
            continue
        out.add(w)
    return out

def jacc(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_dir", required=True, help="Merged field dir containing concept_nodes.jsonl (and optionally concept_edges_cross.jsonl)")
    ap.add_argument("--scored_in", default="", help="Optional path to concept_edges_cross_scored.jsonl (from passX_link_v1). If set, audit this file instead of field_dir/concept_edges_cross.jsonl.")
    ap.add_argument("--only_keep", type=int, default=1, help="When --scored_in is set: 1=audit only rows where keep==true; 0=audit all rows.")
    ap.add_argument("--min_jacc", type=float, default=0.08, help="flag links below this similarity")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    d = Path(args.field_dir)

    # load nodes (needed for canonical_text lookup)
    nodes = {
        r["concept_id"]: r
        for r in (
            json.loads(ln)
            for ln in (d / "concept_nodes.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    }

    # Load cross-links: either from scored file or from field_dir
    cross_path = Path(args.scored_in) if args.scored_in else (d / "concept_edges_cross.jsonl")
    cross = [json.loads(ln) for ln in cross_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # If auditing scored links, optionally restrict to keep==true
    if args.scored_in and args.only_keep:
        cross = [e for e in cross if e.get("keep") is True]
    flags = []
    stats = collections.Counter()

    for e in cross:
        s = e.get("src_concept_id"); t = e.get("dst_concept_id")
        if not s or not t:
            stats["missing_id"] += 1
            continue
        sn = nodes.get(s); tn = nodes.get(t)
        if not sn or not tn:
            stats["missing_node"] += 1
            continue
        src_txt = sn.get("canonical_text","")
        dst_txt = tn.get("canonical_text","")
        q_blob = " ".join(e.get("retrieval_queries") or [])
        js = jacc(content_set(src_txt), content_set(dst_txt))
        jq = jacc(content_set(q_blob), content_set(dst_txt))
        # We flag if dst is dissimilar to src AND also poorly matched to the query keywords.
        if js < args.min_jacc and jq < args.min_jacc:
            flags.append((js, jq, e, src_txt, dst_txt))
            stats["flagged_low_similarity"] += 1

    stats["total_cross_edges"] = len(cross)
    print("total_cross_edges", len(cross))
    print("flagged_low_similarity", stats["flagged_low_similarity"])
    if len(cross):
        print("flag_rate", round(stats["flagged_low_similarity"]/len(cross), 4))

    flags.sort(key=lambda x: (x[0]+x[1]))  # worst first
    for js, jq, e, src_txt, dst_txt in flags[:args.show]:
        print("\n---")
        print("jacc_src_dst", f"{js:.3f}", "jacc_query_dst", f"{jq:.3f}")
        print("queries:", e.get("retrieval_queries", []))
        print("SRC:", src_txt[:200])
        print("DST:", dst_txt[:200])

if __name__ == "__main__":
    main()

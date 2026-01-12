#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, time, re
from typing import Any, Dict, List, Tuple
from openai import OpenAI

# ---- cheap gates (batch-safe) ----
META_PAT = re.compile(
    r"\b("
    r"these statements|these points|these sentences|taken together|one statement|other accounts|"
    r"refer to the same|can be read as"
    r")\b",
    re.IGNORECASE,
)

WORD_PAT = re.compile(r"\b[\w'-]+\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def wc(s: str) -> int:
    return len(WORD_PAT.findall(s or ""))

def toks(s: str) -> List[str]:
    return [w.lower() for w in WORD_PAT.findall(s or "")]

STOP = {
    "the","a","an","and","or","of","to","in","within","inside","across","on","by","for","with",
    "as","is","are","was","were","be","been","being","that","which","this","these","those",
    "because","thereby","thus","then","while","when","where","into","from","at","it","its",
    "facts","fact"
}
BAD_AND_FOLLOW = {"thereby","thus","then","which","that"}
PUNCT_CUT = re.compile(r"[.;:!?]")

def strip_parentheticals(s: str) -> str:
    return re.sub(r"\([^)]*\)", " ", s)

def detect_mixed_context_in_phrase(prose: str, used_idx: List[int], facts_text_by_i: Dict[int, str]) -> bool:
    """
    Generic gate:
    Flags a locative/containment phrase "in/within/inside/across X and Y"
    unless at least one single used fact contains token evidence for BOTH X and Y.
    """
    fact_sets = {i: set(toks(facts_text_by_i.get(i, ""))) for i in used_idx}

    for sent in SENT_SPLIT.split(prose or ""):
        sent0 = sent.strip()
        if not sent0:
            continue

        s = strip_parentheticals(sent0)
        s_lower = s.lower()

        for m in re.finditer(r"\b(in|within|inside|across)\b", s_lower):
            start = m.end()
            window = s[start:start+180]
            cut = PUNCT_CUT.search(window)
            if cut:
                window = window[:cut.start()]

            w = " " + window + " "
            j = w.find(" and ")
            if j == -1:
                continue

            left = w[:j].strip()
            right = w[j+5:].strip()
            if not right:
                continue

            right_first = (toks(right)[:1] or [""])[0]
            if right_first in BAD_AND_FOLLOW:
                continue

            left_t = set(toks(left)) - STOP
            right_t = set(toks(right)) - STOP
            if not left_t or not right_t:
                continue

            supported = False
            for i in used_idx:
                fs = fact_sets.get(i, set())
                if (left_t & fs) and (right_t & fs):
                    supported = True
                    break
            if not supported:
                return True

    return False

def slurp(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def load_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_first_n_jsonl(path: str, n: int):
    got = 0
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if got >= n:
                break
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)
            got += 1

def select_facts(cluster: dict, k: int = 6) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    sp = cluster.get("seed_payload") or {}
    st = (sp.get("text") or "").strip()
    if st:
        out.append((0, st))

    seen_dup = set()
    for mi, m in enumerate(cluster.get("members") or [], start=1):
        if len(out) >= k:
            break
        t = (m.get("text") or "").strip()
        if not t:
            continue
        dg = m.get("dup_group_id")
        if dg and dg in seen_dup:
            continue
        if dg:
            seen_dup.add(dg)
        out.append((mi, t))
    return out

def make_user_text(cluster_id: str, facts: List[Tuple[int, str]]) -> str:
    lines = [f"cluster_id: {cluster_id}", "facts (index -> text):"]
    for i, t in facts:
        lines.append(f"{i}: {t}")
    lines.append("")
    lines.append("Return JSON only per schema.")
    return "\n".join(lines)

def extract_output_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if t is not None:
        return t
    r = resp.model_dump() if hasattr(resp, "model_dump") else resp
    out = []
    for item in (r.get("output") or []):
        for c in (item.get("content") or []):
            if isinstance(c, dict) and "text" in c:
                out.append(c["text"])
    return "".join(out)

def call_model(
    client: OpenAI,
    model: str,
    sys_prompt: str,
    schema_doc: dict,
    cluster_id: str,
    facts: List[Tuple[int, str]],
    max_output_tokens: int,
    extra_user: str = "",
) -> str:
    user_text = make_user_text(cluster_id, facts)
    if extra_user:
        user_text = user_text + "\n\n" + extra_user.strip() + "\n"

    body = dict(
        model=model,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_doc["name"],
                "schema": schema_doc["schema"],
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    resp = client.responses.create(**body)
    return extract_output_text(resp).strip()

def validate_obj(
    obj: Dict[str, Any],
    k: int,
    presented_idx: List[int],
    facts_text_by_i: Dict[int, str],
    word_lo: int,
    word_hi: int,
) -> List[str]:
    errs: List[str] = []
    required = ["cluster_id", "pass", "error", "used_fact_i", "relations", "teaching_prose", "new_claims"]
    for key in required:
        if key not in obj:
            errs.append(f"missing:{key}")
    if errs:
        return errs

    if obj.get("pass") != "R":
        errs.append("pass!=R")

    err = obj.get("error")
    if err not in ("", "insufficient_support"):
        errs.append(f"bad_error:{err}")

    if (obj.get("new_claims") or []) != []:
        errs.append("new_claims_not_empty")

    presented = set(presented_idx)

    if err == "":
        used = obj.get("used_fact_i") or []
        if not (isinstance(used, list) and len(used) == k):
            errs.append("used_fact_i_len!=K")
            return errs
        if len(set(used)) != len(used):
            errs.append("used_fact_i_not_unique")
        if any((not isinstance(i, int) or i not in presented) for i in used):
            errs.append("used_fact_i_not_in_presented")

        used_set = set(used)

        rels = obj.get("relations") or []
        if not isinstance(rels, list) or not (3 <= len(rels) <= 8):
            errs.append("relations_len_out_of_range")
        else:
            for j, e in enumerate(rels):
                if not isinstance(e, dict):
                    errs.append(f"rel_{j}_not_obj")
                    continue
                for rk in ["src_i", "dst_i", "rel_type", "support_i_list"]:
                    if rk not in e:
                        errs.append(f"rel_{j}_missing:{rk}")
                        continue
                si, di = e.get("src_i"), e.get("dst_i")
                sup = e.get("support_i_list") or []
                if not (isinstance(si, int) and isinstance(di, int)):
                    errs.append(f"rel_{j}_bad_src_dst")
                else:
                    if si not in used_set or di not in used_set:
                        errs.append(f"rel_{j}_src_dst_not_in_used")
                if not (isinstance(sup, list) and (1 <= len(sup) <= 3)):
                    errs.append(f"rel_{j}_bad_support_shape")
                else:
                    if len(set(sup)) != len(sup):
                        errs.append(f"rel_{j}_support_not_unique")
                    if any((not isinstance(x, int) or x not in used_set) for x in sup):
                        errs.append(f"rel_{j}_support_not_in_used")
                    if (isinstance(si, int) and isinstance(di, int)) and not ({si, di}.issubset(set(sup))):
                        errs.append(f"rel_{j}_support_missing_endpoints")

        prose = obj.get("teaching_prose") or ""
        n = wc(prose)
        if n < word_lo or n > word_hi:
            errs.append(f"word_count_out_of_band:{n}")

        if META_PAT.search(prose or ""):
            errs.append("meta_provenance_phrase_present")

        if detect_mixed_context_in_phrase(prose, list(used_set), facts_text_by_i):
            errs.append("mixed_context_in_phrase")

    else:
        if (obj.get("used_fact_i") or []) != []:
            errs.append("failure_used_fact_i_not_empty")
        if (obj.get("relations") or []) != []:
            errs.append("failure_relations_not_empty")
        if obj.get("teaching_prose", "") != "":
            errs.append("failure_teaching_prose_not_empty")
        if (obj.get("new_claims") or []) != []:
            errs.append("failure_new_claims_not_empty")

    return errs

ERROR_SEV = {
    "json_parse_error": 100,
    "empty_output_text": 100,
    "used_fact_i_len!=K": 90,
    "used_fact_i_not_in_presented": 90,
    "used_fact_i_not_unique": 80,
    "relations_len_out_of_range": 70,
    "rel_": 60,  # prefix match
    "word_count_out_of_band": 40,  # prefix match
    "mixed_context_in_phrase": 35,
    "meta_provenance_phrase_present": 20,
}

def errs_score(errs: List[str]) -> int:
    score = 0
    for e in (errs or []):
        if e.startswith("json_parse_error"):
            score += ERROR_SEV["json_parse_error"]
        elif e.startswith("rel_"):
            score += ERROR_SEV["rel_"]
        elif e.startswith("word_count_out_of_band"):
            score += ERROR_SEV["word_count_out_of_band"]
        else:
            score += ERROR_SEV.get(e, 10)
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_dir", default="/home/ubuntu/resse-core/sc_fractal_field/results/clusters_by_domain_v0")
    ap.add_argument("--domains", required=True)
    ap.add_argument("--n_per_domain", type=int, default=5)
    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--system_prompt_file", required=True)
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--word_lo", type=int, default=120)
    ap.add_argument("--word_hi", type=int, default=260)
    ap.add_argument("--max_output_tokens", type=int, default=2000)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"/home/ubuntu/resse-core/fm_teacher/evals/passR_prompt_v0_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    ok_path = os.path.join(out_dir, "ok.jsonl")
    bad_path = os.path.join(out_dir, "bad.jsonl")
    review_path = os.path.join(out_dir, "review.md")

    sys_prompt = slurp(args.system_prompt_file)
    schema_doc = load_json(args.schema_file)
    client = OpenAI(timeout=90.0)

    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    ok = bad = 0
    examples_for_review: List[dict] = []

    def parse_and_validate(text: str, facts: List[Tuple[int, str]]):
        try:
            obj = json.loads(text) if text else None
        except Exception as e:
            return None, [f"json_parse_error:{type(e).__name__}:{str(e)[:160]}"]
        if not isinstance(obj, dict):
            return None, ["empty_output_text"]

        facts_text_by_i = {i: t for i, t in facts}
        presented_idx = [i for i, _ in facts]
        errs = validate_obj(obj, args.k, presented_idx, facts_text_by_i, args.word_lo, args.word_hi)
        return obj, errs

    with open(ok_path, "w", encoding="utf-8") as okf, open(bad_path, "w", encoding="utf-8") as badf:
        for d in doms:
            src = os.path.join(args.clusters_dir, f"{d}.jsonl")
            if not os.path.exists(src):
                raise FileNotFoundError(src)

            for cluster in iter_first_n_jsonl(src, args.n_per_domain):
                cluster_id = cluster.get("seed_fact_id") or ""
                facts = select_facts(cluster, k=args.k)

                # First attempt
                out_text = call_model(client, args.model, sys_prompt, schema_doc, cluster_id, facts, args.max_output_tokens)
                obj, errs = parse_and_validate(out_text, facts)

                rec = {
                    "domain": d,
                    "cluster_id": cluster_id,
                    "facts": [{"i": i, "text": t} for i, t in facts],
                    "raw": out_text,
                }

                # One repair retry if failed
                if errs:
                    repair_msg = "VALIDATION FAILED. Fix and return corrected JSON only.\nErrors:\n- " + "\n- ".join(errs)

                    # Make the common mechanical fix unambiguous
                    if "used_fact_i_not_unique" in errs:
                        repair_msg += "\n\nExtra constraint: used_fact_i must contain exactly 6 DISTINCT integers from the provided indices (no repeats)."

                    out_text2 = call_model(
                        client, args.model, sys_prompt, schema_doc, cluster_id, facts, args.max_output_tokens,
                        extra_user=repair_msg
                    )
                    rec["raw_retry"] = out_text2

                    obj2, errs2 = parse_and_validate(out_text2, facts)

                    # Keep the better of the two attempts (do NOT regress)
                    if obj2 is not None:
                        if (not errs2) or (errs_score(errs2) < errs_score(errs)) or (obj is None):
                            obj, errs = obj2, errs2

                if obj is not None:
                    rec["obj"] = obj
                rec["validation_errors"] = errs

                if errs:
                    bad += 1
                    badf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    ok += 1
                    okf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if len(examples_for_review) < 12:
                        examples_for_review.append(rec)

    with open(review_path, "w", encoding="utf-8") as fh:
        fh.write("# PassR prompt eval\n\n")
        fh.write(f"- model: {args.model}\n")
        fh.write(f"- K: {args.k}\n")
        fh.write(f"- word_band: {args.word_lo}-{args.word_hi}\n")
        fh.write(f"- domains: {', '.join(doms)}\n\n")
        fh.write("## Summary\n\n")
        fh.write(f"- ok: {ok}\n- bad: {bad}\n\n")
        for idx, rec in enumerate(examples_for_review, start=1):
            obj = rec.get("obj") or {}
            fh.write("---\n\n")
            fh.write(f"## Example {idx}: {rec['domain']} | {rec['cluster_id']}\n\n")
            fh.write("### Facts\n\n")
            for f in rec["facts"]:
                fh.write(f"- [{f['i']}] {f['text']}\n")
            fh.write("\n### used_fact_i\n\n")
            fh.write(str(obj.get("used_fact_i")) + "\n\n")
            fh.write("### relations\n\n")
            fh.write(json.dumps(obj.get("relations"), ensure_ascii=False, indent=2) + "\n\n")
            fh.write("### teaching_prose\n\n")
            fh.write(obj.get("teaching_prose", "") + "\n\n")

    total = ok + bad
    print("WROTE", out_dir)
    print("ok", ok, "bad", bad, "ok_rate", round(ok / max(total, 1), 4))
    print("review", review_path)

if __name__ == "__main__":
    main()

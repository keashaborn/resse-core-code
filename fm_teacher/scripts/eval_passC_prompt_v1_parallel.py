from __future__ import annotations

import argparse, json, os, time, re, unicodedata, random
from typing import Any, Dict, List, Tuple
from collections import Counter
from openai import OpenAI

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

NEG_CUES = re.compile(r"\b(not|no|never|cannot|can't|does not|do not|is not|aren't|isn't|without)\b", re.IGNORECASE)

def _downgrade_nonexplicit_contradicts(obj: Dict[str, Any], facts_text_by_i: Dict[int, str]) -> bool:
    """
    If an edge is labeled contradicts but neither side contains explicit negation cues,
    downgrade it to 'refines' (conservative) to avoid poisoning the backbone.
    Returns True if any change was made.
    """
    changed = False
    edges = obj.get("edge_candidates") or []
    if not isinstance(edges, list):
        return False

    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("rel_type") != "contradicts":
            continue
        si, di = e.get("src_i"), e.get("dst_i")
        if not isinstance(si, int) or not isinstance(di, int):
            continue
        st = facts_text_by_i.get(si, "") or ""
        dt = facts_text_by_i.get(di, "") or ""
        # If neither contains explicit negation cues, this is almost certainly misuse.
        if not (NEG_CUES.search(st) or NEG_CUES.search(dt)):
            e["rel_type"] = "refines"
            changed = True
    return changed


def _prune_to_canonical_component(obj: Dict[str, Any]) -> bool:
    """
    If keep_fact_i induces a disconnected graph under edge_candidates,
    prune keep_fact_i (and edges/roles) to the connected component containing canonical_i.
    Returns True if it modified the object.
    """
    keep = obj.get("keep_fact_i") or []
    edges = obj.get("edge_candidates") or []
    canonical = obj.get("canonical_i")

    if not isinstance(canonical, int) or canonical not in keep:
        return False
    if len(keep) <= 1:
        return False

    keep_set = set(i for i in keep if isinstance(i, int))
    adj = {i: set() for i in keep_set}
    for e in edges:
        if not isinstance(e, dict):
            continue
        si, di = e.get("src_i"), e.get("dst_i")
        if si in adj and di in adj:
            adj[si].add(di)
            adj[di].add(si)

    # BFS from canonical
    seen = {canonical}
    stack = [canonical]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)

    if seen == keep_set:
        return False  # already connected

    # Prune keep_fact_i
    obj["keep_fact_i"] = [i for i in keep if i in seen]

    # Add dropped facts for removed nodes
    removed = sorted(list(keep_set - seen))
    drops = obj.get("drop_facts") or []
    if not isinstance(drops, list):
        drops = []
    existing_drop_i = {it.get("i") for it in drops if isinstance(it, dict)}
    for i in removed:
        if i not in existing_drop_i:
            drops.append({"i": i, "reason": "off_topic"})
    obj["drop_facts"] = drops

    # Prune roles
    fr = obj.get("fact_roles") or []
    if isinstance(fr, list):
        obj["fact_roles"] = [it for it in fr if isinstance(it, dict) and it.get("i") in seen]

    # Prune edges to within component
    new_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        si, di = e.get("src_i"), e.get("dst_i")
        if si in seen and di in seen:
            # prune support list too
            sup = e.get("support_i_list") or []
            if isinstance(sup, list):
                e = dict(e)
                e["support_i_list"] = [x for x in sup if x in seen]
            new_edges.append(e)
    obj["edge_candidates"] = new_edges

    return True

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

ROLE_SET = {
    "definition","mechanism","regime_condition","measurement","consequence",
    "background","example","paraphrase","qualifier"
}
REL_SET = {
    "same_as","entails","contradicts","refines","mechanism_for","condition_for",
    "example_of","quantifies","distinguishes","depends_on","causes","part_of"
}
DROP_REASONS = {"redundant","off_topic","too_vague","multi_claim","malformed","contradiction","other"}

RQ_WORD_PAT = re.compile(r"\b[\w'-]+\b")
RQ_STOP = {
    "the","a","an","and","or","of","to","in","on","by","for","with","as",
    "is","are","was","were","be","been","being","that","which","this","these","those",
}
DEP_CUES = re.compile(r"\b(require|requires|required|depend|depends|need|needs|must|assume|assumes|only if)\b", re.IGNORECASE)


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

SAME_AS_MIN_JACC = 0.40  # conservative; prefer false negatives over false-positive merges
SAME_AS_STOP = {
    "the","a","an","and","or","of","to","in","on","by","for","with","as",
    "is","are","was","were","be","been","being","that","which","this","these","those",
}

def _content_set(s: str) -> set[str]:
    toks = [w.lower() for w in RQ_WORD_PAT.findall(s or "")]
    out: set[str] = set()
    for t in toks:
        if t in SAME_AS_STOP:
            continue
        if len(t) < 2:
            continue
        if t.isdigit():
            continue
        out.add(t)
    return out

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0


def _rq_tokens(s: str) -> list[str]:
    s = _nfkc(s)
    return [w.lower() for w in RQ_WORD_PAT.findall(s or "")]

def _rq_tokens_ok(query: str, facts_blob: str) -> bool:
    for tok in _rq_tokens(query):
        if len(tok) < 3 or tok in RQ_STOP:
            continue
        if tok in facts_blob:
            continue
        # tolerate basic singular/plural variants (density <-> densities)
        if tok.endswith("y") and (tok[:-1] + "ies") in facts_blob:
            continue
        if tok.endswith("ies") and (tok[:-3] + "y") in facts_blob:
            continue
        if tok.endswith("s") and tok[:-1] and tok[:-1] in facts_blob:
            continue
        if (tok + "s") in facts_blob:
            continue
        return False
    return True

def _graph_connected(nodes: set[int], edges: list[dict]) -> bool:
    if len(nodes) <= 1:
        return True
    adj = {n: set() for n in nodes}
    for e in edges:
        if not isinstance(e, dict):
            continue
        si, di = e.get("src_i"), e.get("dst_i")
        if si in adj and di in adj:
            adj[si].add(di)
            adj[di].add(si)
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(nodes)


def validate_obj(obj: Dict[str, Any], k: int, presented_idx: List[int], facts_text_by_i: Dict[int, str]) -> List[str]:
    errs: List[str] = []
    required = [
        "cluster_id","pass","error","keep_fact_i","drop_facts","canonical_i",
        "fact_roles","edge_candidates","retrieval_queries","new_claims"
    ]
    for key in required:
        if key not in obj:
            errs.append(f"missing:{key}")
    if errs:
        return errs

    if obj.get("pass") != "C":
        errs.append("pass!=C")

    err = obj.get("error")
    if err not in ("", "insufficient_support"):
        errs.append(f"bad_error:{err}")

    if (obj.get("new_claims") or []) != []:
        errs.append("new_claims_not_empty")

    presented = set(presented_idx)

    facts_blob = _nfkc(" ".join((facts_text_by_i or {}).values())).lower()

    if err == "":
        keep = obj.get("keep_fact_i") or []
        if not isinstance(keep, list) or not (1 <= len(keep) <= k):
            errs.append("keep_fact_i_len_out_of_range")
            return errs
        if any((not isinstance(i,int) or i not in presented) for i in keep):
            errs.append("keep_fact_i_not_in_presented")
        if len(set(keep)) != len(keep):
            errs.append("keep_fact_i_not_unique")

        canonical = obj.get("canonical_i")
        if not isinstance(canonical, int):
            errs.append("canonical_i_not_int")
        else:
            if canonical not in set(keep):
                errs.append("canonical_i_not_in_keep")

        drops = obj.get("drop_facts") or []
        if not isinstance(drops, list):
            errs.append("drop_facts_not_list")
        else:
            seen_drop = set()
            for j, it in enumerate(drops):
                if not isinstance(it, dict):
                    errs.append(f"drop_{j}_not_obj")
                    continue
                i = it.get("i")
                r = it.get("reason")
                if not isinstance(i, int) or i not in presented:
                    errs.append(f"drop_{j}_bad_i")
                if r not in DROP_REASONS:
                    errs.append(f"drop_{j}_bad_reason")
                if i in seen_drop:
                    errs.append(f"drop_{j}_dup_i")
                seen_drop.add(i)
                if i in set(keep):
                    errs.append(f"drop_{j}_overlaps_keep")

        roles = obj.get("fact_roles") or []
        if not isinstance(roles, list):
            errs.append("fact_roles_not_list")
        else:
            seen = {}
            for j, it in enumerate(roles):
                if not isinstance(it, dict):
                    errs.append(f"role_{j}_not_obj")
                    continue
                i = it.get("i")
                r = it.get("role")
                if not isinstance(i, int) or i not in set(keep):
                    errs.append(f"role_{j}_bad_i")
                    continue
                if r not in ROLE_SET:
                    errs.append(f"role_{j}_bad_role")
                    continue
                if i in seen:
                    errs.append(f"role_{j}_dup_i")
                seen[i] = r
            if set(keep) - set(seen.keys()):
                errs.append("missing_roles_for_keep")
            # canonical should be a definition when possible
            if isinstance(canonical, int):
                canon_role = seen.get(canonical)
                if canon_role is not None and canon_role != "definition":
                    errs.append("canonical_role_not_definition")

        edges = obj.get("edge_candidates") or []
        if not isinstance(edges, list):
            errs.append("edge_candidates_not_list")
        else:
            keep_set = set(keep)
            for j, e in enumerate(edges):
                if not isinstance(e, dict):
                    errs.append(f"edge_{j}_not_obj")
                    continue
                for rk in ["src_i", "dst_i", "rel_type", "support_i_list"]:
                    if rk not in e:
                        errs.append(f"edge_{j}_missing:{rk}")
                        continue

                si, di = e.get("src_i"), e.get("dst_i")
                rt = e.get("rel_type")
                sup = e.get("support_i_list") or []

                if not (isinstance(si, int) and isinstance(di, int)):
                    errs.append(f"edge_{j}_bad_src_dst")
                else:
                    if si not in keep_set or di not in keep_set:
                        errs.append(f"edge_{j}_src_dst_not_in_keep")

                if rt not in REL_SET:
                    errs.append(f"edge_{j}_bad_rel_type")

                if not (isinstance(sup, list) and (1 <= len(sup) <= 3)):
                    errs.append(f"edge_{j}_bad_support_shape")
                else:
                    if len(set(sup)) != len(sup):
                        errs.append(f"edge_{j}_support_not_unique")
                    if any((not isinstance(x, int) or x not in keep_set) for x in sup):
                        errs.append(f"edge_{j}_support_not_in_keep")
                    if (isinstance(si, int) and isinstance(di, int)) and not ({si, di}.issubset(set(sup))):
                        errs.append(f"edge_{j}_support_missing_endpoints")

                # same_as must be near-duplicate proposition; avoid semantic drift
                if rt == "same_as":
                    src_txt = facts_text_by_i.get(si, "") if isinstance(si, int) else ""
                    dst_txt = facts_text_by_i.get(di, "") if isinstance(di, int) else ""
                    j = _jaccard(_content_set(src_txt), _content_set(dst_txt))
                    if j < SAME_AS_MIN_JACC:
                        errs.append(f"same_as_low_similarity:{si}->{di}:{j:.2f}")

                # depends_on must be explicit in the dst fact text
                if rt == "depends_on":
                    dst_txt = facts_text_by_i.get(di, "") if isinstance(di, int) else ""
                    if not DEP_CUES.search(dst_txt or ""):
                        errs.append(f"depends_on_without_dependency_language:{si}->{di}")

                # contradicts must be explicit (negation cues) or it's almost certainly misuse
                if rt == "contradicts":
                    src_txt = facts_text_by_i.get(si, "") if isinstance(si, int) else ""
                    dst_txt = facts_text_by_i.get(di, "") if isinstance(di, int) else ""
                    if not (NEG_CUES.search(src_txt or "") or NEG_CUES.search(dst_txt or "")):
                        errs.append("contradicts_not_explicit")


            # require at least one edge if we kept >=2 facts
            if len(keep) >= 2 and len(edges) < 1:
                errs.append("too_few_edges_for_keep")
            elif len(keep) >= 2:
                endpoints: set[int] = set()
                same_as_nodes: set[int] = set()

                for e in edges:
                    if not isinstance(e, dict):
                        continue
                    si, di = e.get("src_i"), e.get("dst_i")
                    rt = e.get("rel_type")

                    if isinstance(si, int) and si in keep_set:
                        endpoints.add(si)
                    if isinstance(di, int) and di in keep_set:
                        endpoints.add(di)

                    if rt == "same_as":
                        if isinstance(si, int) and si in keep_set:
                            same_as_nodes.add(si)
                        if isinstance(di, int) and di in keep_set:
                            same_as_nodes.add(di)

                if keep_set - endpoints:
                    errs.append("keep_fact_missing_edge_coverage")

                if not _graph_connected(keep_set, edges):
                    errs.append("keep_graph_disconnected")

                paraphrase_ids = set()
                for it in (obj.get("fact_roles") or []):
                    if isinstance(it, dict) and it.get("role") == "paraphrase":
                        i = it.get("i")
                        if isinstance(i, int) and i in keep_set:
                            paraphrase_ids.add(i)

                if paraphrase_ids and (paraphrase_ids - same_as_nodes):
                    errs.append("paraphrase_role_without_same_as_edge")


        rq = obj.get("retrieval_queries") or []
        if not isinstance(rq, list):
            errs.append("retrieval_queries_not_list")
        else:
            if len(rq) > 5:
                errs.append("retrieval_queries_too_many")
            for j, q in enumerate(rq):
                if not isinstance(q, str) or not q.strip():
                    errs.append(f"retrieval_query_{j}_bad")
                else:
                    if not _rq_tokens_ok(q, facts_blob):
                        errs.append("retrieval_query_token_not_in_facts_warn")

    else:
        # insufficient_support => empty shape
        if (obj.get("keep_fact_i") or []) != []:
            errs.append("failure_keep_not_empty")
        if (obj.get("drop_facts") or []) != []:
            errs.append("failure_drop_not_empty")
        if (obj.get("fact_roles") or []) != []:
            errs.append("failure_roles_not_empty")
        if (obj.get("edge_candidates") or []) != []:
            errs.append("failure_edges_not_empty")
        if (obj.get("retrieval_queries") or []) != []:
            errs.append("failure_queries_not_empty")
        if (obj.get("new_claims") or []) != []:
            errs.append("failure_new_claims_not_empty")
        # canonical_i can be any int; recommend 0 but don't enforce to keep schema flexible

    # Demote semantic heuristics to audit-only.
    # These are useful signals, but too brittle to hard-fail PassC at scale.
    errs = [
        e for e in errs
        if e != "paraphrase_role_without_same_as_edge"
        and not e.startswith("same_as_low_similarity:")
    ]
    # Do not fail the entire cluster for retrieval query issues; treat as warning.
    errs = [e for e in errs if e != "retrieval_query_token_not_in_facts_warn"]
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_dir", default="/home/ubuntu/resse-core/sc_fractal_field/results/clusters_by_domain_v0")
    ap.add_argument("--domains", required=True)
    ap.add_argument("--n_per_domain", type=int, default=10)
    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--system_prompt_file", required=True)
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max_output_tokens", type=int, default=2000)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--resume_from_dir", default="", help="If set, reuse this existing out_dir and skip already-processed cluster_id values found in ok.jsonl/bad.jsonl")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (threads) for API calls.")
    ap.add_argument("--print_every", type=int, default=25, help="Print progress every N completed clusters (0 disables).")
    ap.add_argument("--flush_every", type=int, default=10, help="Flush output files every N completed clusters.")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.resume_from_dir or args.out_dir or f"/home/ubuntu/resse-core/fm_teacher/evals/passC_prompt_v1_parallel_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    ok_path = os.path.join(out_dir, "ok.jsonl")
    bad_path = os.path.join(out_dir, "bad.jsonl")
    review_path = os.path.join(out_dir, "review.md")

    # resume: skip cluster_ids already written (and pre-count ok/bad when resuming)
    done_ids = set()
    ok_existing = 0
    bad_existing = 0
    for _p, _which in ((ok_path, "ok"), (bad_path, "bad")):
        try:
            with open(_p, "r", encoding="utf-8") as _fh:
                for _ln in _fh:
                    _ln = _ln.strip()
                    if not _ln:
                        continue
                    try:
                        _r = json.loads(_ln)
                        cid = _r.get("cluster_id")
                        if cid:
                            done_ids.add(cid)
                            if _which == "ok":
                                ok_existing += 1
                            else:
                                bad_existing += 1
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

    sys_prompt = slurp(args.system_prompt_file)
    schema_doc = load_json(args.schema_file)

    # Thread-local clients so --workers>1 is safe.
    import threading
    import concurrent.futures as cf
    _tls = threading.local()

    def _get_client():
        c = getattr(_tls, "client", None)
        if c is None:
            c = OpenAI(timeout=90.0)
            _tls.client = c
        return c

    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    ok = ok_existing
    bad = bad_existing
    examples_for_review: List[dict] = []

    keep_counts = []
    drop_reason_counts = Counter()
    rel_type_counts = Counter()

    def _process_one(domain: str, cluster: dict) -> tuple[dict, bool, dict]:
        cluster_id = (cluster.get("seed_fact_id") or "").strip()
        facts = select_facts(cluster, k=args.k)
        presented_idx = [i for i, _ in facts]
        facts_text_by_i = {i: t for i, t in facts}

        base_user = make_user_text(cluster_id, facts)

        usage_first = None
        usage_retry = None

        def run_once(extra_user: str = "", max_tokens: int | None = None) -> str:
            user_text = base_user
            if extra_user:
                user_text = user_text + "\n\n" + extra_user.strip() + "\n"
            body = dict(
                model=args.model,
                reasoning={"effort": "low"},
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                ],
                text={"format": {
                    "type": "json_schema",
                    "name": schema_doc["name"],
                    "schema": schema_doc["schema"],
                    "strict": True
                }},
                max_output_tokens=(max_tokens if max_tokens is not None else args.max_output_tokens),
            )
            max_tries = 6
            for attempt in range(1, max_tries + 1):
                try:
                    resp = _get_client().responses.create(**body)

                    # capture token usage (best-effort; varies by SDK/model)
                    try:
                        u = getattr(resp, "usage", None)
                        if u is None and hasattr(resp, "model_dump"):
                            u = (resp.model_dump() or {}).get("usage")
                        if isinstance(u, dict):
                            if extra_user:
                                nonlocal usage_retry
                                usage_retry = u
                            else:
                                nonlocal usage_first
                                usage_first = u
                    except Exception:
                        pass

                    return extract_output_text(resp).strip()
                except Exception as e:
                    ename = type(e).__name__.lower()
                    msg = str(e).lower()
                    code = getattr(e, "status_code", None)
                    transient = (
                        code in (429, 500, 502, 503, 504)
                        or "timeout" in ename
                        or "timeout" in msg
                        or "timed out" in msg
                        or "readtimeout" in msg
                        or "temporarily unavailable" in msg
                        or "connection" in msg
                        or "rate limit" in msg
                        or "overloaded" in msg
                    )
                    if transient and attempt < max_tries:
                        import time as _t, random as _r
                        _sleep = min(30.0, (2 ** (attempt - 1)) + _r.random())
                        _t.sleep(_sleep)
                        continue
                    if transient:
                        return ""
                    raise
                # If we exhausted retries without returning, treat as empty output.
            return ""

        def parse_and_validate(text: str):
            try:
                obj = json.loads(text) if text else None
            except Exception as e:
                return None, [f"json_parse_error:{type(e).__name__}:{str(e)[:160]}"]
            if not isinstance(obj, dict):
                return None, ["empty_output_text"]
            return obj, validate_obj(obj, args.k, presented_idx, facts_text_by_i)

        out_text = run_once()
        obj, errs = parse_and_validate(out_text)
        raw_first = out_text

        # Deterministic salvage (before retry): prune disconnected, downgrade bad contradicts
        if isinstance(obj, dict):
            if "keep_graph_disconnected" in (errs or []):
                if _prune_to_canonical_component(obj):
                    errs = validate_obj(obj, args.k, presented_idx, facts_text_by_i)

            if "contradicts_not_explicit" in (errs or []):
                if _downgrade_nonexplicit_contradicts(obj, facts_text_by_i):
                    errs = validate_obj(obj, args.k, presented_idx, facts_text_by_i)

        # --- one repair retry if needed ---
        raw_retry = ""
        if errs:
            repair_msg = (
                "REPAIR: Your previous output failed validation. Return JSON ONLY matching the schema exactly.\n"
                "Errors:\n- " + "\n- ".join(errs) + "\n\n"
                "Fix rules:\n"
                "- If canonical_role_not_definition: set fact_roles for canonical_i to role='definition'.\n"
                "- If keep_graph_disconnected: either add a supported edge that connects the components OR drop the smaller component from keep_fact_i.\n"
            )
            retry_tokens = max(args.max_output_tokens, 3200)
            out_text2 = run_once(extra_user=repair_msg, max_tokens=retry_tokens)
            raw_retry = out_text2
            obj2, errs2 = parse_and_validate(out_text2)

            # prefer the retry if it parses; otherwise keep the retry errors
            if obj2 is not None:
                obj, errs = obj2, errs2
                out_text = out_text2
            else:
                errs = errs2
                out_text = out_text2

        rec = {
            "domain": domain,
            "cluster_id": cluster_id,
            "facts": [{"i": i, "text": t} for i, t in facts],
            "raw": out_text
        }

        # add provenance to logged facts (fact_id + dup_group_id)
        try:
            members = cluster.get("members") or []
            seed_txt = ((cluster.get("seed_payload") or {}).get("text") or "").strip()
            buckets = {}  # text -> list[(fact_id, dup_group_id)]
            for m in members:
                if not isinstance(m, dict):
                    continue
                t = (m.get("text") or "").strip()
                if not t:
                    continue
                buckets.setdefault(t, []).append((m.get("fact_id"), m.get("dup_group_id")))

            def _pop_meta(txt: str):
                t = (txt or "").strip()
                lst = buckets.get(t) or []
                return lst.pop(0) if lst else (None, None)

            facts_for_log = []
            for i, txt in facts:
                fid, dg = _pop_meta(txt)
                facts_for_log.append({"i": i, "text": txt, "fact_id": fid, "dup_group_id": dg})
            rec["facts"] = facts_for_log

            if seed_txt:
                for m in members:
                    if isinstance(m, dict) and (m.get("text") or "").strip() == seed_txt:
                        rec["seed_fact_uuid"] = m.get("fact_id")
                        rec["seed_dup_group_id"] = m.get("dup_group_id")
                        break
        except Exception:
            pass

        if raw_retry:
            rec["raw_first"] = raw_first
            rec["raw_retry"] = raw_retry

        if obj is not None:
            rec["obj"] = obj
        rec["validation_errors"] = errs

        # attach usage (best-effort)
        rec["usage_first"] = usage_first
        rec["usage_retry"] = usage_retry

        # normalize a total if we can (common keys: input_tokens/output_tokens/total_tokens)
        def _sum_usage(a, b, key):
            va = (a or {}).get(key) if isinstance(a, dict) else None
            vb = (b or {}).get(key) if isinstance(b, dict) else None
            try:
                return (int(va) if va is not None else 0) + (int(vb) if vb is not None else 0)
            except Exception:
                return None

        if isinstance(usage_first, dict) or isinstance(usage_retry, dict):
            rec["usage_total"] = {}
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                s = _sum_usage(usage_first, usage_retry, k)
                if s is not None:
                    rec["usage_total"][k] = s
        # stats for main-thread aggregation
        stats = {"keep_len": 0, "drop_reasons": [], "rel_types": []}
        if obj is not None and not errs:
            try:
                keep = obj.get("keep_fact_i") or []
                stats["keep_len"] = len(keep)
                stats["drop_reasons"] = [
                    (it or {}).get("reason") for it in (obj.get("drop_facts") or [])
                    if (it or {}).get("reason")
                ]
                stats["rel_types"] = [
                    (e or {}).get("rel_type") for e in (obj.get("edge_candidates") or [])
                    if (e or {}).get("rel_type")
                ]
            except Exception:
                pass

        return rec, (not rec["validation_errors"]), stats

    # Build worklist (skip already-done cluster_ids)
    work: List[tuple[str, dict]] = []
    for d in doms:
        src = os.path.join(args.clusters_dir, f"{d}.jsonl")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        for cluster in iter_first_n_jsonl(src, args.n_per_domain):
            cid = (cluster.get("seed_fact_id") or "").strip()
            if cid and cid in done_ids:
                continue
            work.append((d, cluster))

    total_expected = len(done_ids) + len(work)
    print(
        f"passC_start domains={len(doms)} workers={args.workers} "
        f"resume_done={len(done_ids)} queued={len(work)} total_expected={total_expected}"
    )

    file_mode = "a" if args.resume_from_dir else "w"
    with open(ok_path, file_mode, encoding="utf-8", buffering=1) as okf, open(bad_path, file_mode, encoding="utf-8", buffering=1) as badf:
        max_workers = max(1, int(args.workers))
        max_inflight = max_workers * 4  # cap queued futures; avoids submitting N=millions at once

        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures: dict[cf.Future, tuple[str, str]] = {}
            idx = 0
            completed = 0
            total_work = len(work)

            # seed initial in-flight futures
            while idx < total_work and len(futures) < min(max_inflight, total_work):
                d, c = work[idx]
                idx += 1
                cid = (c.get("seed_fact_id") or "").strip()
                fut = ex.submit(_process_one, d, c)
                futures[fut] = (d, cid)

            while futures:
                done_set, _ = cf.wait(list(futures), return_when=cf.FIRST_COMPLETED)
                for fut in done_set:
                    domain, cid_hint = futures.pop(fut)
                    try:
                        rec, is_ok, st = fut.result()
                    except Exception as e:
                        # Worker crashed: record as bad and keep going (don’t kill the whole run)
                        rec = {
                            "domain": domain,
                            "cluster_id": cid_hint,
                            "facts": [],
                            "raw": "",
                            "validation_errors": [f"worker_exception:{type(e).__name__}:{str(e)[:200]}"],
                        }
                        is_ok = False
                        st = {"keep_len": 0, "drop_reasons": [], "rel_types": []}

                    completed += 1

                    cid = (rec.get("cluster_id") or cid_hint or "").strip()
                    if cid:
                        done_ids.add(cid)

                    if is_ok:
                        ok += 1
                        okf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if len(examples_for_review) < 12:
                            examples_for_review.append(rec)
                        keep_counts.append(int(st.get("keep_len") or 0))
                        for r in (st.get("drop_reasons") or []):
                            drop_reason_counts[r] += 1
                        for rt in (st.get("rel_types") or []):
                            rel_type_counts[rt] += 1
                    else:
                        bad += 1
                        badf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    if args.flush_every and completed % int(args.flush_every) == 0:
                        okf.flush()
                        badf.flush()

                    if args.print_every and completed % int(args.print_every) == 0:
                        print(
                            f"passC_progress completed={completed}/{total_work} ok={ok} bad={bad} "
                            f"done_total={len(done_ids)}/{total_expected} inflight={len(futures)}"
                        )

                    # keep inflight full
                    while idx < total_work and len(futures) < max_inflight:
                        d, c = work[idx]
                        idx += 1
                        cid2 = (c.get("seed_fact_id") or "").strip()
                        fut2 = ex.submit(_process_one, d, c)
                        futures[fut2] = (d, cid2)

    with open(review_path, "w", encoding="utf-8") as fh:
        fh.write("# PassC prompt eval\n\n")
        fh.write(f"- model: {args.model}\n")
        fh.write(f"- K: {args.k}\n")
        fh.write(f"- domains: {', '.join(doms)}\n\n")
        fh.write("## Summary\n\n")
        fh.write(f"- ok: {ok}\n- bad: {bad}\n\n")
        fh.write("## Stats\n\n")
        if keep_counts:
            fh.write(f"- keep_mean: {sum(keep_counts)/len(keep_counts):.3f}\n")
            fh.write(f"- keep_min: {min(keep_counts)}\n")
            fh.write(f"- keep_max: {max(keep_counts)}\n")
        fh.write(f"- drop_reasons: {dict(drop_reason_counts)}\n")
        fh.write(f"- rel_types: {dict(rel_type_counts)}\n\n")

        for idx, rec in enumerate(examples_for_review, start=1):
            obj = rec.get("obj") or {}
            fh.write("---\n\n")
            fh.write(f"## Example {idx}: {rec['domain']} | {rec['cluster_id']}\n\n")
            fh.write("### Facts\n\n")
            for f in rec["facts"]:
                fh.write(f"- [{f['i']}] {f['text']}\n")
            fh.write("\n### keep_fact_i\n\n")
            fh.write(str(obj.get("keep_fact_i")) + "\n\n")
            fh.write("### drop_facts\n\n")
            fh.write(json.dumps(obj.get("drop_facts"), ensure_ascii=False, indent=2) + "\n\n")
            fh.write("### canonical_i\n\n")
            fh.write(str(obj.get("canonical_i")) + "\n\n")
            fh.write("### fact_roles\n\n")
            fh.write(json.dumps(obj.get("fact_roles"), ensure_ascii=False, indent=2) + "\n\n")
            fh.write("### edge_candidates\n\n")
            fh.write(json.dumps(obj.get("edge_candidates"), ensure_ascii=False, indent=2) + "\n\n")
            fh.write("### retrieval_queries\n\n")
            fh.write(json.dumps(obj.get("retrieval_queries"), ensure_ascii=False, indent=2) + "\n\n")

    total = ok + bad
    print("WROTE", out_dir)
    print("ok", ok, "bad", bad, "ok_rate", round(ok/max(total,1), 4))
    print("review", review_path)

if __name__ == "__main__":
    main()

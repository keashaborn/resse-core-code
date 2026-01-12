#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


def _get(o: Any, k: str, default: Any = None) -> Any:
    if o is None:
        return default
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


def load_map(map_path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with map_path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            cid = r.get("custom_id")
            if cid:
                out[str(cid)] = r
    return out


def extract_batch_output_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text  # type: ignore[attr-defined]

    # Fallbacks: file-like or bytes
    try:
        raw = content.read()  # type: ignore[attr-defined]
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception:
        return str(content)


def extract_response_output_text(resp_body: Any) -> str:
    """
    Extract assistant text from a Responses API response body.
    Prefer output_text if present; otherwise walk output[].content[].text.
    """
    if not isinstance(resp_body, dict):
        return ""

    ot = resp_body.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = resp_body.get("output")
    if not isinstance(out, list):
        return ""

    chunks: List[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _collect_indices(obj: Dict[str, Any]) -> List[int]:
    idx: List[int] = []

    def add(x: Any):
        if isinstance(x, int):
            idx.append(x)

    for x in (obj.get("keep_fact_i") or []):
        add(x)

    for it in (obj.get("drop_facts") or []):
        if isinstance(it, dict):
            add(it.get("i"))

    add(obj.get("canonical_i"))

    for it in (obj.get("fact_roles") or []):
        if isinstance(it, dict):
            add(it.get("i"))

    for e in (obj.get("edge_candidates") or []):
        if not isinstance(e, dict):
            continue
        add(e.get("src_i"))
        add(e.get("dst_i"))
        sil = e.get("support_i_list")
        if isinstance(sil, list):
            for x in sil:
                add(x)

    return idx


def _apply_index_mapping(obj: Dict[str, Any], mapping: Dict[int, int]) -> Dict[str, Any]:
    # cheap deep-copy
    o = json.loads(json.dumps(obj))

    def map_int(x: Any) -> Any:
        return mapping.get(x, x) if isinstance(x, int) else x

    if isinstance(o.get("keep_fact_i"), list):
        o["keep_fact_i"] = [map_int(x) for x in o["keep_fact_i"]]

    if isinstance(o.get("drop_facts"), list):
        for it in o["drop_facts"]:
            if isinstance(it, dict) and isinstance(it.get("i"), int):
                it["i"] = map_int(it["i"])

    if isinstance(o.get("canonical_i"), int):
        o["canonical_i"] = map_int(o["canonical_i"])

    if isinstance(o.get("fact_roles"), list):
        for it in o["fact_roles"]:
            if isinstance(it, dict) and isinstance(it.get("i"), int):
                it["i"] = map_int(it["i"])

    if isinstance(o.get("edge_candidates"), list):
        for e in o["edge_candidates"]:
            if not isinstance(e, dict):
                continue
            if isinstance(e.get("src_i"), int):
                e["src_i"] = map_int(e["src_i"])
            if isinstance(e.get("dst_i"), int):
                e["dst_i"] = map_int(e["dst_i"])
            sil = e.get("support_i_list")
            if isinstance(sil, list):
                e["support_i_list"] = [map_int(x) for x in sil]

    return o


def maybe_remap_indices(obj: Dict[str, Any], presented_order: List[int]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    If the model used ordinal indexing (0..k-1 or 1..k) instead of the actual presented indices,
    remap to the true presented indices by order.

    This salvages the common failure mode where map.jsonl stores packed indices but the model outputs
    the displayed ordinal indices.
    """
    if not presented_order:
        return obj, None

    presented_set = set(presented_order)
    idxs = _collect_indices(obj)
    if not idxs:
        return obj, None

    if all(i in presented_set for i in idxs):
        return obj, None

    k = len(presented_order)

    # 1-based ordinal
    if all(isinstance(i, int) and 1 <= i <= k for i in idxs):
        mapping = {i: presented_order[i - 1] for i in range(1, k + 1)}
        return _apply_index_mapping(obj, mapping), "ordinal_1_based"

    # 0-based ordinal
    if all(isinstance(i, int) and 0 <= i < k for i in idxs):
        mapping = {i: presented_order[i] for i in range(0, k)}
        return _apply_index_mapping(obj, mapping), "ordinal_0_based"

    return obj, None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="batch run dir containing batch_id.txt and map.jsonl")
    ap.add_argument("--batch_id", default="", help="override batch id (else read run_dir/batch_id.txt)")
    ap.add_argument("--poll", action="store_true", help="poll until completed/failed")
    ap.add_argument("--poll_s", type=int, default=30)
    return ap.parse_args()


def main():
    a = parse_args()
    run_dir = Path(a.run_dir)

    batch_id = a.batch_id.strip()
    if not batch_id:
        batch_id = (run_dir / "batch_id.txt").read_text(encoding="utf-8").strip()

    map_path = run_dir / "map.jsonl"
    id_to_meta = load_map(map_path)

    # Load PassC validator helpers from existing script
    passc_script = Path("/home/ubuntu/resse-core/fm_teacher/scripts/eval_passC_prompt_v0.py")
    spec = importlib.util.spec_from_file_location("pc", str(passc_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import validator module: {passc_script}")
    pc = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(pc)  # type: ignore

    client = OpenAI()

    status = None
    output_file_id = None

    while True:
        b = client.batches.retrieve(batch_id)
        status = _get(b, "status", None)
        output_file_id = _get(b, "output_file_id", None)
        error_file_id = _get(b, "error_file_id", None)
        bid = _get(b, "id", None)
        print("batch_id", bid, "status", status, "output_file_id", output_file_id, "error_file_id", error_file_id)

        if not a.poll:
            break
        if status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(a.poll_s)

    if status != "completed":
        print("NOT_COMPLETED; nothing to collect")
        return

    if not output_file_id:
        print("MISSING output_file_id")
        return

    out_dir = run_dir / "collected"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_txt = extract_batch_output_text(client, str(output_file_id))
    raw_out_path = out_dir / "output.jsonl"
    raw_out_path.write_text(out_txt, encoding="utf-8")
    print("wrote_raw_output", str(raw_out_path))

    ok_path = out_dir / "ok.jsonl"
    bad_path = out_dir / "bad.jsonl"

    ok = 0
    bad = 0

    with ok_path.open("w", encoding="utf-8") as okf, bad_path.open("w", encoding="utf-8") as badf:
        for ln in out_txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue

            try:
                env = json.loads(ln)
            except Exception as e:
                bad += 1
                badf.write(json.dumps(
                    {"error": f"batch_env_json_parse_error:{type(e).__name__}", "raw_line": ln[:500]},
                    ensure_ascii=False
                ) + "\n")
                continue

            custom_id = env.get("custom_id") or ""
            meta = id_to_meta.get(custom_id)
            if not meta:
                bad += 1
                badf.write(json.dumps({"custom_id": custom_id, "error": "missing_custom_id_mapping"}, ensure_ascii=False) + "\n")
                continue

            domain = meta.get("domain")
            cluster_id = meta.get("cluster_id")
            facts = meta.get("facts") or []
            if not isinstance(facts, list):
                facts = []

            presented_order: List[int] = []
            facts_by_i: Dict[int, str] = {}
            for f in facts:
                if not isinstance(f, dict):
                    continue
                i = f.get("i")
                t = f.get("text")
                if isinstance(i, int):
                    presented_order.append(i)
                    if isinstance(t, str):
                        facts_by_i[i] = t

            k = int(meta.get("k", 6) or 6)

            resp = env.get("response") or {}
            status_code = None
            body = None
            if isinstance(resp, dict):
                status_code = resp.get("status_code")
                body = resp.get("body")
            err = env.get("error")

            if err or status_code != 200:
                bad += 1
                badf.write(json.dumps({
                    "domain": domain,
                    "cluster_id": cluster_id,
                    "custom_id": custom_id,
                    "facts": facts,
                    "raw": "",
                    "obj": None,
                    "validation_errors": [f"batch_status:{status_code}", f"batch_error:{err}"],
                }, ensure_ascii=False) + "\n")
                continue

            text = extract_response_output_text(body)
            try:
                obj = json.loads(text) if text else None
            except Exception as e:
                obj = None
                parse_err = f"model_json_parse_error:{type(e).__name__}:{str(e)[:160]}"
            else:
                parse_err = ""

            if not isinstance(obj, dict):
                bad += 1
                badf.write(json.dumps({
                    "domain": domain,
                    "cluster_id": cluster_id,
                    "custom_id": custom_id,
                    "facts": facts,
                    "raw": text,
                    "obj": None,
                    "validation_errors": [parse_err or "empty_or_nonobject_model_output"],
                }, ensure_ascii=False) + "\n")
                continue

            obj2, remap_mode = maybe_remap_indices(obj, presented_order)

            try:
                errs = pc.validate_obj(obj2, k=k, presented_idx=presented_order, facts_text_by_i=facts_by_i)
            except Exception as e:
                errs = [f"validator_exception:{type(e).__name__}:{str(e)[:160]}"]

            rec = {
                "domain": domain,
                "cluster_id": cluster_id,
                "custom_id": custom_id,
                "facts": facts,
                "raw": text,
                "obj": obj2,
                "validation_errors": errs,
            }
            if remap_mode:
                rec["index_remap_mode"] = remap_mode

            if errs:
                bad += 1
                badf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                ok += 1
                okf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"COLLECT_DONE ok={ok} bad={bad} out_dir={out_dir}")


if __name__ == "__main__":
    main()

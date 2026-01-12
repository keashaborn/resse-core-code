#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/resse-core/fm_teacher"
VENV="/home/ubuntu/resse-core/venv/bin/activate"

PASSM_PY="$ROOT/scripts/passM_materialize_v1.py"
PASSMPRIME_PY="$ROOT/scripts/passMprime_normalize_v0.py"
PASSXLINK0_PY="$ROOT/scripts/passX_link_v0.py"
PASSXLINK1_PY="$ROOT/scripts/passX_link_v1.py"
PASSG_PY="$ROOT/scripts/passG_merge_v0.py"
PASSC_PY="$ROOT/scripts/eval_passC_prompt_v0.py"

PASSC_DIR=""
PASSX_RUN_DIR=""
BASE_DIR=""
RUN_TAG="iterX"
N_EXPECTED=2000
WAIT=1
SLEEP_S=60

usage() {
  cat <<USAGE
Usage: $0 --passc_dir DIR --passx_run_dir DIR [--base_dir DIR] [--tag TAG] [--n_expected N] [--wait 0|1] [--sleep_s SECONDS]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --passc_dir) PASSC_DIR="$2"; shift 2;;
    --passx_run_dir) PASSX_RUN_DIR="$2"; shift 2;;
    --base_dir) BASE_DIR="$2"; shift 2;;
    --tag) RUN_TAG="$2"; shift 2;;
    --n_expected) N_EXPECTED="$2"; shift 2;;
    --wait) WAIT="$2"; shift 2;;
    --sleep_s) SLEEP_S="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$PASSC_DIR" || -z "$PASSX_RUN_DIR" ]]; then
  echo "ERROR: --passc_dir and --passx_run_dir are required" >&2
  exit 2
fi

source "$VENV"
cd "$ROOT"
mkdir -p "$ROOT/loop_logs"

# lock (same lock file as iter script, so we never overlap)
LOCK_DIR="$ROOT/.locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="$LOCK_DIR/run_field_iter_v0.lock"
exec 9>"$LOCK_FILE"
flock 9
echo "LOCK_ACQUIRED $LOCK_FILE"

if [[ -z "$BASE_DIR" ]]; then
  BASE_DIR="$(readlink -f "$ROOT/field/_current/MERGED")"
fi

OK_PATH="$PASSC_DIR/ok.jsonl"
BAD_PATH="$PASSC_DIR/bad.jsonl"
CLUSTERS_DIR="$PASSX_RUN_DIR/clusters_by_domain_passX"

echo "PASSC_DIR=$PASSC_DIR"
export PASSC_DIR
echo "PASSX_RUN_DIR=$PASSX_RUN_DIR"
echo "CLUSTERS_DIR=$CLUSTERS_DIR"
echo "BASE_DIR=$BASE_DIR"
echo "RUN_TAG=$RUN_TAG"
echo "N_EXPECTED=$N_EXPECTED"

if [[ "$WAIT" == "1" ]]; then
  echo "WAITING for PassC completion..."
  while true; do
    ok=0; bad=0
    [[ -f "$OK_PATH" ]] && ok=$(wc -l < "$OK_PATH" || echo 0)
    [[ -f "$BAD_PATH" ]] && bad=$(wc -l < "$BAD_PATH" || echo 0)
    done=$((ok+bad))

    if [[ "$done" -ge "$N_EXPECTED" ]]; then
      # also wait for PassC process to be gone (avoid writing while we salvage)
      if pgrep -af "eval_passC_prompt_v0.py" >/dev/null 2>&1; then
        echo "done=$done >= $N_EXPECTED but PassC process still running; sleeping ${SLEEP_S}s"
        sleep "$SLEEP_S"
        continue
      fi
      echo "PassC looks complete: done=$done (ok=$ok bad=$bad)"
      break
    fi

    if ! pgrep -af "eval_passC_prompt_v0.py" >/dev/null 2>&1; then
      echo "ERROR: PassC not running but done=$done < $N_EXPECTED (incomplete). Aborting finalize." >&2
      exit 12
    fi

    echo "progress done=$done / $N_EXPECTED (ok=$ok bad=$bad); sleeping ${SLEEP_S}s"
    sleep "$SLEEP_S"
  done
fi

# -------------------------
# Deterministic salvage (cheap): move fixable bad rows into ok.jsonl
# -------------------------
python3 - <<'PY'
import json, pathlib, importlib.util
import os
import re

NEG = re.compile(r"\b(not|no|never|cannot|can't|does not|do not|is not|isn't|aren't|without)\b", re.I)

PASSC_DIR = pathlib.Path(os.environ["PASSC_DIR"])

SCRIPT = "/home/ubuntu/resse-core/fm_teacher/scripts/eval_passC_prompt_v0.py"

spec = importlib.util.spec_from_file_location("pc", SCRIPT)
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

ok_path  = PASSC_DIR/"ok.jsonl"
bad_path = PASSC_DIR/"bad.jsonl"

def read_jsonl(p: pathlib.Path):
    if not p.exists():
        return []
    out=[]
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln=ln.strip()
        if not ln: continue
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

ok_rows = read_jsonl(ok_path)
bad_rows = read_jsonl(bad_path)

ok_ids = set()
for r in ok_rows:
    cid = r.get("cluster_id")
    if cid: ok_ids.add(cid)

def fix_support_endpoints(obj: dict) -> int:
    edges = obj.get("edge_candidates") or []
    if not isinstance(edges, list): return 0
    changed=0
    for e in edges:
        if not isinstance(e, dict): continue
        si, di = e.get("src_i"), e.get("dst_i")
        sup = e.get("support_i_list")
        if not isinstance(sup, list): continue
        new=[]
        for x in sup:
            if isinstance(x, int) and x not in new:
                new.append(x)
        if isinstance(si, int) and si not in new:
            new.insert(0, si)
        if isinstance(di, int) and di not in new:
            new.insert(1 if (new and new[0]==si) else 0, di)
        if len(new) > 3:
            keep=[]
            for x in new:
                if x not in keep:
                    keep.append(x)
                if len(keep) == 3:
                    break
            new = keep
        if new != sup:
            e["support_i_list"] = new
            changed += 1
    return changed

def downgrade_contradicts(obj: dict, facts_text_by_i: dict[int,str]) -> int:
    edges = obj.get("edge_candidates") or []
    if not isinstance(edges, list):
        return 0
    changed = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("rel_type") != "contradicts":
            continue
        si = e.get("src_i")
        di = e.get("dst_i")
        if not isinstance(si, int) or not isinstance(di, int):
            continue
        src_txt = (facts_text_by_i.get(si, "") or "")
        dst_txt = (facts_text_by_i.get(di, "") or "")
        # If contradicts isn't explicit (negation cues), downgrade to refines
        if not (NEG.search(src_txt) or NEG.search(dst_txt)):
            e["rel_type"] = "refines"
            changed += 1
    return changed

def downgrade_depends_on(obj: dict, facts_text_by_i: dict[int,str]) -> int:
    edges = obj.get("edge_candidates") or []
    if not isinstance(edges, list): return 0
    changed=0
    for e in edges:
        if not isinstance(e, dict): continue
        if e.get("rel_type") != "depends_on": continue
        di = e.get("dst_i")
        if not isinstance(di, int): continue
        dst_txt = (facts_text_by_i.get(di,"") or "")
        if not pc.DEP_CUES.search(dst_txt):
            e["rel_type"] = "refines"
            changed += 1
    return changed

fixed = 0
kept_bad = []

for r in bad_rows:
    cid = r.get("cluster_id")
    if cid in ok_ids:
        continue

    raw = r.get("raw") or ""
    obj = r.get("obj")
    if not isinstance(obj, dict):
        try:
            obj = json.loads(raw) if raw else None
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        kept_bad.append(r)
        continue

    facts = r.get("facts") or []
    facts_by_i = {}
    for f in facts:
        if isinstance(f, dict) and isinstance(f.get("i"), int):
            facts_by_i[f["i"]] = f.get("text","") or ""
    presented = list(facts_by_i.keys())

    errs = pc.validate_obj(obj, 6, presented, facts_by_i)

    changed = 0
    changed += fix_support_endpoints(obj)
    changed += downgrade_contradicts(obj, facts_by_i)
    changed += downgrade_depends_on(obj, facts_by_i)
    if changed:
        errs = pc.validate_obj(obj, 6, presented, facts_by_i)

    r["obj"] = obj
    r["validation_errors"] = errs

    if not errs:
        ok_rows.append(r)
        if cid: ok_ids.add(cid)
        fixed += 1
    else:
        kept_bad.append(r)

ok_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ok_rows) + ("\n" if ok_rows else ""), encoding="utf-8")
bad_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in kept_bad) + ("\n" if kept_bad else ""), encoding="utf-8")
print(f"SALVAGE fixed={fixed} ok={len(ok_rows)} bad={len(kept_bad)}")
PY

# -------------------------
# PassM -> PassM' -> PassX-link0 -> PassX-link1 (score) -> PassG merge
# -------------------------
ts() { date +%Y%m%d_%H%M%S; }

PASSM_LOG="$ROOT/loop_logs/finalize_passM_${RUN_TAG}_$(ts).log"
python3 "$PASSM_PY" --passc_eval_dir "$PASSC_DIR" --run_tag "passM_v1_${RUN_TAG}" | tee "$PASSM_LOG"
PASSM_DIR="$(awk '/^WROTE /{p=$2} END{print p}' "$PASSM_LOG")"
echo "PASSM_DIR=$PASSM_DIR"

PASSMPRIME_LOG="$ROOT/loop_logs/finalize_passMprime_${RUN_TAG}_$(ts).log"
python3 "$PASSMPRIME_PY" --in_dir "$PASSM_DIR" --run_tag "passMprime_${RUN_TAG}" | tee "$PASSMPRIME_LOG"
PASSMPRIME_DIR="$(awk '/^WROTE /{p=$2} END{print p}' "$PASSMPRIME_LOG")"
echo "PASSMPRIME_DIR=$PASSMPRIME_DIR"

PASSXLINK0_LOG="$ROOT/loop_logs/finalize_passXlink0_${RUN_TAG}_$(ts).log"
python3 "$PASSXLINK0_PY" --passx_run_dir "$PASSX_RUN_DIR" --exp_field_dir "$PASSMPRIME_DIR" --run_tag "passXlink_${RUN_TAG}" | tee "$PASSXLINK0_LOG"
LINK_DIR="$(awk '/^WROTE /{p=$2} END{print p}' "$PASSXLINK0_LOG")"
echo "LINK_DIR=$LINK_DIR"

PASSXLINK1_LOG="$ROOT/loop_logs/finalize_passXlink1_${RUN_TAG}_$(ts).log"
python3 "$PASSXLINK1_PY" --merged_field_dir "$BASE_DIR" --cross_in "$LINK_DIR/concept_edges_cross.jsonl" --run_tag "passXlink_v1_${RUN_TAG}" | tee "$PASSXLINK1_LOG"
SCORED_DIR="$(awk '/^WROTE /{p=$2} END{print p}' "$PASSXLINK1_LOG")"
echo "SCORED_DIR=$SCORED_DIR"

PASSG_LOG="$ROOT/loop_logs/finalize_passG_${RUN_TAG}_$(ts).log"
python3 "$PASSG_PY" \
  --base_dir "$BASE_DIR" \
  --exp_dir "$PASSMPRIME_DIR" \
  --cross_edges "$LINK_DIR/concept_edges_cross.jsonl" \
  --cross_edges_scored "$SCORED_DIR/concept_edges_cross_scored.jsonl" \
  --run_tag "passG_${RUN_TAG}" \
  | tee "$PASSG_LOG"

MERGED_DIR="$(awk '/^WROTE /{p=$2} END{print p}' "$PASSG_LOG")"
echo "MERGED_DIR=$MERGED_DIR"

# -------------------------
# Update _current pointers
# -------------------------
mkdir -p "$ROOT/field/_current" "$ROOT/evals/_current"
ln -sfn "../$(basename "$PASSC_DIR")"        "$ROOT/evals/_current/PASSC_LATEST"
ln -sfn "../$(basename "$PASSMPRIME_DIR")"  "$ROOT/field/_current/EXP_LATEST"
ln -sfn "../$(basename "$LINK_DIR")"        "$ROOT/field/_current/LINKS_LATEST"
ln -sfn "../$(basename "$MERGED_DIR")"      "$ROOT/field/_current/MERGED"

# Manifest
MANIFEST="$MERGED_DIR/manifest.json"
python3 - <<PY
import json, pathlib
m = {
  "base_dir": "$BASE_DIR",
  "passx_run_dir": "$PASSX_RUN_DIR",
  "clusters_dir": "$CLUSTERS_DIR",
  "passc_dir": "$PASSC_DIR",
  "passm_dir": "$PASSM_DIR",
  "passmprime_dir": "$PASSMPRIME_DIR",
  "link_dir": "$LINK_DIR",
  "scored_dir": "$SCORED_DIR",
  "merged_dir": "$MERGED_DIR",
  "run_tag": "$RUN_TAG",
  "n_expected": $N_EXPECTED,
}
pathlib.Path("$MANIFEST").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("WROTE_MANIFEST", "$MANIFEST")
PY

echo "FINALIZE_DONE $(date -Is)"

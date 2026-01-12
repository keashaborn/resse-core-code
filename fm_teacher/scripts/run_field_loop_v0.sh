#!/usr/bin/env bash
set -euo pipefail

# Loop/daemon wrapper for run_field_iter_v0.sh
# - one loop at a time (flock)
# - waits for idle if other pipeline processes are running
# - retries iterations with exponential backoff on failure
# - appends an iterations.jsonl ledger for auditability

ROOT="/home/ubuntu/resse-core/fm_teacher"
VENV="/home/ubuntu/resse-core/venv/bin/activate"
ITER_SH="$ROOT/scripts/run_field_iter_v0.sh"

# Defaults
N_CONCEPTS=2000
QUERIES_PER_CONCEPT=3
HITS_PER_QUERY=15
MAX_MEMBERS=30
TAG_PREFIX="iter"
SLEEP_S=60
MAX_ITERS=0          # 0 = infinite
MAX_FAILS=3
BACKOFF_S=300
MAX_BACKOFF_S=3600
IDLE_CHECK_S=30

usage() {
  cat <<USAGE
Usage: $0 [args]
  --n_concepts N
  --queries_per_concept N
  --hits_per_query N
  --max_members N
  --tag_prefix STR
  --sleep_s SECONDS
  --max_iters N         (0 = infinite)
  --max_fails N
  --backoff_s SECONDS
  --max_backoff_s SECONDS
  --idle_check_s SECONDS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_concepts) N_CONCEPTS="$2"; shift 2;;
    --queries_per_concept) QUERIES_PER_CONCEPT="$2"; shift 2;;
    --hits_per_query) HITS_PER_QUERY="$2"; shift 2;;
    --max_members) MAX_MEMBERS="$2"; shift 2;;
    --tag_prefix) TAG_PREFIX="$2"; shift 2;;
    --sleep_s) SLEEP_S="$2"; shift 2;;
    --max_iters) MAX_ITERS="$2"; shift 2;;
    --max_fails) MAX_FAILS="$2"; shift 2;;
    --backoff_s) BACKOFF_S="$2"; shift 2;;
    --max_backoff_s) MAX_BACKOFF_S="$2"; shift 2;;
    --idle_check_s) IDLE_CHECK_S="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

mkdir -p "$ROOT/.locks" "$ROOT/loop_logs"

LOOP_LOCK="$ROOT/.locks/run_field_loop_v0.lock"
exec 8>"$LOOP_LOCK"
if ! flock -n 8; then
  echo "ERROR: loop already running (lock: $LOOP_LOCK)" >&2
  exit 11
fi

trap 'echo "LOOP_STOP $(date -Is)"; exit 0' SIGINT SIGTERM

wait_for_idle() {
  while true; do
    # If anything pipeline-like is running, wait. This prevents clobbering _current pointers.
    if pgrep -af "scripts/(run_field_iter_v0\.sh|eval_passC_prompt_v0\.py|passX_expand_mini_v0\.py|passM_materialize_v1\.py|passG_merge_v0\.py)" >/dev/null 2>&1; then
      echo "WAIT_IDLE $(date -Is) pipeline_processes_detected"
      pgrep -af "scripts/(run_field_iter_v0\.sh|eval_passC_prompt_v0\.py|passX_expand_mini_v0\.py|passM_materialize_v1\.py|passG_merge_v0\.py)" | tail -n 5 || true
      sleep "$IDLE_CHECK_S"
      continue
    fi
    return 0
  done
}

get_from_log() {
  local key="$1" log="$2"
  # matches lines like KEY=/path
  awk -F= -v k="$key" '$0 ~ ("^"k"=") {sub("^"k"=",""); print $0}' "$log" | tail -n 1
}

fail_count=0
iter_count=0
backoff="$BACKOFF_S"

echo "LOOP_START $(date -Is) tag_prefix=$TAG_PREFIX n_concepts=$N_CONCEPTS qpc=$QUERIES_PER_CONCEPT hpq=$HITS_PER_QUERY max_members=$MAX_MEMBERS"

while true; do
  wait_for_idle

  iter_count=$((iter_count + 1))
  run_tag="${TAG_PREFIX}_$(date +%Y%m%d_%H%M%S)"
  iter_log="$ROOT/loop_logs/iter_${run_tag}.log"
  ts_start="$(date -Is)"

  {
    echo "=== ITER_START $ts_start tag=$run_tag ==="
    echo "params n_concepts=$N_CONCEPTS queries_per_concept=$QUERIES_PER_CONCEPT hits_per_query=$HITS_PER_QUERY max_members=$MAX_MEMBERS"
  } | tee -a "$iter_log"

  set +e
  bash "$ITER_SH" \
    --n_concepts "$N_CONCEPTS" \
    --queries_per_concept "$QUERIES_PER_CONCEPT" \
    --hits_per_query "$HITS_PER_QUERY" \
    --max_members "$MAX_MEMBERS" \
    --tag "$run_tag" \
    2>&1 | tee -a "$iter_log"
  rc=${PIPESTATUS[0]}
  set -e

  ts_end="$(date -Is)"
  merged="$(readlink -f "$ROOT/field/_current/MERGED" 2>/dev/null || true)"
  passc="$(readlink -f "$ROOT/evals/_current/PASSC_LATEST" 2>/dev/null || true)"

  passx_run_dir="$(get_from_log PASSX_RUN_DIR "$iter_log")"
  clusters_dir="$(get_from_log CLUSTERS_DIR "$iter_log")"
  passc_dir="$(get_from_log PASSC_DIR "$iter_log")"
  passm_dir="$(get_from_log PASSM_DIR "$iter_log")"
  passmprime_dir="$(get_from_log PASSMPRIME_DIR "$iter_log")"
  link_dir="$(get_from_log LINK_DIR "$iter_log")"
  scored_dir="$(get_from_log SCORED_DIR "$iter_log")"
  merged_dir="$(get_from_log MERGED_DIR "$iter_log")"

  nodes=0; edges=0; seeds=0
  if [[ -n "$merged" && -f "$merged/concept_nodes.jsonl" ]]; then
    nodes=$(wc -l < "$merged/concept_nodes.jsonl" || echo 0)
    edges=$(wc -l < "$merged/concept_edges.jsonl" || echo 0)
    seeds=$(wc -l < "$merged/retrieval_seeds.jsonl" || echo 0)
  fi

  python3 - <<PY >> "$ROOT/loop_logs/iterations.jsonl"
import json
rec = {
  "ts_start": "$ts_start",
  "ts_end": "$ts_end",
  "run_tag": "$run_tag",
  "exit_code": $rc,
  "params": {
    "n_concepts": $N_CONCEPTS,
    "queries_per_concept": $QUERIES_PER_CONCEPT,
    "hits_per_query": $HITS_PER_QUERY,
    "max_members": $MAX_MEMBERS,
  },
  "current": {
    "merged": "$merged",
    "passc_latest": "$passc",
  },
  "dirs": {
    "passx_run_dir": "$passx_run_dir",
    "clusters_dir": "$clusters_dir",
    "passc_dir": "$passc_dir",
    "passm_dir": "$passm_dir",
    "passmprime_dir": "$passmprime_dir",
    "link_dir": "$link_dir",
    "scored_dir": "$scored_dir",
    "merged_dir": "$merged_dir",
  },
  "counts": {
    "nodes": $nodes,
    "edges": $edges,
    "seeds": $seeds,
  },
}
print(json.dumps(rec, ensure_ascii=False))
PY

  echo "=== ITER_END $ts_end tag=$run_tag rc=$rc nodes=$nodes edges=$edges seeds=$seeds ===" | tee -a "$iter_log"

  if [[ $rc -eq 0 ]]; then
    fail_count=0
    backoff="$BACKOFF_S"
  else
    fail_count=$((fail_count + 1))
    echo "ITER_FAIL tag=$run_tag rc=$rc fail_count=$fail_count/$MAX_FAILS backoff_s=$backoff" | tee -a "$iter_log"
    if [[ $fail_count -ge $MAX_FAILS ]]; then
      echo "LOOP_ABORT too_many_failures fail_count=$fail_count" | tee -a "$iter_log"
      exit 2
    fi
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    if [[ $backoff -gt $MAX_BACKOFF_S ]]; then
      backoff="$MAX_BACKOFF_S"
    fi
    continue
  fi

  if [[ $MAX_ITERS -gt 0 && $iter_count -ge $MAX_ITERS ]]; then
    echo "LOOP_DONE max_iters_reached iter_count=$iter_count" | tee -a "$iter_log"
    exit 0
  fi

  sleep "$SLEEP_S"
done

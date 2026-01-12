#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SLEEP_S="${SLEEP_S:-60}"

# Pick the python PassC process (ignore wrapper bash/tee)
PASSC_PID="$(pgrep -af 'eval_passC_prompt_v1_parallel\.py' | awk '$2 ~ /^python/ {print $1}' | tail -n 1 || true)"
if [[ -z "${PASSC_PID}" ]]; then
  echo "NO_PASSC_PYTHON_PID"
  exit 0
fi

echo "PASSC_PID=${PASSC_PID}"
ps -p "${PASSC_PID}" -o pid,etime,stat,nlwp,pcpu,pmem,cmd --width 240

OK_PATH="$(lsof -p "${PASSC_PID}" 2>/dev/null | awk '/ok\.jsonl$/ {print $NF; exit}')"
BAD_PATH="$(lsof -p "${PASSC_PID}" 2>/dev/null | awk '/bad\.jsonl$/ {print $NF; exit}')"

# Fallback if lsof output is noisy/unavailable
if [[ -z "${OK_PATH}" || -z "${BAD_PATH}" ]]; then
  PASSC_DIR="$(ls -1dt /home/ubuntu/resse-core/fm_teacher/evals/passC_prompt_v1_parallel_* 2>/dev/null | head -n 1 || true)"
  OK_PATH="${PASSC_DIR}/ok.jsonl"
  BAD_PATH="${PASSC_DIR}/bad.jsonl"
fi

echo "OK_PATH=${OK_PATH}"
echo "BAD_PATH=${BAD_PATH}"

if [[ ! -f "${OK_PATH}" || ! -f "${BAD_PATH}" ]]; then
  echo "MISSING_OK_OR_BAD"
  exit 1
fi

counts() {
  local ok bad
  ok="$(wc -l < "${OK_PATH}")"
  bad="$(wc -l < "${BAD_PATH}")"
  echo "${ok} ${bad} $((ok+bad))"
}

read -r ok1 bad1 done1 < <(counts)
echo "t1=$(date -Is) done1=${done1} ok1=${ok1} bad1=${bad1}"

if [[ "${MODE}" == "--rate60" ]]; then
  sleep "${SLEEP_S}"
  read -r ok2 bad2 done2 < <(counts)
  echo "t2=$(date -Is) done2=${done2} ok2=${ok2} bad2=${bad2} delta_done=$((done2-done1)) per_min=$((done2-done1))"
fi

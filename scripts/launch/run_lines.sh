#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/run_lines.sh LINES --model SERVED [--jobs J]"
  echo "Runs every line of LINES (printed by experiments/<x>.py lines) against the servers that serve.sh started"
  echo "for SERVED, at most J lines at a time. Finished records are skipped, so an interrupted file can be rerun."
  exit 2
}

[ $# -ge 3 ] || usage
LINES=$1; shift
SERVED=""; JOBS=4
while [ $# -gt 0 ]; do
  case $1 in
    --model) SERVED=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$SERVED" ] || usage
[ "${BASH_VERSINFO[0]}${BASH_VERSINFO[1]}" -ge 43 ] || { echo "run_lines.sh needs bash 4.3 or newer (wait -n)"; exit 2; }
[ -f "$LINES" ] || { echo "no such file: $LINES"; exit 2; }
URLS_FILE=${RCI_RESULTS:-./results}/.urls/$SERVED.txt
if [ -z "${RCI_BASE_URLS:-}" ]; then
  [ -f "$URLS_FILE" ] || { echo "no servers for $SERVED: run scripts/launch/serve.sh start $SERVED first"; exit 2; }
  export RCI_BASE_URLS=$(cat "$URLS_FILE")
fi
LOGS=${RCI_LOGS:-./logs}/runs
mkdir -p "$LOGS"

FAIL=0; n=0; running=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case $line in "#"*) continue ;; esac
  n=$((n + 1))
  echo "[run_lines] $n: $line"
  bash -c "$line" > "$LOGS/${SERVED}_line$n.log" 2>&1 &
  running=$((running + 1))
  if [ $running -ge "$JOBS" ]; then
    wait -n || FAIL=1
    running=$((running - 1))
  fi
done < "$LINES"
while [ $running -gt 0 ]; do
  wait -n || FAIL=1
  running=$((running - 1))
done
echo "[run_lines] $n lines finished, failures=$FAIL"
exit $FAIL

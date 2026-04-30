#!/bin/bash
# submit_sort.sh — Build a queue of HD-MEA recordings and submit them to
# Slurm for kilosort4 processing via mxw-sort.
#
# Each recording in --input is enqueued (one .todo marker per file) into a
# fresh queue directory under $MYSCRATCH/queues/. A Slurm array job is then
# submitted; each task runs ./mxw_sort.sbatch as a worker that atomically
# claims files from the queue and processes them.
#
# Usage:
#   submit_sort.sh --input <dir> [--gpus N] [--output <spike_depot>]
#
# Arguments:
#   --input    Directory containing .h5 / .raw.h5 recordings to sort.
#   --gpus     Number of parallel workers (= array tasks). Default 1.
#   --output   Override default spike_depot location.
#              Default: $MYSCRATCH/spike_depot.
#
# Environment:
#   MYSCRATCH  Per-user scratch root. Defaults to /fastscratch/myscratch/$USER.
#
# The launcher always queues every input file. The worker uses
# `mxw-sort --skip-existing` to no-op already-completed wells, so re-running
# on a fully-sorted directory is fast and idempotent.

set -uo pipefail

N_GPUS=1
IN_DIR=""
SPIKE_DEPOT="${MYSCRATCH:-/fastscratch/myscratch/$USER}/spike_depot"
VENV="$HOME/mxw-sort/.venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)   N_GPUS=$2;      shift 2 ;;
    --input)  IN_DIR=$2;      shift 2 ;;
    --output) SPIKE_DEPOT=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -z "$IN_DIR" ] && { echo "ERROR: --input is required" >&2; exit 1; }
[ -d "$IN_DIR" ] || { echo "ERROR: input dir not found: $IN_DIR" >&2; exit 1; }

if [ -z "${MYSCRATCH:-}" ]; then
  MYSCRATCH="/fastscratch/myscratch/$USER"
fi

# Build a fresh queue directory for this submission. Workers atomically move
# .todo markers from todo/ to claimed/, then to done/ once finished.
QUEUE_DIR="$MYSCRATCH/queues/sort_$(date +%s%N)"
TODO_DIR="$QUEUE_DIR/todo"
CLAIMED_DIR="$QUEUE_DIR/claimed"
DONE_DIR="$QUEUE_DIR/done"
mkdir -p "$TODO_DIR" "$CLAIMED_DIR" "$DONE_DIR"

echo "Scanning $IN_DIR for recordings…"
n_queued=0

while IFS= read -r f; do
  bn=$(basename "$f")
  stem="${bn%.raw.h5}"
  stem="${stem%.h5}"

  # Report current state for visibility, but always queue. The worker's
  # --skip-existing flag handles already-sorted wells idempotently.
  wells_n=$(find "$SPIKE_DEPOT/$stem" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  done_n=$(find "$SPIKE_DEPOT/$stem" -mindepth 3 -maxdepth 3 -name "spike_times.npy" 2>/dev/null | wc -l)
  echo "$f" > "$TODO_DIR/${stem}.todo"
  printf "  QUEUED:  %s  (%d/%d wells already sorted)\n" "$stem" "$done_n" "$wells_n"
  (( n_queued++ )) || true
done < <(find "$IN_DIR" -maxdepth 1 -type f \( -name "*.raw.h5" -o -name "*.h5" \) | sort)

echo
echo "Queued: $n_queued files"

if [ "$n_queued" -eq 0 ]; then
  echo "Nothing to do."
  rm -rf "$QUEUE_DIR"
  exit 0
fi

JOB_ID=$(sbatch \
  --array=0-$((N_GPUS-1)) \
  --export=ALL,TODO_DIR="$TODO_DIR",CLAIMED_DIR="$CLAIMED_DIR",DONE_DIR="$DONE_DIR",SPIKE_DEPOT="$SPIKE_DEPOT",VENV="$VENV" \
  "$SCRIPT_DIR/mxw_sort.sbatch" | awk '{print $NF}')

echo
echo "Submitted job array: $JOB_ID"
echo "Tasks:               $N_GPUS"
echo "Queue dir:           $QUEUE_DIR"
echo "Logs:                $MYSCRATCH/logs/${JOB_ID}_*.{out,err}"

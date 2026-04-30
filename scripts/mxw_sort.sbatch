#!/bin/bash
# mxw_sort.sbatch — Slurm worker for HD-MEA spike sorting.
#
# Each array task acts as a worker that:
#   1. atomically claims a recording from $TODO_DIR by `mv`-ing its marker
#      to $CLAIMED_DIR (filesystem-level mutex);
#   2. validates the .h5 with h5py to catch corrupt source files;
#   3. runs mxw-sort, which invokes kilosort4 on each well in the file;
#   4. moves the marker to $DONE_DIR with a .done / .failed / .corrupt suffix;
#   5. deletes per-well preprocessed/traces.bin to keep scratch usage bounded.
#
# Multiple tasks pull from the same queue, so any number of array tasks
# can drain the same workload without coordination beyond the rename-mv.
#
# Required exported variables (set by submit_sort.sh):
#   TODO_DIR, CLAIMED_DIR, DONE_DIR, SPIKE_DEPOT, VENV, MYSCRATCH
#
# Design notes:
#   - `set -e` is intentionally NOT used. We've observed transient
#     filesystem hiccups on /fastscratch that can spuriously fail single
#     commands; with -e they would take down the entire array task and
#     orphan its in-flight claim. Each per-file step has explicit error
#     handling instead, and an EXIT trap moves any in-flight claim to
#     .failed so it gets reported rather than disappearing into claimed/.
#   - traces.bin is deleted from each well after kilosort finishes
#     (success or failure) because it can be regenerated from the source h5
#     and would otherwise dominate scratch usage (~3 GB / well).
#   - The --constraint pins the job to high-VRAM GPUs (A100 80 GB / L40S 46 GB).
#     Tesla V100 32 GB nodes OOM on this workload (kilosort peaks near 25 GB).

#SBATCH --job-name=mxw_sort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-80gb|gpu-46gb"
#SBATCH --exclude=compute-128
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/fastscratch/myscratch/%u/logs/%A_%a.out
#SBATCH --error=/fastscratch/myscratch/%u/logs/%A_%a.err

set -uo pipefail

if [ -z "${MYSCRATCH:-}" ] || [ ! -d "$MYSCRATCH" ]; then
  echo "ERROR: MYSCRATCH not set or /fastscratch not mounted on $(hostname)" >&2
  exit 1
fi

for var in TODO_DIR CLAIMED_DIR DONE_DIR SPIKE_DEPOT VENV; do
  [ -n "${!var:-}" ] || { echo "ERROR: $var not set" >&2; exit 1; }
done
[ -d "$TODO_DIR" ] || { echo "ERROR: TODO_DIR not found: $TODO_DIR" >&2; exit 1; }

export HDF5_PLUGIN_PATH=$HOME/lib/hdf5/plugins
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

ts() { date +%H:%M:%S; }
echo "[$(ts)] Task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

source "$VENV/bin/activate"

# If we exit unexpectedly with a file in flight, mark it failed so the next
# launcher run can re-queue cleanly rather than orphaning the marker.
claimed=""
cleanup_on_exit() {
  local rc=$?
  if [ -n "${claimed:-}" ] && [ -e "$claimed" ]; then
    echo "[$(ts)] EXIT trap (rc=$rc): marking $(basename "$claimed").failed"
    mv "$claimed" "$DONE_DIR/$(basename "$claimed").failed" 2>/dev/null || true
  fi
}
trap cleanup_on_exit EXIT

# Remove preprocessed binary intermediates from any well whose kilosort
# output completed. The .bin is regenerable from the source h5 and is
# typically ~3 GB per well; leaving it would quickly blow scratch quota.
prune_bins() {
  local outdir="$1"
  [ -d "$outdir" ] || return 0
  for well in "$outdir"*/; do
    [ -d "$well" ] || continue
    if [ -f "${well}ks4/spike_times.npy" ] && [ -f "${well}preprocessed/traces.bin" ]; then
      rm -f "${well}preprocessed/traces.bin" 2>/dev/null || true
    fi
  done
}

# Move the in-flight claim to .failed and bump the failure counter.
mark_failed() {
  mv "$claimed" "$DONE_DIR/$(basename "$claimed").failed" 2>/dev/null || true
  claimed=""
  files_failed=$((files_failed+1))
}

files_processed=0
files_skipped=0
files_failed=0

while true; do
  claimed=""

  # Atomic claim: rename a .todo marker into claimed/. Whichever task wins
  # the rename owns the file.
  for todo in "$TODO_DIR"/*.todo; do
    [ -e "$todo" ] || break
    target="$CLAIMED_DIR/$(basename "$todo")"
    if mv "$todo" "$target" 2>/dev/null; then
      claimed="$target"
      break
    fi
  done
  [ -z "$claimed" ] && break

  f=$(cat "$claimed" 2>/dev/null) || {
    echo "[$(ts)] cat failed on $claimed"
    mark_failed; continue
  }
  bn=$(basename "$f")
  stem="${bn%.raw.h5}"; stem="${stem%.h5}"

  OUTDIR="$SPIKE_DEPOT/$stem"
  TMPDIR="$MYSCRATCH/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${stem}"
  export TMPDIR

  if ! mkdir -p "$OUTDIR" "$TMPDIR" 2>/dev/null; then
    echo "[$(ts)] mkdir failed for $stem"
    mark_failed; continue
  fi

  echo "========================================"
  echo "[$(ts)] Task $SLURM_ARRAY_TASK_ID | File $((files_processed+files_failed+files_skipped+1))"
  echo "Input:  $f"
  echo "Output: $OUTDIR"
  echo "========================================"

  # Discard incomplete wells from prior aborted runs so --skip-existing
  # doesn't see them as "done".
  echo "[$(ts)] Pruning incomplete wells in $OUTDIR…"
  while IFS= read -r well; do
    if ! find "$well" -mindepth 2 -maxdepth 2 -name "spike_times.npy" 2>/dev/null | grep -q .; then
      echo "  removing incomplete: $(basename "$well")"
      rm -rf "$well" 2>/dev/null || true
    fi
  done < <(find "$OUTDIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

  # Validate h5 readability before paying the kilosort startup cost.
  echo "[$(ts)] Validating $f"
  if ! VALIDATE_F="$f" python -c "
import h5py, sys, os
f = os.environ['VALIDATE_F']
try:
    with h5py.File(f, 'r') as fh:
        if not list(fh.keys()):
            print('EMPTY: no keys'); sys.exit(1)
        sys.exit(0)
except Exception as e:
    print(f'CORRUPT: {e}'); sys.exit(1)
" 2>&1; then
    echo "[$(ts)] $f is corrupt; skipping"
    echo "$f" >> "$MYSCRATCH/logs/corrupt_files_${SLURM_ARRAY_JOB_ID}.txt" 2>/dev/null || true
    mv "$claimed" "$DONE_DIR/$(basename "$claimed").corrupt" 2>/dev/null || true
    claimed=""; files_skipped=$((files_skipped+1)); continue
  fi

  if ! mxw-sort "$f" --out "$OUTDIR" --flat --skip-existing --dur-s 0; then
    echo "[$(ts)] mxw-sort failed on $f"
    echo "$f" >> "$MYSCRATCH/logs/failed_files_${SLURM_ARRAY_JOB_ID}.txt" 2>/dev/null || true
    prune_bins "$OUTDIR/"
    rm -rf "$TMPDIR" 2>/dev/null || true
    mark_failed; continue
  fi

  n_complete=$(find "$OUTDIR" -mindepth 3 -maxdepth 3 -name "spike_times.npy" 2>/dev/null | wc -l)
  echo "[$(ts)] Completed wells: $n_complete"

  prune_bins "$OUTDIR/"
  mv "$claimed" "$DONE_DIR/$(basename "$claimed").done" 2>/dev/null || true
  rm -rf "$TMPDIR" 2>/dev/null || true
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  claimed=""; files_processed=$((files_processed+1))
done

# Normal exit: nothing should be in flight, so disarm the trap.
claimed=""
echo "========================================"
echo "[$(ts)] Task $SLURM_ARRAY_TASK_ID done"
echo "  processed:         $files_processed"
echo "  skipped (corrupt): $files_skipped"
echo "  failed:            $files_failed"

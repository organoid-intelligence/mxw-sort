#!/bin/bash
# run_batch.sh — Local-side pipeline driver for HD-MEA spike sorting on JHPCE.
#
# For each batch in $MANIFEST: rsyncs the listed files (with rename) to
# fastscratch on the cluster, triggers submit_sort.sh, polls until the
# Slurm array job finishes, rsyncs ks4/ outputs back to a local directory,
# then deletes the batch from the cluster. A ledger file tracks completed
# batches so reruns are safely resumable.
#
# Usage:
#   run_batch.sh <batch_id>    # run a single batch
#   run_batch.sh --all         # iterate every batch in the manifest
#
# Manifest format ($MANIFEST is a CSV with header):
#   batch,size_bytes,target_name,source_path
#     batch        — integer batch id (groups files for one upload/sort cycle)
#     size_bytes   — informational
#     target_name  — flat filename to use on the cluster (must be unique)
#     source_path  — absolute path to the source .h5 on the local drive
#
# Output layout (under $LOCAL_OUT):
#   <stem>/<wellNNN>/<ks4 files>
# where <stem> is target_name with .raw.h5/.h5 stripped.
#
# The configuration block below should be edited per project. Copy this
# script into a new pipeline directory and adjust PIPE_DIR, LOCAL_OUT, etc.

set -uo pipefail

# Configuration. Edit per project.

# Local pipeline state directory (manifest, ledger, per-batch logs).
PIPE_DIR="$HOME/domoic_pipeline"

# Where to deposit kilosort output on this machine. Mirror under here:
#   <stem>/<wellNNN>/<ks4 files>
LOCAL_OUT="/media/achinn/Elements/User_Storage/AChinn/DOMOIC_ROOT/spike_sorted"

# Cluster-side paths. h5_warehouse must already exist (typically under
# fastscratch). spike_depot is where mxw-sort writes per-well ks4/ outputs.
HPC_WAREHOUSE="/fastscratch/myscratch/achinn/h5_warehouse"
HPC_DEPOT="/fastscratch/myscratch/achinn/spike_depot"

# SSH aliases (defined in ~/.ssh/config). The transfer host is used for
# bulk rsync; the login host is used for submit_sort.sh and squeue polling.
# Both should have ControlMaster enabled to amortise connection overhead.
SSH_LOGIN="jhpce03"
SSH_TRANSFER="jhpce-transfer"

# Number of parallel Slurm array tasks. Each task uses one GPU.
N_GPUS=4


MANIFEST="$PIPE_DIR/manifest.csv"
LEDGER="$PIPE_DIR/ledger.txt"
LOG_DIR="$PIPE_DIR/logs"
mkdir -p "$LOCAL_OUT" "$LOG_DIR"
touch "$LEDGER"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# Process one batch end-to-end. Five steps, each labelled in the log:
#   1. upload   — rsync each file in this batch to $HPC_WAREHOUSE, applying
#                 the rename from manifest.target_name (so files from a
#                 nested source layout end up flat and uniquely-named on
#                 the cluster);
#   2. submit   — invoke submit_sort.sh remotely; capture the job ID;
#   3. wait     — poll squeue every 2 minutes until the job leaves the queue,
#                 emitting a wells-done count every 12 minutes for visibility;
#   4. offload  — rsync back the ks4/ contents per well into $LOCAL_OUT;
#   5. cleanup  — delete this batch's h5s and spike_depot dirs from the
#                 cluster, plus stale queue/tmp scratch.
#
# Both ssh and rsync invocations inside `while read` loops use < /dev/null
# or `ssh -n` to keep them from consuming the manifest stdin we're iterating.
run_batch() {
  local bid=$1
  local logfile="$LOG_DIR/batch_${bid}.log"

  if grep -qx "batch_${bid}=done" "$LEDGER"; then
    echo "[$(ts)] batch $bid already done — skipping"
    return 0
  fi

  exec > >(tee -a "$logfile") 2>&1
  echo "[$(ts)] ============================================="
  echo "[$(ts)] === BATCH $bid START"
  echo "[$(ts)] ============================================="

  local n_files
  n_files=$(awk -F, -v b="$bid" 'NR>1 && $1==b' "$MANIFEST" | wc -l)
  echo "[$(ts)] $n_files files in batch"

  # Step 1: upload
  echo "[$(ts)] STEP 1: upload"
  local i=0
  while IFS=, read -r batch sz target src; do
    [ "$batch" = "$bid" ] || continue
    i=$((i+1))
    printf "[%s]   [%d/%d] %s\n" "$(ts)" "$i" "$n_files" "$target"
    if ! rsync -a --partial "$src" "$SSH_TRANSFER:$HPC_WAREHOUSE/$target" < /dev/null; then
      echo "[$(ts)] UPLOAD FAILED on $target"
      return 1
    fi
  done < <(tail -n +2 "$MANIFEST")
  echo "[$(ts)] uploads done"

  # Step 2: submit
  echo "[$(ts)] STEP 2: submit sort"
  local submit_out jobid
  submit_out=$(ssh "$SSH_LOGIN" "bash ~/mxw-sort/jobs/submit_sort.sh --input $HPC_WAREHOUSE --gpus $N_GPUS" 2>&1)
  jobid=$(echo "$submit_out" | awk '/Submitted job array:/{print $NF}')
  if [ -z "$jobid" ]; then
    echo "[$(ts)] SUBMIT FAILED"
    echo "$submit_out"
    return 1
  fi
  echo "[$(ts)] job $jobid submitted"

  # Step 3: wait
  echo "[$(ts)] STEP 3: wait for $jobid"
  local poll=0
  while true; do
    local state
    state=$(ssh "$SSH_LOGIN" "squeue -h -u $USER -j $jobid -o '%T' 2>/dev/null | sort -u | tr '\n' '/'")
    if [ -z "$state" ]; then
      echo "[$(ts)] $jobid left queue"
      break
    fi
    poll=$((poll+1))
    if [ $((poll % 6)) -eq 0 ]; then
      local wells
      wells=$(ssh "$SSH_LOGIN" "grep -hc '\[DONE\] well' /fastscratch/myscratch/$USER/logs/${jobid}_*.out 2>/dev/null | paste -sd+ | bc")
      echo "[$(ts)] still running ($state); wells done: ${wells:-0}"
    fi
    sleep 120
  done

  # Step 4: offload
  echo "[$(ts)] STEP 4: offload ks4 outputs to $LOCAL_OUT/"
  local offloaded=0
  while IFS=, read -r batch sz target src; do
    [ "$batch" = "$bid" ] || continue
    local stem="${target%.raw.h5}"; stem="${stem%.h5}"

    local wells_list
    wells_list=$(ssh -n "$SSH_LOGIN" "ls -d $HPC_DEPOT/$stem/well*/ 2>/dev/null | xargs -n1 basename")
    [ -z "$wells_list" ] && { echo "[$(ts)]   no output for $stem (file failed)"; continue; }

    for well in $wells_list; do
      mkdir -p "$LOCAL_OUT/$stem/$well"
      if rsync -a "$SSH_TRANSFER:$HPC_DEPOT/$stem/$well/ks4/" "$LOCAL_OUT/$stem/$well/" < /dev/null; then
        offloaded=$((offloaded+1))
      else
        echo "[$(ts)]   WARNING: rsync failed for $stem/$well"
      fi
    done
  done < <(tail -n +2 "$MANIFEST")
  echo "[$(ts)] offloaded $offloaded wells"

  # Step 5: cleanup
  echo "[$(ts)] STEP 5: clean up cluster"
  while IFS=, read -r batch sz target src; do
    [ "$batch" = "$bid" ] || continue
    local stem="${target%.raw.h5}"; stem="${stem%.h5}"
    ssh -n "$SSH_LOGIN" "rm -f $HPC_WAREHOUSE/$target; rm -rf $HPC_DEPOT/$stem" 2>/dev/null
  done < <(tail -n +2 "$MANIFEST")

  ssh "$SSH_LOGIN" "find /fastscratch/myscratch/$USER/queues -mindepth 1 -delete 2>/dev/null; \
                    find /fastscratch/myscratch/$USER/tmp    -mindepth 1 -delete 2>/dev/null"

  echo "batch_${bid}=done" >> "$LEDGER"
  echo "[$(ts)] === BATCH $bid COMPLETE ==="
  return 0
}


if [ "${1:-}" = "--all" ]; then
  max_b=$(awk -F, 'NR>1{print $1}' "$MANIFEST" | sort -un | tail -1)
  for b in $(seq 0 "$max_b"); do
    if ! run_batch "$b"; then
      echo "[$(ts)] BATCH $b FAILED — stopping pipeline"
      exit 1
    fi
  done
  echo "[$(ts)] ALL BATCHES COMPLETE"
else
  run_batch "${1:?need batch id (or --all)}"
fi

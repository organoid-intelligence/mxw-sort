#!/bin/bash
# cout.sh — Quick completeness report for a spike_depot directory.
#
# Walks $MYSCRATCH/spike_depot/* and labels each stem as COMPLETE,
# PARTIAL, or INCOMPLETE based on per-well presence of spike_times.npy
# at depth 3 (i.e. <stem>/<wellNNN>/ks4/spike_times.npy).
#
# Usage:
#   cout.sh

set -uo pipefail

MYSCRATCH="${MYSCRATCH:-/fastscratch/myscratch/$USER}"

for d in "$MYSCRATCH"/spike_depot/*/; do
  stem=$(basename "$d")
  wells=$(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)
  matches=$(find "$d" -mindepth 3 -maxdepth 3 -name "spike_times.npy" | wc -l)

  if [ "$matches" -eq 0 ]; then
    printf 'INCOMPLETE  %-40s no kilosort output\n' "$stem"
  elif [ "$matches" -lt "$wells" ]; then
    printf 'PARTIAL     %-40s %d / %d wells\n' "$stem" "$matches" "$wells"
  else
    printf 'COMPLETE    %-40s %d / %d wells\n' "$stem" "$matches" "$wells"
  fi
done

# scripts/

Operational scripts for running mxw-sort on a Slurm cluster (developed against
JHPCE) and driving end-to-end batches from a local workstation.

| Script | Where it runs | What it does |
|---|---|---|
| `submit_sort.sh` | cluster login node | Builds a file-claim queue from a directory of `.h5` recordings and submits a Slurm array job. |
| `mxw_sort.sbatch` | cluster compute node | Per-task worker invoked by the array job. Atomically claims files, validates them, runs `mxw-sort`, and prunes the regenerable `traces.bin` intermediate. |
| `cout.sh` | cluster login node | Quick completeness report over `$MYSCRATCH/spike_depot/`. |
| `run_batch.sh` | local workstation | End-to-end driver: rsync up → submit → wait → rsync ks4 outputs back → clean up cluster. Iterates batches from a CSV manifest, with a ledger for resumability. |

Each script has a header docstring with usage, arguments, and design notes.
`run_batch.sh` has a clearly-marked configuration block at the top to edit
per project (paths, SSH aliases, GPU count).

## Manifest format consumed by `run_batch.sh`

CSV with header:

```
batch,size_bytes,target_name,source_path
0,11318175209,Run22_DIV16_0p1uM_000236.raw.h5,/path/to/source/data.raw.h5
0,5653685920,Run22_DIV16_0p1uM_000243.raw.h5,/path/to/another/data.raw.h5
1,...
```

- `batch` — integer id; files in the same batch upload, sort, and offload together
- `target_name` — flat filename to use on the cluster (must be unique across the manifest)
- `source_path` — absolute path on the local machine

The intent of `target_name` is to flatten nested source layouts where many
recordings collide on the same `data.raw.h5` filename. Encode the
project's hierarchy (run/DIV/concentration/etc.) into the target name so
each file is uniquely addressable on the cluster.

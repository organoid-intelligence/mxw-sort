# mxw-sort

Pipeline for Maxwell `.raw.h5` recordings: SpikeInterface preprocessing, Kilosort4 spike sorting, and QC output generation.

## Requirements

- Python 3.10–3.12
- CUDA-capable GPU (for Kilosort4)

## Installation

```bash
git clone https://github.com/chinan3/mxw-sort && cd mxw-sort
uv sync        # or: pip install -e .
```

## Usage

### Single file

```bash
mxw-sort path/to/data.raw.h5 --out results/
```

### Directory (batch)

Point at a root directory to recursively find and process all `.h5` files. Output mirrors the input directory structure.

```bash
mxw-sort run path/to/dataset_root/ --out results/
```

```
dataset_root/                   results/
  experiment1/data.raw.h5   →    experiment1/well000/  well001/ ...
  experiment2/data.raw.h5   →    experiment2/well000/  well001/ ...
```

### Dry run

Preview what would be processed without doing any work:

```bash
mxw-sort run path/to/data.raw.h5 --out results/ --dry-run
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output root folder |
| `--start-s` | `0.0` | Start time in seconds |
| `--dur-s` | `30.0` | Duration to process (seconds). `0` = full file |
| `--wells` | `auto` | Wells to process: `auto`, a range (`0-5`), or a list (`0,2,4`) |
| `--only-well` | — | Process exactly one well index |
| `--bp-min` | `300.0` | Bandpass filter minimum frequency (Hz) |
| `--bp-max-frac-nyq` | `0.9` | Bandpass max as fraction of Nyquist |
| `--remove-stim-artifacts / --no-remove-stim-artifacts` | `False` | Remove stimulation artifacts before the bandpass and sorting |
| `--stim-artifact-mode` | `fit` | Correction mode: `fit` (subtract model) or `blank` (noise fill) |
| `--stim-source` | `auto` | Stim-time source: `auto`, `events` (h5 log), `npz` (sidecar), or `detect` (from signal) |
| `--ks4-highpass-cutoff` | `1.0` | KS4 highpass cutoff (`1.0` = disabled) |
| `--ks4-batch-size` | `60000` | KS4 batch size |
| `--skip-existing / --no-skip-existing` | `True` | Skip wells with existing KS4 outputs |
| `--dry-run` | `False` | Print actions without executing |
| `--flat` | `False` | Run on flat input directories where all .h5 files are uniquely named and in the same folder (non-recursive) |

## Pipeline stages

For each well in an `.h5` file:

1. **Read** — Load Maxwell recording via SpikeInterface (`MaxwellRecordingExtractor`)
2. **Preprocess** — Convert unsigned→signed, slice time window, bandpass filter
3. **Export** — Write binary traces + probe geometry JSON (for Kilosort)
4. **Sort** — Run Kilosort4
5. **QC** — Generate raster plot, spike position scatter, drift scatter, and a summary JSON

When `--remove-stim-artifacts` is set, stimulation windows are cleaned during preprocessing, before the bandpass filter (see below).

## Stimulation artifact removal

Recordings that contain electrical stimulation carry a stimulus artifact that can seed false units and corrupt template estimation. Pass `--remove-stim-artifacts` to clean each stimulation window before the bandpass and Kilosort. The step is off by default; only the samples inside each stimulation window change, and everything else is byte-for-byte identical to a run without the flag.

Stimulation times are read from the MaxWell events log when present, then from a `*_metadata.npz` sidecar (the `X_times` schedule, used by the reservoir-computing recordings), and otherwise detected from the signal; `--stim-source` forces a specific one. Two correction modes are available through `--stim-artifact-mode`:

- `fit` (default): fits and subtracts the artifact (an exponential decay plus a cubic baseline), leaving spikes that ride the decay intact. Use this when you need spike responses close to the stimulus.
- `blank`: replaces the artifact window with matched noise. Cheaper on recordings with very many stimulations.

The mode, source, stimulation count, and window sizes are recorded in each well's `meta.json`.

## Output structure

```
<out_root>/
  well000/
    preprocessed/
      traces.bin          # Binary recording
      ks4_probe.json      # Probe geometry for KS4
      channel_xy.npy      # Channel positions
      meta.json           # Processing metadata
    ks4/
      spike_times.npy     # KS4 outputs
      spike_clusters.npy
      ...
    qc/
      raster.png
      spike_positions.png
      drift_scatter.png
      qc_summary.json
  well001/
    ...
```

## License

GNU General Public License v3.0

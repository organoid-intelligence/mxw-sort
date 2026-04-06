import json
from pathlib import Path

import numpy as np
import spikeinterface.core as sc

DEFAULT_DTYPE = "int16"
DEFAULT_CHUNK_DURATION = "1s"


def write_binary(
    rec,
    bin_path: Path,
    dtype: str = DEFAULT_DTYPE,
    chunk_duration: str = DEFAULT_CHUNK_DURATION,
):
    sc.write_binary_recording(
        rec,
        file_paths=str(bin_path),
        dtype=dtype,
        n_jobs=1,
        chunk_duration=chunk_duration,
        progress_bar=True,
    )


def write_probe_json(xy: np.ndarray, probe_path: Path):
    xy = np.asarray(xy)
    n_chan = int(xy.shape[0])
    probe = {
        "chanMap": np.arange(n_chan, dtype=int).tolist(),
        "xc": xy[:, 0].astype(float).tolist(),
        "yc": xy[:, 1].astype(float).tolist(),
        "kcoords": np.zeros(n_chan, dtype=int).tolist(),
        "n_chan": n_chan,
    }
    probe_path.write_text(json.dumps(probe, indent=2))


def write_meta_json(meta: dict, meta_path: Path):
    meta_path.write_text(json.dumps(meta, indent=2))

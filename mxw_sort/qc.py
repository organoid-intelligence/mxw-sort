from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_npy(p: Path) -> np.ndarray | None:
    if p.exists():
        return np.load(p, allow_pickle=False)
    return None


def write_qc(
    ks_dir: Path,
    qc_dir: Path,
    fs_hz: float,
    dur_s_processed: float | None = None,
    max_units_raster: int = 80,
    max_spikes_per_unit: int = 500,
) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)

    spike_times = _load_npy(ks_dir / "spike_times.npy")
    spike_clusters = _load_npy(ks_dir / "spike_clusters.npy")
    amplitudes = _load_npy(ks_dir / "amplitudes.npy")
    spike_positions = _load_npy(ks_dir / "spike_positions.npy")

    if spike_times is None or spike_clusters is None:
        raise FileNotFoundError("Missing spike_times.npy or spike_clusters.npy in KS output.")

    spike_times = spike_times.reshape(-1)  # KS stores as (N, 1)
    spike_clusters = spike_clusters.reshape(-1)

    t_s = spike_times.astype(np.float64) / float(fs_hz)
    if dur_s_processed is None:
        dur_s_processed = float(np.max(t_s)) if t_s.size else 0.0

    unit_ids = np.unique(spike_clusters)
    n_units = int(unit_ids.size)

    fr_hz = {}
    for u in unit_ids:
        n = int(np.sum(spike_clusters == u))
        fr_hz[int(u)] = (n / dur_s_processed) if dur_s_processed > 0 else 0.0

    summary = {
        "n_units": n_units,
        "n_spikes": int(spike_times.size),
        "duration_s": float(dur_s_processed),
        "fs_hz": float(fs_hz),
        "has_amplitudes": bool(amplitudes is not None),
        "has_spike_positions": bool(spike_positions is not None),
        "unit_firing_rate_hz_first10": {k: fr_hz[k] for k in list(fr_hz.keys())[:10]},
    }
    (qc_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    chosen_units = unit_ids
    if n_units > max_units_raster:
        chosen_units = rng.choice(unit_ids, size=max_units_raster, replace=False)
        chosen_units = np.sort(chosen_units)

    unit_to_y = {int(u): i for i, u in enumerate(chosen_units)}

    xs = []
    ys = []
    for u in chosen_units:
        mask = spike_clusters == u
        ts = t_s[mask]
        if ts.size > max_spikes_per_unit:
            ts = rng.choice(ts, size=max_spikes_per_unit, replace=False)
        xs.append(ts)
        ys.append(np.full(ts.size, unit_to_y[int(u)], dtype=np.int32))

    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
    else:
        x = np.array([])
        y = np.array([])

    plt.figure()
    plt.plot(x, y, linestyle="None", marker=".", markersize=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Unit (subset)")
    plt.title("Raster (subsampled)")
    plt.tight_layout()
    plt.savefig(qc_dir / "raster.png", dpi=200)
    plt.close()

    if spike_positions is not None:
        spike_positions = np.asarray(spike_positions)
        if spike_positions.ndim == 2 and spike_positions.shape[1] >= 2:
            x_um = spike_positions[:, 0]
            y_um = spike_positions[:, 1]

            N = t_s.size
            keep = np.arange(N)
            if N > 200_000:
                keep = rng.choice(keep, size=200_000, replace=False)
            keep = np.sort(keep)

            plt.figure()
            plt.plot(x_um[keep], y_um[keep], linestyle="None", marker=".", markersize=1)
            plt.xlabel("x (um)")
            plt.ylabel("y (um)")
            plt.title("Spike positions (subsampled)")
            plt.tight_layout()
            plt.savefig(qc_dir / "spike_positions.png", dpi=200)
            plt.close()

            plt.figure()
            if amplitudes is not None and amplitudes.size == t_s.size:
                amp = np.asarray(amplitudes).reshape(-1)
                s = 2 + 8 * (amp - np.min(amp)) / (np.ptp(amp) + 1e-9)
                plt.scatter(t_s[keep], y_um[keep], s=s[keep], marker=".")
            else:
                plt.plot(t_s[keep], y_um[keep], linestyle="None", marker=".", markersize=1)
            plt.xlabel("Time (s)")
            plt.ylabel("y (um)")
            plt.title("Drift scatter (subsampled)")
            plt.tight_layout()
            plt.savefig(qc_dir / "drift_scatter.png", dpi=200)
            plt.close()

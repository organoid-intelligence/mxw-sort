import h5py
import numpy as np

from .io_maxwell import _resolve_rec_group

_DEDUP_GAP = 5  # frames; merge the multiple events logged per pulse


def load_stim_frames(h5_path: str, stream: str) -> np.ndarray | None:
    """Stim onset sample indices (recording coordinate) from the events log, or
    None if there is no usable log."""
    with h5py.File(h5_path, "r") as f:
        rg = _resolve_rec_group(f, stream)
        if "events" not in rg:
            return None
        ev = rg["events"][()]
        names = ev.dtype.names or ()
        if ev.shape[0] == 0 or "eventtype" not in names or "frameno" not in names:
            return None
        fr = np.sort(ev[ev["eventtype"] == 1]["frameno"].astype(np.int64))
        if fr.size == 0:
            return None
        frame_nos = rg["groups/routed/frame_nos"]
        f0 = int(frame_nos[0])
        n_fr = frame_nos.shape[0]
        # contiguous frames map by a fixed offset; dropped frames need searchsorted
        if int(frame_nos[-1]) - f0 == n_fr - 1:
            fn_full = None
        else:
            fn_full = frame_nos[()]
    keep = np.concatenate([[True], np.diff(fr) > _DEDUP_GAP])
    onsets = fr[keep]
    if fn_full is None:
        return onsets - f0
    return np.searchsorted(fn_full, onsets).astype(np.int64)


def merge_windows(onsets, pre: int, post: int) -> list[tuple[int, int]]:
    """Build [onset-pre, onset+post] windows and merge overlaps. Returns sorted
    list of (lo, hi) with lo clipped at 0."""
    onsets = np.sort(np.asarray(onsets, dtype=np.int64))
    out: list[tuple[int, int]] = []
    for o in onsets:
        lo, hi = max(0, int(o) - pre), int(o) + post
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


_CHUNK = 200000


def detect_stim_frames(recording, cfg):
    """Detect stim onsets from the signal: samples where many channels are
    saturated (railed flatline) at once. Returns onset sample indices (recording
    coordinate); empty if none found."""
    # local import to avoid an artifacts <-> stim_times import cycle
    from .artifacts import flatline_runs

    n = recording.get_num_samples()
    sample = recording.get_traces(start_frame=0, end_frame=min(n, _CHUNK)).astype(
        np.int64
    )
    lo, hi = int(sample.min()), int(sample.max())
    onsets = []
    for s in range(0, n, _CHUNK):
        e = min(n, s + _CHUNK)
        x = recording.get_traces(start_frame=s, end_frame=e).T.astype(
            np.int64
        )  # (C, L)
        flat = flatline_runs(x, cfg.sat_flat_run)
        railed = (x <= lo + 2) | (x >= hi - 2)
        live = flat.mean(axis=1) < 0.5  # drop persistently-flat dead channels
        sat = flat & railed & live[:, None]
        nsat = sat.sum(axis=0)
        quorum = max(3, int(cfg.detect_quorum_frac * int(live.sum())))
        hot = np.where(nsat >= quorum)[0]
        onsets.extend((hot + s).tolist())
    if not onsets:
        return np.array([], dtype=np.int64)
    onsets = np.array(sorted(onsets), dtype=np.int64)
    keep = np.concatenate([[True], np.diff(onsets) > 2000])  # cluster within 0.2 s
    return onsets[keep]


def resolve_stim_frames(h5_path, stream, recording, cfg):
    """Resolve stim onsets from the events log, falling back to blind detection.
    Returns (frames, source); empty with source 'none' if neither finds any."""
    if cfg.source in ("auto", "events"):
        fr = load_stim_frames(h5_path, stream)
        if fr is not None and len(fr):
            return fr, "events"
        if cfg.source == "events":
            return np.array([], dtype=np.int64), "events_none"
    if cfg.source in ("auto", "detect"):
        fr = detect_stim_frames(recording, cfg)
        if len(fr):
            return fr, "detect"
    return np.array([], dtype=np.int64), "none"

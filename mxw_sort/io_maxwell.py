import h5py
import numpy as np
import spikeinterface.core as sc


def _resolve_rec_group(f, stream_name):
    """Handle both `wells/wellXXX/rec0000/...` and `wells/wellXXX/...` layouts."""
    well = f[f"wells/{stream_name}"]
    rec_keys = sorted(k for k in well.keys() if k.startswith("rec"))
    if rec_keys:
        return well[rec_keys[0]]
    if "settings" in well:
        return well
    raise KeyError(f"Cannot find recording data under wells/{stream_name}")


class _H5Segment(sc.BaseRecordingSegment):
    """Lazy recording segment backed by an h5py dataset."""

    def __init__(self, raw_dataset, n_samples, sampling_frequency):
        super().__init__(sampling_frequency=sampling_frequency)
        self._raw = raw_dataset
        self._n_samples = n_samples
        self._2d = raw_dataset.ndim == 2

    def get_num_samples(self):
        return self._n_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        start = start_frame or 0
        end = end_frame if end_frame is not None else self._n_samples
        if self._2d:
            traces = self._raw[:, start:end].T  # (n_ch, n_samp) -> (n_samp, n_ch)
        else:
            n_ch = self._raw.shape[0] // self._n_samples
            traces = self._raw[start * n_ch : end * n_ch].reshape(-1, n_ch)
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces


class MaxwellH5Recording(sc.BaseRecording):
    """Bypasses Neo's channel-uniqueness check that fails on some Maxwell files."""

    def __init__(self, h5_path: str, stream_name: str):
        self._h5_file = h5py.File(h5_path, "r")
        rec_group = _resolve_rec_group(self._h5_file, stream_name)

        mapping = rec_group["settings/mapping"][()]
        fs = float(rec_group["settings/sampling"][()].item())
        raw = rec_group["groups/routed/raw"]

        if raw.ndim == 2:
            n_channels_raw = raw.shape[0]
            n_samples = raw.shape[1]
        else:
            n_channels_raw = mapping.shape[0]
            n_samples = raw.shape[0] // n_channels_raw

        # channels dataset holds channel IDs (matching mapping["channel"]),
        # not array indices — look up by channel ID
        channels_ds = rec_group["groups/routed"].get("channels")
        if channels_ds is not None:
            chan_ids = channels_ds[()].astype(int)
            chan_to_row = {int(m["channel"]): i for i, m in enumerate(mapping)}
            row_idx = np.array([chan_to_row[c] for c in chan_ids])
            xy = np.column_stack([
                mapping["x"][row_idx].astype(float),
                mapping["y"][row_idx].astype(float),
            ])
        else:
            xy = np.column_stack([
                mapping["x"].astype(float),
                mapping["y"].astype(float),
            ])

        channel_ids = np.arange(n_channels_raw)
        super().__init__(sampling_frequency=fs, channel_ids=channel_ids, dtype=raw.dtype)

        segment = _H5Segment(raw, n_samples, fs)
        self.add_recording_segment(segment)
        self.set_channel_locations(xy)

    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file:
            self._h5_file.close()


def read_maxwell(h5_path: str, stream_name: str):
    """h5py reader that avoids Neo's duplicate-channel-ID error."""
    return MaxwellH5Recording(h5_path, stream_name)


def get_available_wells(h5_path: str) -> tuple[int, ...]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "wells" not in f:
                return (0, 1, 2, 3, 4, 5)
            wells = []
            for name in f["wells"]:
                if name.startswith("well"):
                    try:
                        wells.append(int(name[4:]))
                    except ValueError:
                        continue
            return tuple(sorted(wells)) if wells else (0, 1, 2, 3, 4, 5)
    except Exception:
        return (0, 1, 2, 3, 4, 5)


def get_well_duration_s(h5_path: str, stream_name: str) -> float:
    with h5py.File(h5_path, "r") as f:
        rec_group = _resolve_rec_group(f, stream_name)
        raw = rec_group["groups/routed/raw"]
        fs = float(rec_group["settings/sampling"][()].item())
        if raw.ndim == 2:
            n_samples = raw.shape[1]
        else:
            n_channels = rec_group["settings/mapping"].shape[0]
            n_samples = raw.shape[0] // n_channels
        return n_samples / fs

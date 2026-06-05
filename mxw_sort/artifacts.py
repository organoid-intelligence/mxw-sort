import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from .stim_times import merge_windows


def flatline_runs(x: np.ndarray, min_run: int) -> np.ndarray:
    """Boolean mask of samples that lie within a run of >= min_run identical
    consecutive values along the last axis."""
    x = np.asarray(x)
    if x.ndim == 1:
        return flatline_runs(x[None, :], min_run)[0]
    C, L = x.shape
    out = np.zeros((C, L), dtype=bool)
    k = int(min_run)
    if k <= 1 or L < k:
        return out
    same = x[:, 1:] == x[:, :-1]  # (C, L-1): equal to previous
    starts = sliding_window_view(same, k - 1, axis=1).all(axis=2)  # (C, L-k+1)
    # starts[c, s] True => x[c, s:s+k] are all equal; mark those k samples
    m = starts.shape[1]
    for off in range(k):
        out[:, off : off + m] |= starts
    return out


def detect_saturation(
    region: np.ndarray,
    baseline: np.ndarray,
    sigma: np.ndarray,
    min_run: int,
    dev_sigma: float,
) -> np.ndarray:
    """Flag samples that are both in a flatline run and far from the channel
    baseline (amplifier saturation). region (C, L); baseline and sigma (C,)."""
    flat = flatline_runs(region, min_run)
    far = np.abs(region - baseline[:, None]) > (dev_sigma * sigma[:, None])
    return flat & far


def estimate_sigma(block: np.ndarray) -> np.ndarray:
    """Per-channel robust noise std (1.4826 * MAD), floored. block (C, L)."""
    block = np.asarray(block, dtype=np.float64)
    med = np.median(block, axis=1, keepdims=True)
    mad = np.median(np.abs(block - med), axis=1)
    return np.maximum(1.4826 * mad, 1e-6)


def noise_fill(
    sigma: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Matched per-channel Gaussian noise, (C, n_samples) float32."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return (rng.standard_normal((sigma.shape[0], n_samples)) * sigma[:, None]).astype(
        np.float32
    )


def fit_exp(t: np.ndarray, x: np.ndarray, x0: float):
    """Constrained exponential decay a*exp(-b*t)+c, fixed to pass through
    (t[0], x0). Returns (model, ok)."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.size < 3:
        return np.full(t.size, x0, dtype=np.float64), False
    t0 = t[0]
    a0 = float(x[0] - np.median(x[-3:]))
    span = max(1.0, t[-1] - t[0])
    p0 = np.array([a0 if a0 != 0 else 1.0, 1.0 / (0.3 * span)])

    def resid(p):
        a, b = p
        return a * (np.exp(-b * (t - t0)) - 1.0) + x0 - x

    try:
        sol = least_squares(
            resid,
            p0,
            bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
            method="trf",
            max_nfev=200,
        )
        a, b = sol.x
        model = a * (np.exp(-b * (t - t0)) - 1.0) + x0
        return model, bool(sol.success)
    except Exception:
        return np.full(t.size, x0, dtype=np.float64), False


def fit_poly3(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Degree-3 OLS model evaluated on t."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    deg = min(3, max(1, t.size - 1))
    coeffs = np.polyfit(t, x, deg)
    return np.polyval(coeffs, t)


def accept_fit(
    model: np.ndarray, x: np.ndarray, sigma: float, M: int, eps: float
) -> bool:
    """Accept the fit when its mean-squared residual over the first M points is
    below eps * sigma^2."""
    m = min(M, model.shape[0])
    mse = float(np.mean((model[:m] - x[:m]) ** 2))
    return mse < eps * float(sigma) ** 2


def clean_window(
    wbuf: np.ndarray,
    onset_local: int,
    sigma: np.ndarray,
    cfg,
    fs: float,
    rng: np.random.Generator,
    fill_floor: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean one window buffer. wbuf (C, L): columns [:onset_local] are pre-stim
    (baseline), [onset_local:] is the artifact region. Returns (cleaned float32,
    status int8 per channel: 1=fit subtracted, 0=filled)."""
    wbuf = np.asarray(wbuf, dtype=np.float64)
    C, L = wbuf.shape
    cleaned = wbuf.astype(np.float32, copy=True)
    status = np.zeros(C, dtype=np.int8)
    sigma = np.asarray(sigma, dtype=np.float64)

    # window sizes in samples
    post_n = max(1, round(cfg.post_ms * fs / 1000.0))
    decay_n = max(2, round(cfg.fit_decay_ms * fs / 1000.0))
    max_n = L - onset_local
    if max_n <= 0:
        return cleaned, status

    if onset_local > 0:
        baseline = wbuf[:, :onset_local].mean(axis=1)  # (C,)
    else:
        baseline = wbuf[:, :1].mean(axis=1)

    post = wbuf[:, onset_local:]  # (C, max_n)
    sat = detect_saturation(post, baseline, sigma, cfg.sat_flat_run, cfg.sat_dev_sigma)
    sat_any = sat.any(axis=1)  # (C,) bool
    # index of last saturated sample per channel (undefined if not sat_any)
    last = max_n - 1 - sat[:, ::-1].argmax(axis=1)
    sat_end = np.where(sat_any, last + 1, 0)  # exclusive end of saturation
    fill_end = np.clip(np.maximum(np.maximum(post_n, sat_end), fill_floor), 0, max_n)

    # blank mode: replace [onset_local : onset_local+fill_end] with matched noise
    if cfg.mode == "blank":
        noise = noise_fill(sigma, max_n, rng).astype(np.float64) + baseline[:, None]
        cols = np.arange(max_n)[None, :]
        m = cols < fill_end[:, None]  # (C, max_n) mask
        block = cleaned[:, onset_local:].astype(np.float64)
        block[m] = noise[m]
        cleaned[:, onset_local:] = block.astype(np.float32)
        return cleaned, status

    # fit mode: per channel - try exp+poly3 subtraction, fall back to noise fill
    sat_frac = sat[:, :post_n].mean(axis=1) if post_n <= max_n else sat.mean(axis=1)
    for c in range(C):
        if sat_frac[c] > cfg.sat_frac_thresh:
            # heavily saturated -> fill with noise
            n = int(fill_end[c])
            cleaned[c, onset_local : onset_local + n] = rng.standard_normal(n).astype(
                np.float32
            ) * np.float32(sigma[c]) + np.float32(baseline[c])
            continue
        fit_len = min(post_n, max_n)
        x = wbuf[c, onset_local : onset_local + fit_len]
        t = np.arange(fit_len, dtype=np.float64)
        d = min(decay_n, fit_len)  # exp covers first d samples, poly3 the rest
        model = np.empty(fit_len, dtype=np.float64)
        m_exp, ok = fit_exp(t[:d], x[:d], x0=float(x[0]))
        model[:d] = m_exp
        if fit_len > d:
            model[d:] = fit_poly3(t[d:] - t[d], x[d:])
        if ok and accept_fit(model, x, sigma[c], cfg.fit_M, cfg.fit_eps):
            # subtract model, re-center on baseline
            cleaned[c, onset_local : onset_local + fit_len] = (
                x - model + baseline[c]
            ).astype(np.float32)
            status[c] = 1
        else:
            n = int(fill_end[c])
            cleaned[c, onset_local : onset_local + n] = rng.standard_normal(n).astype(
                np.float32
            ) * np.float32(sigma[c]) + np.float32(baseline[c])
    return cleaned, status


class RemoveStimArtifactsRecording(BasePreprocessor):
    """Lazy preprocessor that cleans stim windows and passes the rest through
    unchanged. Output dtype matches the parent recording."""

    def __init__(self, recording, stim_frames, cfg):
        out_dtype = (
            recording.get_dtype()
        )  # match parent so non-stim stays bit-identical
        BasePreprocessor.__init__(self, recording, dtype=out_dtype)
        fs = recording.get_sampling_frequency()
        stim_frames = np.asarray(stim_frames, dtype=np.int64)
        pre_n = max(1, round(cfg.pre_ms * fs / 1000.0))
        post_n = max(1, round(cfg.post_ms * fs / 1000.0))
        self._delta_n = max(1, round(cfg.delta_ms * fs / 1000.0))
        self._sat_max_n = max(post_n, round(cfg.sat_max_ms * fs / 1000.0))
        self._windows = merge_windows(stim_frames, pre_n, post_n)
        # per-channel noise sigma from the first second
        n0 = min(recording.get_num_samples(), int(fs))
        base = recording.get_traces(start_frame=0, end_frame=n0).T.astype(np.float64)
        self._sigma = (
            estimate_sigma(base)
            if base.shape[1]
            else np.ones(recording.get_num_channels())
        )
        for parent in recording._recording_segments:
            self.add_recording_segment(
                _Segment(
                    parent,
                    self._windows,
                    self._sigma,
                    self._delta_n,
                    self._sat_max_n,
                    cfg,
                    fs,
                    out_dtype,
                )
            )
        self._kwargs = dict(
            recording=recording, stim_frames=stim_frames.tolist(), cfg=cfg
        )


class _Segment(BasePreprocessorSegment):
    def __init__(self, parent, windows, sigma, delta_n, sat_max_n, cfg, fs, dtype):
        BasePreprocessorSegment.__init__(self, parent)
        self._windows = windows
        self._sigma = sigma
        self._delta_n = delta_n
        self._sat_max_n = sat_max_n
        self._cfg = cfg
        self._fs = fs
        self._dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        n = self.parent_recording_segment.get_num_samples()
        start = 0 if start_frame is None else start_frame
        end = n if end_frame is None else end_frame
        overlapping = [
            (lo, hi) for (lo, hi) in self._windows if hi > start and lo < end
        ]
        if not overlapping:
            out = self.parent_recording_segment.get_traces(start, end, None).astype(
                self._dtype
            )
            return out if channel_indices is None else out[:, channel_indices]

        ext_lo = max(0, min(start, min(lo - self._delta_n for lo, _ in overlapping)))
        ext_hi = min(
            n, max(end, max(max(lo + self._sat_max_n, hi) for lo, hi in overlapping))
        )
        buf = self.parent_recording_segment.get_traces(ext_lo, ext_hi, None).T.astype(
            np.float64
        )  # (C, L)
        for lo, hi in overlapping:
            onset_local = lo - ext_lo
            w_start = max(0, onset_local - self._delta_n)
            w_end = min(buf.shape[1], max(onset_local + self._sat_max_n, hi - ext_lo))
            sub = buf[:, w_start:w_end]
            rng = np.random.default_rng([self._cfg.seed, int(lo)])
            cleaned, _ = clean_window(
                sub,
                onset_local - w_start,
                self._sigma,
                self._cfg,
                self._fs,
                rng,
                fill_floor=hi - lo,
            )
            buf[:, w_start:w_end] = cleaned
        out = buf[:, (start - ext_lo) : (end - ext_lo)].T.astype(self._dtype)
        return out if channel_indices is None else out[:, channel_indices]


def remove_stim_artifacts(recording, stim_frames, cfg):
    """Wrap recording in RemoveStimArtifactsRecording."""
    return RemoveStimArtifactsRecording(recording, stim_frames, cfg)

import spikeinterface.preprocessing as spre


def bandpass_to_frac_nyq(rec, fmin: float, frac_nyq: float):
    fs_hz = rec.get_sampling_frequency()
    nyq = fs_hz / 2.0
    fmax = frac_nyq * nyq
    if fmax >= nyq:
        fmax = 0.99 * nyq
    if not (0 < fmin < fmax < nyq):
        raise ValueError(f"Bad bandpass: fmin={fmin}, fmax={fmax}, nyq={nyq}")
    return spre.bandpass_filter(rec, freq_min=fmin, freq_max=fmax)


def slice_seconds(rec, start_s: float, dur_s: float | None):
    if dur_s is None:
        return rec
    fs_hz = rec.get_sampling_frequency()
    start_f = int(round(start_s * fs_hz))
    end_f = int(round((start_s + dur_s) * fs_hz))
    return rec.frame_slice(start_f, end_f)

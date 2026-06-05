from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactConfig:
    enabled: bool = False
    mode: str = "fit"  # fit (default, exp+poly3 subtraction) | blank (lean, dense stim)
    source: str = "auto"  # "auto" (events->detect) | "events" | "detect"
    pre_ms: float = 0.5
    post_ms: float = 3.0
    delta_ms: float = 10.0  # pre-stim baseline window length
    sat_flat_run: int = 5
    sat_dev_sigma: float = 10.0
    sat_max_ms: float = 50.0
    sat_frac_thresh: float = 0.5
    fit_decay_ms: float = 2.0
    fit_M: int = 15
    fit_eps: float = 1.0
    seed: int = 0
    detect_quorum_frac: float = 0.10


# Create frozen dataclass for parameters that affect processing results (for reproducibility (fingers crossed)).
@dataclass(frozen=True)
class PipelineConfig:

    # Time selection
    start_s: float = 0.0
    dur_s: float | None = 30  # None means full recording

    # Preprocessing
    bp_min_hz: float = 300.0
    bp_max_frac_nyq: float = 0.9  # 0.9 * (fs/2)

    # KS4 settings
    ks4_highpass_cutoff_hz: float = 1.0  # 1.0 disables (already bandpassed)
    ks4_batch_size: int = 60000

    # Stim artifact removal
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

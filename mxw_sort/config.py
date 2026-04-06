from dataclasses import dataclass


# Create frozen dataclass for parameters that affect processing results (for reproducibility (fingers crossed)).
@dataclass(frozen=True)
class PipelineConfig:

    # Time selection
    start_s: float = 0.0
    dur_s: float | None = 30   # None means full recording

    # Preprocessing
    bp_min_hz: float = 300.0
    bp_max_frac_nyq: float = 0.9  # 0.9 * (fs/2)

    # KS4 settings
    ks4_highpass_cutoff_hz: float = 1.0  # 1.0 disables (already bandpassed)
    ks4_batch_size: int = 60000

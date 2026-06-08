from pathlib import Path

import typer

from .config import ArtifactConfig, PipelineConfig
from .pipeline import process_directory, process_directory_flat, process_h5

app = typer.Typer(add_completion=False)


# Decodes user-input well selection from CLI. None triggers well auto-detection
def parse_wells(s: str | None) -> tuple[int, ...] | None:  # Only works in Python 3.10+
    if s is None or s.strip().lower() == "auto":
        return None
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


# Decorator here allows function to be called from CLI via Typer ($ mxw-sort run "[PATH TO H5]" --out "[PATH TO OUTPUT FOLDER]")
@app.command()
def run(
    h5: str = typer.Argument(..., help="Path to data.raw.h5"),
    out: Path = typer.Option(..., "--out", help="Output root folder"),
    start_s: float = typer.Option(0.0, "--start-s", help="Start time in seconds"),
    dur_s: float = typer.Option(
        30.0, "--dur-s", help="Seconds to process; set 0 to mean full file"
    ),
    wells: str = typer.Option(
        "auto",
        "--wells",
        help="Wells to process (e.g., '0-5', '0,2,4', or 'auto' to detect)",
    ),
    only_well: int = typer.Option(
        None, "--only-well", help="Run exactly one Maxwell well index (0-5)"
    ),
    bp_min: float = typer.Option(
        300.0, "--bp-min", help="Bandpass filter min frequency (Hz)"
    ),
    bp_max_frac_nyq: float = typer.Option(
        0.9, "--bp-max-frac-nyq", help="Bandpass max as fraction of Nyquist"
    ),
    ks4_hp: float = typer.Option(
        1.0, "--ks4-highpass-cutoff", help="KS4 highpass cutoff (1.0 = disabled)"
    ),
    ks4_batch_size: int = typer.Option(
        60000, "--ks4-batch-size", help="KS4 batch size"
    ),
    remove_stim_artifacts: bool = typer.Option(
        False,
        "--remove-stim-artifacts/--no-remove-stim-artifacts",
        help="Remove stimulation artifacts before bandpass + Kilosort",
    ),
    stim_artifact_mode: str = typer.Option(
        "fit",
        "--stim-artifact-mode",
        help="Artifact correction mode: fit (default) | blank",
    ),
    stim_source: str = typer.Option(
        "auto", "--stim-source", help="Stim-time source: auto | events | npz | detect"
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Flat directory mode: each h5 file gets its own output folder named after the file",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip wells with existing KS4 outputs",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print actions without doing any work"
    ),
):
    # Interpret dur_s=0 as "just run the whole recording"
    dur = None if float(dur_s) == 0 else float(dur_s)

    # Create INFALLABLE config with processing parameters only
    cfg = PipelineConfig(
        start_s=float(start_s),
        dur_s=dur,
        bp_min_hz=float(bp_min),
        bp_max_frac_nyq=float(bp_max_frac_nyq),
        ks4_highpass_cutoff_hz=float(ks4_hp),
        ks4_batch_size=int(ks4_batch_size),
        artifacts=ArtifactConfig(
            enabled=remove_stim_artifacts,
            mode=stim_artifact_mode,
            source=stim_source,
        ),
    )

    out.mkdir(parents=True, exist_ok=True)

    # Process wells (execution parameters are now separated from processing parameters)
    h5_resolved = Path(h5)
    if h5_resolved.is_dir():
        if flat:
            process_directory_flat(
                root_dir=h5_resolved,
                out_root=out,
                cfg=cfg,
                wells=parse_wells(wells),
                skip_existing=skip_existing,
                dry_run=dry_run,
                only_well=only_well,
            )
        else:
            process_directory(
                root_dir=h5_resolved,
                out_root=out,
                cfg=cfg,
                wells=parse_wells(wells),
                skip_existing=skip_existing,
                dry_run=dry_run,
                only_well=only_well,
            )
    else:
        process_h5(
            h5_path=h5,
            out_root=out,
            cfg=cfg,
            wells=parse_wells(wells),
            skip_existing=skip_existing,
            dry_run=dry_run,
            only_well=only_well,
        )


if __name__ == "__main__":
    app()

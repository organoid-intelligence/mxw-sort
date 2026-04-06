import time
from pathlib import Path

import numpy as np
import spikeinterface.preprocessing as spre

from .config import PipelineConfig
from .export import write_binary, write_meta_json, write_probe_json
from .io_maxwell import get_available_wells, get_well_duration_s, read_maxwell
from .ks4 import run_ks4
from .preprocess import bandpass_to_frac_nyq, slice_seconds
from .qc import write_qc


def _ks4_done(ks_dir: Path) -> bool:
    return (ks_dir / "spike_times.npy").exists() and (ks_dir / "spike_clusters.npy").exists()


def process_one_well(
    h5_path: str,
    out_root: Path,
    cfg: PipelineConfig,
    well_idx: int,
    skip_existing: bool = True,
    dry_run: bool = False,
):
    """Single-well pipeline: read -> preprocess -> export binary -> KS4 -> QC."""
    stream = f"well{well_idx:03d}"

    well_dir = out_root / stream
    prep_dir = well_dir / "preprocessed"
    ks_dir = well_dir / "ks4"
    qc_dir = well_dir / "qc"

    bin_path = prep_dir / "traces.bin"
    probe_path = prep_dir / "ks4_probe.json"
    xy_path = prep_dir / "channel_xy.npy"
    meta_path = prep_dir / "meta.json"

    if skip_existing and _ks4_done(ks_dir):
        print(f"[SKIP] {h5_path} {stream} (ks4 outputs exist)")
        return

    print(f"[RUN] {h5_path} {stream} -> {well_dir}")

    if dry_run:
        try:
            dur = get_well_duration_s(h5_path, stream)
            print(f"  duration: {dur:.1f}s")
        except Exception:
            print("  duration: unknown")
        print("  (dry-run) would write:", bin_path)
        print("  (dry-run) would write:", probe_path)
        print("  (dry-run) would run ks4 into:", ks_dir)
        print("  (dry-run) would write qc into:", qc_dir)
        return

    t0 = time.time()

    prep_dir.mkdir(parents=True, exist_ok=True)
    ks_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    rec = read_maxwell(h5_path, stream)
    rec = spre.unsigned_to_signed(rec)
    rec = slice_seconds(rec, cfg.start_s, cfg.dur_s)
    rec = bandpass_to_frac_nyq(rec, cfg.bp_min_hz, cfg.bp_max_frac_nyq)

    fs_hz = rec.get_sampling_frequency()

    write_binary(rec, bin_path)
    xy = rec.get_channel_locations()
    np.save(xy_path, xy)
    write_probe_json(xy, probe_path)

    meta = {
        "h5": h5_path,
        "stream": stream,
        "fs_hz": float(fs_hz),
        "start_s": cfg.start_s,
        "dur_s": cfg.dur_s,
        "bp_min_hz": cfg.bp_min_hz,
        "bp_max_frac_nyq": cfg.bp_max_frac_nyq,
        "n_chan": int(xy.shape[0]),
    }
    write_meta_json(meta, meta_path)

    run_ks4(
        bin_file=bin_path,
        probe_path=probe_path,
        out_dir=ks_dir,
        fs_hz=float(fs_hz),
        n_chan=int(xy.shape[0]),
        batch_size=cfg.ks4_batch_size,
        highpass_cutoff_hz=cfg.ks4_highpass_cutoff_hz,
    )

    write_qc(ks_dir=ks_dir, qc_dir=qc_dir, fs_hz=float(fs_hz), dur_s_processed=cfg.dur_s)

    elapsed = time.time() - t0
    print(f"[DONE] {stream} in {elapsed:.1f}s")


def process_h5(
    h5_path: str,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    if only_well is not None:
        wells = (only_well,)
    elif wells is None:
        wells = get_available_wells(h5_path)
        print(f"Auto-detected {len(wells)} wells: {wells}")

    for w in wells:
        process_one_well(
            h5_path=h5_path,
            out_root=out_root,
            cfg=cfg,
            well_idx=w,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )


def process_directory(
    root_dir: Path,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    h5_files = sorted(root_dir.rglob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found under {root_dir}")
        return

    print(f"Found {len(h5_files)} .h5 file(s) under {root_dir}:")
    for f in h5_files:
        try:
            detected = get_available_wells(str(f))
            stream = f"well{detected[0]:03d}" if detected else "well000"
            dur = get_well_duration_s(str(f), stream)
            print(f"  {f}  ({dur:.1f}s)")
        except Exception:
            print(f"  {f}")
    print()

    for h5_file in h5_files:
        file_out = out_root / h5_file.relative_to(root_dir).parent
        file_out.mkdir(parents=True, exist_ok=True)
        process_h5(
            h5_path=str(h5_file),
            out_root=file_out,
            cfg=cfg,
            wells=wells,
            skip_existing=skip_existing,
            dry_run=dry_run,
            only_well=only_well,
        )


def process_directory_flat(
    root_dir: Path,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    """Non-recursive: all .h5 in one folder, output dirs named by file stem."""
    h5_files = sorted(root_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {root_dir}")
        return

    print(f"[FLAT] Found {len(h5_files)} .h5 file(s) in {root_dir}:")
    for f in h5_files:
        try:
            detected = get_available_wells(str(f))
            stream = f"well{detected[0]:03d}" if detected else "well000"
            dur = get_well_duration_s(str(f), stream)
            print(f"  {f.name}  ({dur:.1f}s)")
        except Exception:
            print(f"  {f.name}")
    print()

    for h5_file in h5_files:
        file_out = out_root / h5_file.stem
        file_out.mkdir(parents=True, exist_ok=True)
        process_h5(
            h5_path=str(h5_file),
            out_root=file_out,
            cfg=cfg,
            wells=wells,
            skip_existing=skip_existing,
            dry_run=dry_run,
            only_well=only_well,
        )

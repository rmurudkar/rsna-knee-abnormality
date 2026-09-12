"""
Phase A, step 1 — build the slice manifest.

Walks the competition DICOM tree and produces one row per slice. No pixels are
read here; this reads DICOM headers only, and everything downstream in the
preprocessing pipeline reads its inputs from the table this produces.

What each group of columns is for:

  identity     study_uid, series_uid, path
               how we group slices into series and series into studies.

  geometry     ipp_*, iop_*, depth, plane
               ipp / iop are the raw DICOM fields. `depth` is how far along the
               stacking direction each slice sits, and is what step 3 sorts by.
               `plane` (sagittal / coronal / axial) is computed from the
               geometry rather than parsed out of the free-text series
               description, which differs between hospitals.

  acquisition  series_description, laterality, series_number, instance_number
               `laterality` drives the left-knee flip in step 6.

  decoding     rescale_slope, rescale_intercept, photometric_interpretation
               needed by step 4. Grabbed now so step 4 never has to reopen
               headers.

  shape        rows, cols, pixel_spacing_*, slice_thickness, spacing_between
               used for sanity checks and for the resize decision in step 7.

Usage
-----
    from build_manifest import build_manifest, summarise_manifest

    df = build_manifest("train", limit=20)     # smoke test on 20 studies
    summarise_manifest(df)

    df = build_manifest("train")               # the real run
    df.to_parquet("/kaggle/working/manifest_train.parquet")
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm.auto import tqdm

COMP_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
OUTPUT_DIR = Path("/kaggle/working")

# Listing the fields explicitly lets pydicom skip the rest of the header. That
# matters when you are opening a few hundred thousand files.
DICOM_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SeriesDescription",
    "Laterality",
    "BodyPartExamined",
    "Modality",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
]

PLANES = ("sagittal", "coronal", "axial")


# --------------------------------------------------------------------------- #
# Geometry
#
# ImageOrientationPatient is six numbers: the first three say which way you
# travel through the patient as you step one pixel RIGHT in the image, the last
# three as you step one pixel DOWN. ImagePositionPatient is where the slice's
# top-left corner sits, in millimetres.
# --------------------------------------------------------------------------- #

def slice_normal(iop):
    """The direction the slices stack along — perpendicular to the image."""
    row_dir = np.asarray(iop[:3], dtype=float)
    col_dir = np.asarray(iop[3:], dtype=float)
    return np.cross(row_dir, col_dir)


def slice_depth(ipp, normal):
    """How far along the stacking direction this slice sits, in millimetres.

    One number per slice. Sorting a series by this puts it in anatomical
    order, which sorting by filename or InstanceNumber does not reliably do.
    """
    return float(np.dot(np.asarray(ipp, dtype=float), normal))


def plane_from_normal(normal):
    """sagittal / coronal / axial, from which patient axis the stack runs along.

    Patient axes are x = toward the left, y = toward the back, z = toward the
    head. Slices stacking along x are side views (sagittal), along y are front
    views (coronal), along z are top-down (axial).
    """
    return PLANES[int(np.argmax(np.abs(normal)))]


# --------------------------------------------------------------------------- #
# Reading one slice's header
# --------------------------------------------------------------------------- #

def _get(ds, name, default=None):
    value = getattr(ds, name, None)
    return default if value is None else value


def _floats(value):
    """DICOM multi-valued numbers come back as a MultiValue, not a list."""
    return None if value is None else [float(v) for v in value]


def read_slice_header(path: Path, study_dir_name: str) -> dict:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=DICOM_TAGS)

    ipp = _floats(_get(ds, "ImagePositionPatient"))
    iop = _floats(_get(ds, "ImageOrientationPatient"))
    spacing = _floats(_get(ds, "PixelSpacing"))

    row = {
        "study_uid": str(_get(ds, "StudyInstanceUID", "")),
        "study_dir": study_dir_name,
        "series_uid": str(_get(ds, "SeriesInstanceUID", "")),
        "path": str(path),

        "series_number": _get(ds, "SeriesNumber"),
        "instance_number": _get(ds, "InstanceNumber"),
        "series_description": str(_get(ds, "SeriesDescription", "")),
        "laterality": str(_get(ds, "Laterality", "")),
        "body_part": str(_get(ds, "BodyPartExamined", "")),
        "modality": str(_get(ds, "Modality", "")),

        "rows": _get(ds, "Rows"),
        "cols": _get(ds, "Columns"),
        "pixel_spacing_row": spacing[0] if spacing else None,
        "pixel_spacing_col": spacing[1] if spacing else None,
        "slice_thickness": _get(ds, "SliceThickness"),
        "spacing_between_slices": _get(ds, "SpacingBetweenSlices"),

        # Everything step 4 needs, so it never has to reopen the header.
        "photometric_interpretation": str(_get(ds, "PhotometricInterpretation", "")),
        "rescale_slope": _get(ds, "RescaleSlope"),
        "rescale_intercept": _get(ds, "RescaleIntercept"),

        "error": None,
    }

    # Raw geometry, kept so we never have to rescan.
    for i, key in enumerate(("ipp_x", "ipp_y", "ipp_z")):
        row[key] = ipp[i] if ipp else None
    for i, key in enumerate(("iop_rx", "iop_ry", "iop_rz", "iop_cx", "iop_cy", "iop_cz")):
        row[key] = iop[i] if iop else None

    # Derived geometry. Both come from the two vectors above, so we compute
    # them here rather than carrying nine columns around and redoing it later.
    if iop is not None:
        normal = slice_normal(iop)
        row["plane"] = plane_from_normal(normal)
        row["depth"] = slice_depth(ipp, normal) if ipp is not None else None
    else:
        row["plane"] = None
        row["depth"] = None

    return row


def scan_study(study_dir: Path) -> list[dict]:
    """Every .dcm under one study folder. Errors are recorded, not raised —
    one unreadable file should not kill a 40-minute scan."""
    rows = []
    for path in sorted(Path(study_dir).rglob("*.dcm")):
        try:
            rows.append(read_slice_header(path, Path(study_dir).name))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "path": str(path),
                "study_dir": Path(study_dir).name,
                "error": repr(exc),
            })
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_manifest(split: str = "train", limit: int | None = None,
                   workers: int | None = None) -> pd.DataFrame:
    """One row per slice for every study in {split}_series/.

    `limit` caps the number of studies — use it to smoke-test on 20 studies
    before committing to the full scan.
    """
    root = COMP_ROOT / f"{split}_series"
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    studies = sorted(p for p in root.iterdir() if p.is_dir())
    if limit is not None:
        studies = studies[:limit]

    workers = workers or os.cpu_count() or 2

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for study_rows in tqdm(pool.map(scan_study, studies, chunksize=4),
                               total=len(studies), desc=f"scanning {split}"):
            rows.extend(study_rows)

    df = pd.DataFrame(rows)
    df["split"] = split
    return df


# --------------------------------------------------------------------------- #
# Summary
#
# This is the point of running the scan first: it answers the questions we
# would otherwise be guessing at for the rest of the pipeline.
# --------------------------------------------------------------------------- #

def summarise_manifest(df: pd.DataFrame) -> None:
    ok = df[df["error"].isna()] if "error" in df else df
    failed = len(df) - len(ok)

    print(f"slices        {len(df):,}   ({failed:,} failed to read)")
    print(f"series        {ok['series_uid'].nunique():,}")
    print(f"studies       {ok['study_uid'].nunique():,}")

    mismatch = (ok["study_uid"] != ok["study_dir"]).sum()
    print(f"folder name disagrees with header StudyInstanceUID: {mismatch:,} slices")

    print("\nplane")
    print(ok["plane"].value_counts(dropna=False).to_string())

    print("\nlaterality")
    print(ok["laterality"].value_counts(dropna=False).to_string())

    per_series = ok.groupby("series_uid").size()
    print("\nslices per series")
    print(per_series.describe().to_string())
    print(f"series with < 8 slices (step 2 drops these): "
          f"{(per_series < 8).sum():,} of {len(per_series):,}")

    # Answers 'do I actually need to apply rescale slope/intercept?'
    print("\n(rescale_slope, rescale_intercept) combinations")
    pairs = ok[["rescale_slope", "rescale_intercept"]].astype(str)
    print(pairs.value_counts(dropna=False).head(10).to_string())

    # Answers 'do I need the MONOCHROME1 inversion?'
    print("\nphotometric interpretation")
    print(ok["photometric_interpretation"].value_counts(dropna=False).to_string())

    print("\nimage size")
    print(ok[["rows", "cols"]].astype(str).value_counts().head(10).to_string())

    print("\nmost common series descriptions")
    print(ok["series_description"].value_counts().head(20).to_string())


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        manifest = build_manifest(split)
        out = OUTPUT_DIR / f"manifest_{split}.parquet"
        manifest.to_parquet(out, index=False)
        print(f"\n=== {split} -> {out} ===")
        summarise_manifest(manifest)

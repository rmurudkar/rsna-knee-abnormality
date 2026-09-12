"""
Phase A, step 2 — filter the manifest down to usable series.

Takes the slice manifest from step 1 and removes what we cannot or should not
train on. Nothing here reads pixels; it is all decisions made from the header
table.

Two kinds of removal happen, and it is worth keeping them straight:

  slice-level   individual slices that are unusable (failed to read, missing
                geometry, duplicate position). If enough of these go, the
                series may then fail a series-level rule.

  series-level  whole series we do not want (too short, localizers, wrong
                modality, inconsistent image size).

Every rejection is recorded with a reason rather than silently dropped, so you
can look at what went and decide whether a rule is too aggressive.

Usage
-----
    from filter_series import filter_manifest, summarise_filter

    kept, report = filter_manifest(manifest)
    summarise_filter(manifest, kept, report)

    # what got thrown away and why
    report[report.reject_reason.notna()].head(50)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_SLICES = 8

# Long distinctive words can be matched anywhere in the description; short ones
# need word boundaries so "loc" does not match some unrelated substring.
LOCALIZER_SUBSTRINGS = ("localizer", "localiser", "scout", "survey", "topogram",
                        "scanogram", "tracker", "smartbrain", "calibration",
                        "3-plane", "3 plane")
LOCALIZER_WORDS = re.compile(r"\b(loc|plan|ref|cal)\b", re.IGNORECASE)

# Slice positions are floats; round before comparing so two slices genuinely at
# the same place compare equal. 0.01 mm is far below any real slice gap.
DEPTH_ROUNDING = 2


def _looks_like_localizer(description: str) -> bool:
    text = (description or "").lower()
    if any(word in text for word in LOCALIZER_SUBSTRINGS):
        return True
    return bool(LOCALIZER_WORDS.search(text))


# --------------------------------------------------------------------------- #
# Slice-level cleaning
# --------------------------------------------------------------------------- #

def _drop_unusable_slices(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove slices we cannot place in space, and duplicate positions.

    Duplicate positions are the interesting case. Some series arrive with two
    slices at the same physical location — multi-echo sequences, or magnitude
    and phase images filed into one folder. Sorting those by depth produces a
    stack that alternates between two different images, which would poison the
    2.5D channel stack. We keep the first by instance number and flag it.
    """
    counts = {"start": len(manifest)}
    m = manifest

    if "error" in m.columns:
        m = m[m["error"].isna()]
    counts["read_errors"] = counts["start"] - len(m)

    before = len(m)
    m = m.dropna(subset=["depth", "plane", "series_uid"])
    counts["missing_geometry"] = before - len(m)

    before = len(m)
    m = m.assign(_depth_key=m["depth"].round(DEPTH_ROUNDING))
    m = (m.sort_values(["series_uid", "_depth_key", "instance_number"])
           .drop_duplicates(subset=["series_uid", "_depth_key"], keep="first")
           .drop(columns="_depth_key"))
    counts["duplicate_positions"] = before - len(m)

    return m, counts


# --------------------------------------------------------------------------- #
# Series-level summary and rules
# --------------------------------------------------------------------------- #

def build_series_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per series, with the aggregates the filter rules need."""
    m = manifest.sort_values(["series_uid", "depth"]).copy()

    # Gap between consecutive slices. In a healthy series these are all about
    # the same — that is the slice spacing. Wildly uneven gaps mean the folder
    # holds more than one acquisition.
    m["_gap"] = m.groupby("series_uid")["depth"].diff()
    gaps = m.groupby("series_uid")["_gap"]
    gap_mean = gaps.mean()
    gap_std = gaps.std()

    table = pd.DataFrame({
        "study_uid": m.groupby("series_uid")["study_uid"].first(),
        "n_slices": m.groupby("series_uid").size(),
        "plane": m.groupby("series_uid")["plane"].agg(lambda s: s.mode().iat[0]),
        "n_planes": m.groupby("series_uid")["plane"].nunique(),
        "series_description": m.groupby("series_uid")["series_description"].first(),
        "laterality": m.groupby("series_uid")["laterality"].first(),
        "modality": m.groupby("series_uid")["modality"].first(),
        "n_sizes": m.groupby("series_uid").apply(
            lambda g: g[["rows", "cols"]].drop_duplicates().shape[0],
            include_groups=False),
        "spacing_median": gaps.median(),
        "spacing_cv": (gap_std / gap_mean.abs().replace(0, np.nan)).abs(),
    })
    return table.reset_index()


def _reject_reason(row, min_slices: int) -> str | None:
    """First rule that fires wins, so the report reads as one reason per series.

    Ordered cheapest-and-most-certain first.
    """
    if str(row["modality"]).upper() not in ("MR", ""):
        return f"modality={row['modality']}"

    if _looks_like_localizer(row["series_description"]):
        return "localizer/scout"

    if row["n_slices"] < min_slices:
        return f"too few slices ({int(row['n_slices'])} < {min_slices})"

    # Different image dimensions inside one series cannot be stacked into a
    # single volume in step 7.
    if row["n_sizes"] > 1:
        return "inconsistent image size"

    # Slices facing different directions in one series means two acquisitions
    # were filed together.
    if row["n_planes"] > 1:
        return "mixed planes in one series"

    return None


def filter_manifest(manifest: pd.DataFrame, min_slices: int = MIN_SLICES,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (kept slice manifest, per-series report).

    The report has a row for every series seen, with `reject_reason` set on the
    ones that were dropped and null on the ones that survived. Keeping the
    rejected rows is deliberate — a filter you cannot inspect is a filter you
    cannot trust.
    """
    cleaned, slice_counts = _drop_unusable_slices(manifest)

    report = build_series_table(cleaned)
    report["reject_reason"] = report.apply(_reject_reason, axis=1,
                                           min_slices=min_slices)

    keep_uids = set(report.loc[report["reject_reason"].isna(), "series_uid"])
    kept = cleaned[cleaned["series_uid"].isin(keep_uids)].copy()

    kept.attrs["slice_counts"] = slice_counts
    return kept, report


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def summarise_filter(original: pd.DataFrame, kept: pd.DataFrame,
                     report: pd.DataFrame) -> None:
    counts = kept.attrs.get("slice_counts", {})

    print("slice-level removals")
    for name in ("read_errors", "missing_geometry", "duplicate_positions"):
        print(f"  {name:22} {counts.get(name, 0):,}")

    print("\nseries-level removals")
    reasons = report["reject_reason"].fillna("KEPT")
    print(reasons.value_counts().to_string())

    print(f"\nslices   {len(original):,} -> {len(kept):,}")
    print(f"series   {report['series_uid'].nunique():,} -> "
          f"{kept['series_uid'].nunique():,}")
    print(f"studies  {original['study_uid'].nunique():,} -> "
          f"{kept['study_uid'].nunique():,}")

    # A study with nothing left cannot be predicted on. For train that is a
    # dropped row; for test it means you still owe a submission for it.
    lost = set(original["study_uid"].dropna()) - set(kept["study_uid"])
    print(f"\nstudies with no surviving series: {len(lost):,}")
    if lost:
        print("  " + ", ".join(sorted(lost)[:5]) + (" ..." if len(lost) > 5 else ""))

    print("\ndescriptions dropped as localizer/scout (check these are junk)")
    dropped = report[report["reject_reason"] == "localizer/scout"]
    print(dropped["series_description"].value_counts().head(15).to_string())

    kept_series = report[report["reject_reason"].isna()]
    print("\nsurviving series per study")
    print(kept_series.groupby("study_uid").size().describe().to_string())

    print("\nsurviving series by plane")
    print(kept_series["plane"].value_counts().to_string())

    # Not a rejection rule — uneven spacing is suspicious but not always fatal,
    # so it is reported for you to look at rather than acted on automatically.
    uneven = kept_series[kept_series["spacing_cv"] > 0.1]
    print(f"\nkept series with uneven slice spacing (cv > 0.1): {len(uneven):,}")

    missing_lat = kept_series["laterality"].isin(["", "nan", "None"]).sum()
    print(f"kept series with no Laterality field: {missing_lat:,} "
          f"(these need the geometry fallback in step 6)")

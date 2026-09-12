"""
Phase A, step 2 — filter and select the series we will actually cache.

Input is the slice manifest from step 1. Output is the exact list of slices the
cache builder should process. Nothing here reads pixels.

Four things happen, in order:

  laterality     normalise L/LEFT/R/RIGHT to one letter, then fill blanks from
                 other series in the same study. 44% of slices arrive with no
                 Laterality field, and step 6's left-knee flip depends on it.

  step 2   drop  slices we cannot use (unreadable, no geometry, duplicate
                 position) and series we do not want (localizers, non-MR,
                 too short, inconsistent size, mixed planes, bilateral).

  step 2b  select at most N series per plane per study. The dataset has ~5.5
                 series per study, which is more than the model needs and more
                 than fits on disk. We keep the most informative one per plane.

  step 6b  thin  each kept series to roughly one slice every few millimetres,
                 with a hard cap. 3D sequences arrive with 320 sub-millimetre
                 slices; thinning them to the same through-plane spacing as the
                 2D sequences makes every series comparable and makes the cache
                 fit.

Usage
-----
    from filter_series import filter_manifest, summarise_filter

    kept, report = filter_manifest(manifest)
    summarise_filter(manifest, kept, report)

    report[report.reject_reason.notna()].head(50)      # what was dropped
    report[~report.selected & report.reject_reason.isna()]  # kept but not chosen
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_SLICES = 8
MAX_SERIES_PER_PLANE = 1        # step 2b: how many series to keep per plane
TARGET_SPACING_MM = 3.5         # step 6b: desired gap between kept slices
MAX_SLICES_PER_SERIES = 24      # step 6b: hard cap, drives the cache size

BYTES_PER_SLICE = 224 * 224     # uint8 at the cache resolution

# Long distinctive words can be matched anywhere in the description; short ones
# are matched as whole tokens.
LOCALIZER_SUBSTRINGS = ("localizer", "localiser", "scout", "survey", "topogram",
                        "scanogram", "tracker", "smartbrain", "calibration",
                        "3-plane", "3 plane")
LOCALIZER_TOKENS = {"loc", "plan", "ref", "cal"}

# Slice positions are floats; round before comparing so two slices genuinely at
# the same place compare equal. 0.01 mm is far below any real slice gap.
DEPTH_ROUNDING = 2


# --------------------------------------------------------------------------- #
# Reading the free-text series description
#
# Descriptions look like "pd_tse_fs_sag_320" or "SAG 3D_VIEW_PD_SPAIR_HR L".
# Splitting on every non-alphanumeric character turns those into clean tokens,
# which is safer than regex word boundaries — in regex an underscore counts as
# a word character, so \bfs\b does NOT match inside "pd_tse_fs_sag".
# --------------------------------------------------------------------------- #

def _tokens(description: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (description or "").lower()) if t}


FAT_SAT_TOKENS = {"fs", "fatsat", "spair", "spir", "stir", "fsat", "spectral"}
PD_TOKENS = {"pd", "pdw", "proton"}
T2_TOKENS = {"t2", "t2w"}
T1_TOKENS = {"t1", "t1w"}
THREE_D_TOKENS = {"3d", "de3d", "dess", "space", "vibe", "wats", "fiesta", "medic"}


def classify_sequence(description: str) -> str:
    """Coarse sequence type, used to decide which series is worth keeping.

    Fat suppression is treated as a modifier rather than a type, so a 3D PD
    SPAIR still classifies as pd_fs — it is an excellent meniscus sequence and
    should not be downranked for being 3D.
    """
    t = _tokens(description)
    if not t:
        return "unknown"

    fat = bool(t & FAT_SAT_TOKENS)

    if "stir" in t:
        return "stir"
    if t & PD_TOKENS:
        return "pd_fs" if fat else "pd"
    if t & T2_TOKENS:
        return "t2_fs" if fat else "t2"
    if t & T1_TOKENS:
        return "t1"
    if t & THREE_D_TOKENS:
        return "3d"
    return "unknown"


# Higher is better. Fluid-sensitive fat-suppressed sequences show meniscal
# tears, effusion, synovitis, bone oedema and contusion — most of our labels.
# T1 is mainly for anatomy and marrow, so it ranks last. "unknown" sits mid
# table on purpose: 19% of this dataset has its description stripped, and we
# must not systematically exclude all of it.
SEQUENCE_SCORE = {
    "stir": 100,
    "pd_fs": 100,
    "t2_fs": 95,
    "pd": 70,
    "t2": 65,
    "unknown": 55,
    "3d": 50,
    "t1": 40,
}


def _looks_like_localizer(description: str) -> bool:
    text = (description or "").lower()
    if any(word in text for word in LOCALIZER_SUBSTRINGS):
        return True
    return bool(_tokens(description) & LOCALIZER_TOKENS)


# --------------------------------------------------------------------------- #
# Laterality
# --------------------------------------------------------------------------- #

def normalise_laterality(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One letter per slice, filled in from the rest of the study where blank.

    Every series in a study images the same knee, so a study where any series
    recorded laterality can fill in all the others. Whatever is still blank
    after that needs the geometry fallback in step 6.
    """
    m = manifest.copy()
    raw = m["laterality"].fillna("").astype(str).str.strip().str.upper()

    # "LEFT" -> "L", "RIGHT" -> "R", "B" (bilateral) kept so step 2 can drop it.
    letter = raw.str[:1]
    m["laterality"] = letter.where(letter.isin(["L", "R", "B"]), "")

    before_blank = int((m["laterality"] == "").sum())

    # Propagate the study's known value into its blank slices.
    known = m.loc[m["laterality"].isin(["L", "R"])]
    by_study = known.groupby("study_uid")["laterality"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")
    filled = m["study_uid"].map(by_study).fillna("")
    m["laterality"] = m["laterality"].where(m["laterality"] != "", filled)

    stats = {
        "blank_before": before_blank,
        "blank_after": int((m["laterality"] == "").sum()),
        "studies_with_none": int(
            m.groupby("study_uid")["laterality"]
             .agg(lambda s: not s.isin(["L", "R"]).any()).sum()),
    }
    return m, stats


# --------------------------------------------------------------------------- #
# Step 2 — slice-level cleaning
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
# Step 2 — series-level summary and rules
# --------------------------------------------------------------------------- #

def build_series_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per series, with the aggregates the rules and ranking need."""
    m = manifest.sort_values(["series_uid", "depth"]).copy()

    # A plain string is a version-proof way to count distinct image sizes.
    m["_size"] = m["rows"].astype(str) + "x" + m["cols"].astype(str)

    # Gap between consecutive slices. In a healthy series these are all about
    # the same — that is the slice spacing. Wildly uneven gaps mean the folder
    # holds more than one acquisition.
    m["_gap"] = m.groupby("series_uid")["depth"].diff()
    by_series = m.groupby("series_uid")
    gaps = by_series["_gap"]
    gap_mean, gap_std = gaps.mean(), gaps.std()

    table = pd.DataFrame({
        "study_uid": by_series["study_uid"].first(),
        "n_slices": by_series.size(),
        "plane": by_series["plane"].agg(lambda s: s.mode().iat[0]),
        "n_planes": by_series["plane"].nunique(),
        "series_description": by_series["series_description"].first(),
        "laterality": by_series["laterality"].first(),
        "modality": by_series["modality"].first(),
        "n_sizes": by_series["_size"].nunique(),
        "rows": by_series["rows"].first(),
        "cols": by_series["cols"].first(),
        "spacing_median": gaps.median().abs(),
        "spacing_cv": (gap_std / gap_mean.abs().replace(0, np.nan)).abs(),
    }).reset_index()

    table["sequence"] = table["series_description"].map(classify_sequence)
    table["sequence_score"] = table["sequence"].map(SEQUENCE_SCORE).fillna(50)
    return table


def _reject_reason(row, min_slices: int) -> str | None:
    """First rule that fires wins, so the report reads as one reason per series."""
    if str(row["modality"]).upper() not in ("MR", ""):
        return f"modality={row['modality']}"

    if _looks_like_localizer(row["series_description"]):
        return "localizer/scout"

    if row["n_slices"] < min_slices:
        return f"too few slices ({int(row['n_slices'])} < {min_slices})"

    # Both knees in frame — medial vs lateral is undefined, so the step 6 flip
    # cannot be applied and six of the twelve labels lose their meaning.
    if row["laterality"] == "B":
        return "bilateral"

    # Different image dimensions inside one series cannot be stacked into a
    # single volume in step 7.
    if row["n_sizes"] > 1:
        return "inconsistent image size"

    # Slices facing different directions in one series means two acquisitions
    # were filed together.
    if row["n_planes"] > 1:
        return "mixed planes in one series"

    return None


# --------------------------------------------------------------------------- #
# Step 2b — series selection
# --------------------------------------------------------------------------- #

def select_series(report: pd.DataFrame,
                  max_per_plane: int = MAX_SERIES_PER_PLANE) -> pd.DataFrame:
    """Mark the best `max_per_plane` surviving series in each study and plane.

    Ranking, in order:

      1. sequence score — fluid-sensitive fat-suppressed sequences first,
         because they carry most of our twelve findings.
      2. slice count closest to that plane's median. This is the tiebreak that
         does the work when the description is stripped, and it also pushes
         away both the unusually short series and the 320-slice 3D ones.
    """
    report = report.copy()
    report["selected"] = False

    alive = report["reject_reason"].isna()
    if not alive.any():
        return report

    plane_median = report.loc[alive].groupby("plane")["n_slices"].median()
    report["_slice_dist"] = (
        report["n_slices"] - report["plane"].map(plane_median)).abs()

    ranked = (report[alive]
              .sort_values(["study_uid", "plane", "sequence_score", "_slice_dist"],
                           ascending=[True, True, False, True]))
    ranked["_rank"] = ranked.groupby(["study_uid", "plane"]).cumcount()

    chosen = ranked.loc[ranked["_rank"] < max_per_plane, "series_uid"]
    report.loc[report["series_uid"].isin(set(chosen)), "selected"] = True
    return report.drop(columns="_slice_dist")


# --------------------------------------------------------------------------- #
# Step 6b — thin each series along the stack
# --------------------------------------------------------------------------- #

def _keep_positions(n: int, spacing_mm: float, target_spacing_mm: float,
                    max_slices: int) -> np.ndarray:
    """Which positions within a sorted series to keep.

    A 2D series already at ~3.5 mm spacing is untouched. A 3D series at 0.6 mm
    keeps every sixth slice, landing at comparable through-plane spacing — so
    the 2.5D neighbour stack spans a similar amount of anatomy on both.
    """
    if n <= 0:
        return np.array([], dtype=int)

    stride = 1
    if spacing_mm and spacing_mm > 0:
        stride = max(1, int(round(target_spacing_mm / spacing_mm)))

    idx = np.arange(0, n, stride)
    if len(idx) > max_slices:
        # Still too many (a long series covering a lot of leg): spread the cap
        # evenly across the whole stack so coverage is preserved.
        idx = np.unique(np.linspace(0, n - 1, max_slices).round().astype(int))
    return idx


def thin_series(manifest: pd.DataFrame, report: pd.DataFrame,
                target_spacing_mm: float = TARGET_SPACING_MM,
                max_slices: int = MAX_SLICES_PER_SERIES) -> pd.DataFrame:
    """Keep an evenly spaced subset of each series' slices."""
    spacing = report.set_index("series_uid")["spacing_median"].to_dict()

    m = manifest.sort_values(["series_uid", "depth"]).copy()
    m["_pos"] = m.groupby("series_uid").cumcount()

    keep = []
    for uid, group in m.groupby("series_uid", sort=False):
        positions = _keep_positions(len(group), spacing.get(uid, 0.0),
                                    target_spacing_mm, max_slices)
        keep.append(group[group["_pos"].isin(set(positions.tolist()))])

    return pd.concat(keep).drop(columns="_pos") if keep else m.drop(columns="_pos")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def filter_manifest(manifest: pd.DataFrame,
                    min_slices: int = MIN_SLICES,
                    max_per_plane: int = MAX_SERIES_PER_PLANE,
                    target_spacing_mm: float = TARGET_SPACING_MM,
                    max_slices: int = MAX_SLICES_PER_SERIES,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (slices to cache, per-series report).

    The report has a row for every series seen. `reject_reason` says why a
    series was dropped by step 2; `selected` says whether step 2b chose it. A
    series can be perfectly usable and still not selected — that is the normal
    case, since most studies have more series than we need.
    """
    m, laterality_stats = normalise_laterality(manifest)
    m, slice_counts = _drop_unusable_slices(m)

    report = build_series_table(m)
    report["reject_reason"] = report.apply(_reject_reason, axis=1,
                                           min_slices=min_slices)
    report = select_series(report, max_per_plane=max_per_plane)

    chosen = set(report.loc[report["selected"], "series_uid"])
    kept = m[m["series_uid"].isin(chosen)].copy()
    slice_counts["after_selection"] = len(kept)

    kept = thin_series(kept, report, target_spacing_mm, max_slices)
    slice_counts["after_thinning"] = len(kept)

    kept.attrs["slice_counts"] = slice_counts
    kept.attrs["laterality_stats"] = laterality_stats
    return kept, report


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def summarise_filter(original: pd.DataFrame, kept: pd.DataFrame,
                     report: pd.DataFrame) -> None:
    counts = kept.attrs.get("slice_counts", {})
    lat = kept.attrs.get("laterality_stats", {})

    print("laterality")
    print(f"  blank before study-level fill  {lat.get('blank_before', 0):,}")
    print(f"  blank after                    {lat.get('blank_after', 0):,}")
    print(f"  studies with no laterality at all  {lat.get('studies_with_none', 0):,}"
          "   <- these need the geometry fallback in step 6")

    print("\nslice-level removals")
    for name in ("read_errors", "missing_geometry", "duplicate_positions"):
        print(f"  {name:22} {counts.get(name, 0):,}")

    print("\nseries-level removals (step 2)")
    print(report["reject_reason"].fillna("KEPT").value_counts().to_string())

    alive = report[report["reject_reason"].isna()]
    print(f"\nseries selection (step 2b)")
    print(f"  usable series      {len(alive):,}")
    print(f"  selected           {int(report['selected'].sum()):,}")
    print("  selected by sequence type:")
    print(report[report["selected"]]["sequence"].value_counts()
          .to_string().replace("\n", "\n    "))

    print("\nslices")
    print(f"  manifest                {len(original):,}")
    print(f"  after step 2 + 2b       {counts.get('after_selection', 0):,}")
    print(f"  after thinning (6b)     {counts.get('after_thinning', 0):,}")

    gb = len(kept) * BYTES_PER_SLICE / 1e9
    print(f"\nprojected cache at 224x224 uint8: {gb:.1f} GB"
          f"   {'OK' if gb < 19 else 'TOO BIG -- lower MAX_SLICES_PER_SERIES'}")

    print(f"\nstudies   {original['study_uid'].nunique():,} -> "
          f"{kept['study_uid'].nunique():,}")
    lost = set(original["study_uid"].dropna()) - set(kept["study_uid"])
    print(f"studies with nothing left: {len(lost):,}")
    if lost:
        print("  " + ", ".join(sorted(lost)[:5]) + (" ..." if len(lost) > 5 else ""))

    print("\nselected series per study")
    per_study = report[report["selected"]].groupby("study_uid").size()
    print(per_study.value_counts().sort_index().to_string())

    print("\nplanes present among selected series")
    print(report[report["selected"]]["plane"].value_counts().to_string())

    print("\ndescriptions dropped as localizer/scout (check these are junk)")
    dropped = report[report["reject_reason"] == "localizer/scout"]
    print(dropped["series_description"].value_counts().head(15).to_string()
          if len(dropped) else "  none")

    uneven = alive[alive["spacing_cv"] > 0.1]
    print(f"\nusable series with uneven slice spacing (cv > 0.1): {len(uneven):,}")

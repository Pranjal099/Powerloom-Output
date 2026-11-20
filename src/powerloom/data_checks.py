"""
src/powerloom/data_checks.py
Seven data quality checks for the Power Loom Output dataset (G018 MP243).

Run:
    python -m src.powerloom.data_checks

Each check prints its finding and writes the final results dict (which you can
paste into reports/findings.yaml's data_issues section).

The seven kinds checked (per STUDENT_HANDBOOK.md §7):
  1. leaking_column     – column known only after the event
  2. unit_mixup         – one shed recorded a column in a different unit
  3. sentinel_value     – impossible placeholder number typed instead of blank
  4. duplicate_rows     – same record appears more than once
  5. stuck_sensor       – device sent the same value for many consecutive days
  6. target_typos       – targets typed with an extra zero (10× too big)
  7. spelling_variants  – same category written in multiple ways

Author: Divyanshi Maurya (Data Lead) — S24CSEU0803
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with open(ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)


PARAMS = _load_params()

TRAIN_PATH = ROOT / PARAMS["paths"]["train_raw"]
LATER_PATH = ROOT / PARAMS["paths"]["later_raw"]
HOLDOUT_PATH = ROOT / PARAMS["paths"]["holdout"]
TARGET_COL = PARAMS["data"]["target_col"]
GROUP_COL = PARAMS["data"]["group_col"]
LEAKING_COL = PARAMS["data"]["leaking_col"]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  CHECK: {title}")
    print(f"{'─' * 60}")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, later, and holdout CSVs."""
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    return train, later, holdout


# ─── 1. Leaking column ───────────────────────────────────────────────────────


def check_leaking_column(
    train: pd.DataFrame, holdout: pd.DataFrame
) -> dict:
    """
    A leaking column is filled AFTER the event (shift ends) and therefore
    unavailable at prediction time.  It will be missing from holdout_features.csv.

    'metres_checked_at_folding' is measured at the folding table AFTER the
    shift completes, so it cannot be known when the prediction is requested.
    """
    _section("1. Leaking Column")

    holdout_cols = set(holdout.columns)
    candidates = [c for c in train.columns if c not in holdout_cols and c != TARGET_COL]

    result = {"present": False, "column": None}

    if candidates:
        # Confirm by checking correlation with target
        correlations = {
            c: abs(train[c].corr(train[TARGET_COL]))
            for c in candidates
            if pd.api.types.is_numeric_dtype(train[c])
        }
        leaking = max(correlations, key=correlations.get) if correlations else candidates[0]
        corr_val = correlations.get(leaking, None)
        result = {"present": True, "column": leaking}
        print(f"  PRESENT  — column: '{leaking}'")
        print(f"  Correlation with target: {corr_val:.4f}")
        print(f"  Missing from holdout: Yes")
        print(f"  Reason: measured at folding table AFTER shift ends")
    else:
        print("  NOT PRESENT — all train columns appear in holdout (or are the target).")

    return result


# ─── 2. Unit mixup ───────────────────────────────────────────────────────────


def check_unit_mixup(train: pd.DataFrame) -> dict:
    """
    One office (one value of shed) may have recorded a numeric column in a
    different unit.  Spot by comparing min/mean/max per shed group.
    A genuine mixup shows one shed's values ~10×, 12×, 60×, or 1 000× off.
    """
    _section("2. Unit Mixup")

    numeric_cols = [
        c for c in train.select_dtypes(include=np.number).columns
        if c not in (TARGET_COL, "metres_checked_at_folding")
    ]

    result = {"present": False, "column": None, "group_value": None, "factor": None}
    found = False

    for col in numeric_cols:
        medians = train.groupby(GROUP_COL)[col].median()
        for s1 in medians.index:
            for s2 in medians.index:
                if s1 >= s2:
                    continue
                ratio = medians[s1] / medians[s2] if medians[s2] != 0 else float("inf")
                if ratio < 1:
                    ratio = 1.0 / ratio
                    outlier_shed = s1
                else:
                    outlier_shed = s2
                if ratio >= 8:  # threshold: 8× or more suggests unit mixup
                    result = {
                        "present": True,
                        "column": col,
                        "group_value": outlier_shed,
                        "factor": round(ratio),
                    }
                    print(
                        f"  PRESENT  — column '{col}', shed '{outlier_shed}' "
                        f"has ~{round(ratio)}× different values"
                    )
                    found = True
                    break
            if found:
                break
        if found:
            break

    if not found:
        print("  NOT PRESENT — no column shows a ≥8× ratio between any two sheds.")
        print("  Per-shed medians for reference:")
        print(train.groupby(GROUP_COL)[numeric_cols].median().round(2).to_string())

    return result


# ─── 3. Sentinel value ───────────────────────────────────────────────────────


def check_sentinel_value(train: pd.DataFrame) -> dict:
    """
    Sentinel values are impossible placeholder numbers (-1, -999, 999, 9999)
    typed instead of leaving a field blank.  value_counts() reveals them as
    an impossible value repeated more times than expected by chance.
    """
    _section("3. Sentinel Value")

    # Common sentinels to check
    CANDIDATES = [-9999, -999, -99, -1, 99, 999, 9999]
    MIN_COUNT = 3  # must appear at least this many times to be suspicious

    numeric_cols = [
        c for c in train.select_dtypes(include=np.number).columns
        if c not in (TARGET_COL,)
    ]

    result = {"present": False, "column": None, "value": None, "count": None}

    hits = []
    for col in numeric_cols:
        col_min = train[col].min()
        col_max = train[col].max()
        for val in CANDIDATES:
            cnt = (train[col] == val).sum()
            if cnt >= MIN_COUNT:
                # Only flag if the value is at or outside the realistic range
                # (e.g. -1 when min should be 0, or 999 when max should be 120)
                is_impossible = (val < 0 and col_min >= 0) or (val > col_max * 1.5)
                hits.append(
                    {
                        "col": col,
                        "val": val,
                        "cnt": cnt,
                        "impossible": is_impossible,
                    }
                )

    # Filter to impossible sentinels first; fall back to any if none found
    real_hits = [h for h in hits if h["impossible"]]
    if not real_hits:
        real_hits = hits  # accept in-range sentinels if count is high

    if real_hits:
        # Pick the one with the highest count
        best = max(real_hits, key=lambda x: x["cnt"])
        result = {
            "present": True,
            "column": best["col"],
            "value": best["val"],
            "count": int(best["cnt"]),
        }
        print(
            f"  PRESENT  — column '{best['col']}', value={best['val']}, "
            f"count={best['cnt']}"
        )
    else:
        print("  NOT PRESENT — no impossible placeholder value found in numeric columns.")

    return result


# ─── 4. Duplicate rows ───────────────────────────────────────────────────────


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    """
    Exact duplicate rows where every column value matches.
    df.duplicated().sum() gives the count of EXTRA copies (rows to delete).
    """
    _section("4. Duplicate Rows")

    extra_copies = int(train.duplicated().sum())
    present = extra_copies > 0

    result = {"present": present, "count": extra_copies if present else None}

    if present:
        print(f"  PRESENT  — {extra_copies} extra duplicate rows found.")
        print("  Sample duplicate record_ids:")
        dup_ids = train[train.duplicated(keep="first")]["record_id"].head(5).tolist()
        print(f"  {dup_ids}")
    else:
        print("  NOT PRESENT — no duplicate rows.")

    return result


# ─── 5. Stuck sensor ─────────────────────────────────────────────────────────


def check_stuck_sensor(train: pd.DataFrame) -> dict:
    """
    A stuck sensor repeats the SAME reading for many consecutive days.
    Sort by date, then look for the longest run of identical values per column.
    """
    _section("5. Stuck Sensor")

    MIN_RUN = 7  # days considered 'stuck'

    df = train.sort_values("date").reset_index(drop=True)
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in (TARGET_COL, "metres_checked_at_folding")
    ]

    result = {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }

    best: dict = {}

    for col in numeric_cols:
        cur_val = None
        cur_start = None
        cur_len = 0

        for idx, row in df.iterrows():
            val = row[col]
            date = row["date"]

            if val == cur_val:
                cur_len += 1
                cur_end = date
            else:
                if cur_len >= MIN_RUN:
                    if not best or cur_len > best.get("length", 0):
                        best = {
                            "col": col,
                            "val": cur_val,
                            "start": cur_start,
                            "end": cur_end,
                            "length": cur_len,
                        }
                cur_val = val
                cur_start = date
                cur_end = date
                cur_len = 1

        # Check final run
        if cur_len >= MIN_RUN:
            if not best or cur_len > best.get("length", 0):
                best = {
                    "col": col,
                    "val": cur_val,
                    "start": cur_start,
                    "end": df.iloc[-1]["date"],
                    "length": cur_len,
                }

    if best:
        result = {
            "present": True,
            "column": best["col"],
            "start_date": best["start"],
            "end_date": best["end"],
        }
        print(
            f"  PRESENT  — column '{best['col']}' stuck at {best['val']} "
            f"for {best['length']} consecutive rows"
        )
        print(f"  Start: {best['start']}   End: {best['end']}")
    else:
        print(f"  NOT PRESENT — no column stuck for ≥{MIN_RUN} consecutive rows.")

    return result


# ─── 6. Target typos ─────────────────────────────────────────────────────────


def check_target_typos(train: pd.DataFrame) -> dict:
    """
    A few target values may have been typed with an extra zero (10× too big).
    They appear as a small cluster far above all normal values with a gap below.
    """
    _section("6. Target Typos")

    tgt = train[TARGET_COL]
    p95 = tgt.quantile(0.95)

    # Rows sitting >5× above p95 with a clear gap are typos
    MULTIPLIER = 5
    typo_mask = tgt > (p95 * MULTIPLIER)
    typo_count = int(typo_mask.sum())
    present = typo_count > 0

    result = {"present": present, "count": typo_count if present else None}

    if present:
        print(f"  PRESENT  — {typo_count} rows with {TARGET_COL} > {MULTIPLIER}×p95={p95:.1f}")
        print("  Typo rows:")
        print(
            train[typo_mask][["record_id", "date", TARGET_COL]]
            .sort_values(TARGET_COL)
            .to_string(index=False)
        )
        print(f"\n  Normal p95:         {p95:.1f} m")
        print(f"  Typo values range:  {tgt[typo_mask].min():.1f} – {tgt[typo_mask].max():.1f} m")
    else:
        print("  NOT PRESENT — no obvious 10× target typos found.")

    return result


# ─── 7. Spelling variants ─────────────────────────────────────────────────────


def check_spelling_variants(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """
    The same category may be written in different ways in train.csv
    (e.g. 'Silk', 'silk ', 'SILK').  later.csv has the canonical spellings.
    """
    _section("7. Spelling Variants")

    text_cols = [
        c for c in train.select_dtypes(include="object").columns
        if c not in ("record_id", "date")
    ]

    result = {
        "present": False,
        "column": None,
        "true_levels": None,
    }

    for col in text_cols:
        if col not in later.columns:
            continue

        # Normalise: strip whitespace, lower-case
        train_norm = train[col].str.strip().str.lower()
        later_norm = later[col].str.strip().str.lower()

        train_levels = set(train_norm.unique())
        later_levels = set(later_norm.unique())
        raw_train_levels = set(train[col].unique())

        # Variants exist if train has MORE raw levels than later (which is clean)
        if len(raw_train_levels) > len(later_levels):
            result = {
                "present": True,
                "column": col,
                "true_levels": len(later_levels),
            }
            print(f"  PRESENT  — column '{col}'")
            print(f"  Raw train levels ({len(raw_train_levels)}): {sorted(raw_train_levels)}")
            print(f"  Clean later levels ({len(later_levels)}): {sorted(later_levels)}")
            break
        else:
            print(
                f"  {col}: {len(raw_train_levels)} raw levels in train, "
                f"{len(later_levels)} in later — OK"
            )

    if not result["present"]:
        print("  NOT PRESENT — no spelling variants found in text columns.")

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def run_all_checks() -> dict:
    """Run all 7 checks and return a data_issues dict."""
    if not TRAIN_PATH.exists():
        print(
            f"[data_checks] train.csv not found at {TRAIN_PATH}.\n"
            "Run `dvc pull` to fetch data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    train, later, holdout = _load()

    print(f"\nRunning all 7 data checks on {TRAIN_PATH}")
    print(f"train.csv: {len(train):,} rows × {len(train.columns)} columns")

    issues = {
        "leaking_column":    check_leaking_column(train, holdout),
        "unit_mixup":        check_unit_mixup(train),
        "sentinel_value":    check_sentinel_value(train),
        "duplicate_rows":    check_duplicate_rows(train),
        "stuck_sensor":      check_stuck_sensor(train),
        "target_typos":      check_target_typos(train),
        "spelling_variants": check_spelling_variants(train, later),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, res in issues.items():
        status = "✓ PRESENT" if res["present"] else "✗ absent"
        print(f"  {name:<22} {status}")

    return issues


if __name__ == "__main__":
    run_all_checks()

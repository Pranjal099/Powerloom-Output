"""Unit tests for data cleaning functions (MP243).

Authored by: Pranjal (Service Lead) & Divyanshi Maurya (Data Lead)
Verifies:
  - Deduplication drops exact duplicate records.
  - Leaking column metres_checked_at_folding is removed.
  - Target typos (> 200m) are divided by 10.
  - Stuck sensor readings are identified and replaced.
  - Category normalization trims and lowercases text.
No raw CSV dependencies (pure synthetic fixtures).
"""

import pandas as pd
import pytest

from powerloom.data import (
    remove_duplicates,
    remove_leaking_column,
    correct_target_typos,
    correct_stuck_sensor,
    normalize_categories,
)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Create in-memory synthetic DataFrame mimicking raw train.csv."""
    return pd.DataFrame(
        {
            "record_id": ["R001", "R002", "R003", "R004", "R004"],
            "date": ["2026-01-10", "2026-02-15", "2026-03-20", "2026-04-05", "2026-04-05"],
            "loom_age_yr": [5.0, 10.0, 15.0, 8.0, 8.0],
            "thread_count": [120, 130, 147, 125, 125],
            "power_cuts_min": [30, 45, 60, 15, 15],
            "weaver_exp_yr": [10.0, 15.0, 20.0, 5.0, 5.0],
            "humidity_pct": [65.0, 60.0, 55.0, 70.0, 70.0],
            "shift_hours": [8.0, 9.0, 10.0, 8.0, 8.0],
            "fabric": ["cotton", "silk", "polyester", "cotton", "cotton"],
            "shed": ["Shed 1", "Shed 2", "Shed 3", "Shed 1", "Shed 1"],
            "metres_checked_at_folding": [42.0, 39.5, 51.0, 45.0, 45.0],
            "metres_per_shift": [42.5, 40.0, 510.0, 44.0, 44.0],  # R003 has typo: 510.0
        }
    )


def test_remove_duplicates(synthetic_raw_df):
    """Test that duplicate rows are completely removed."""
    df_dedup = remove_duplicates(synthetic_raw_df)
    assert len(df_dedup) == 4
    assert df_dedup["record_id"].tolist() == ["R001", "R002", "R003", "R004"]


def test_remove_leaking_column(synthetic_raw_df):
    """Test that metres_checked_at_folding is dropped."""
    df_noleak = remove_leaking_column(synthetic_raw_df, "metres_checked_at_folding")
    assert "metres_checked_at_folding" not in df_noleak.columns


def test_correct_target_typos(synthetic_raw_df):
    """Test that target typos (> 200m) are divided by 10."""
    df_fixed, typo_count = correct_target_typos(synthetic_raw_df, "metres_per_shift", 200.0)
    assert typo_count == 1
    # R003 had 510.0, should now be 51.0
    r3_target = df_fixed.loc[df_fixed["record_id"] == "R003", "metres_per_shift"].iloc[0]
    assert r3_target == pytest.approx(51.0)
    assert (df_fixed["metres_per_shift"] <= 150.0).all()


def test_correct_stuck_sensor(synthetic_raw_df):
    """Test that stuck sensor readings (147 during stuck window) are detected."""
    df_fixed, stuck_count = correct_stuck_sensor(
        synthetic_raw_df, "thread_count", 147, "2026-03-14", "2026-03-28"
    )
    assert stuck_count == 1
    # Value in R003 should be set to NA for downstream imputation
    assert pd.isna(df_fixed.loc[df_fixed["record_id"] == "R003", "thread_count"].iloc[0])


def test_normalize_categories():
    """Test that categories are trimmed and lowercased."""
    df = pd.DataFrame({"fabric": [" Cotton ", "SILK", "polyester "]})
    df_norm = normalize_categories(df, ["fabric"])
    assert df_norm["fabric"].tolist() == ["cotton", "silk", "polyester"]

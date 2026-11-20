"""
src/powerloom/eda.py
Exploratory Data Analysis for the Power Loom Output dataset (G018 MP243).

Run:
    python -m src.powerloom.eda

Produces summary statistics, feature distributions, and a correlation table
to guide model selection and feature engineering decisions.

Author: Gitali Kumari (Model Lead) — S24CSEU0775
"""

from __future__ import annotations

import hashlib
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
TARGET_COL = PARAMS["data"]["target_col"]
GROUP_COL = PARAMS["data"]["group_col"]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV dataset from the given path."""
    df = pd.read_csv(path)
    return df


def print_section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ─── EDA functions ───────────────────────────────────────────────────────────


def describe_dataset(df: pd.DataFrame, name: str = "train") -> None:
    """Print shape, dtypes and basic statistics."""
    print_section(f"Dataset Overview — {name}")
    print(f"Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print(f"\nNumeric summary:\n{df.describe().round(2).to_string()}")


def categorical_summary(df: pd.DataFrame) -> None:
    """Print value counts for every text column."""
    print_section("Categorical Columns")
    for col in df.select_dtypes(include="object").columns:
        if col in ("record_id", "date"):
            continue
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())


def target_distribution(df: pd.DataFrame) -> None:
    """Describe the target variable distribution."""
    print_section(f"Target Distribution — '{TARGET_COL}'")
    tgt = df[TARGET_COL]
    print(f"Min:    {tgt.min():.2f}")
    print(f"Max:    {tgt.max():.2f}")
    print(f"Mean:   {tgt.mean():.2f}")
    print(f"Median: {tgt.median():.2f}")
    print(f"Std:    {tgt.std():.2f}")
    print(f"Skew:   {tgt.skew():.4f}")
    print(f"Kurtosis: {tgt.kurtosis():.4f}")

    # Identify obvious high outliers (potential 10x typos)
    p95 = tgt.quantile(0.95)
    outliers = df[tgt > p95 * 5]
    if len(outliers) > 0:
        print(f"\nSuspected target typos (> 5×p95={p95:.1f}):")
        print(
            outliers[["record_id", "date", TARGET_COL]].to_string(index=False)
        )


def feature_target_correlation(df: pd.DataFrame) -> None:
    """Print Pearson correlations between numeric features and target."""
    print_section("Feature-Target Correlations (Pearson)")
    numeric_df = df.select_dtypes(include=np.number)
    corrs = numeric_df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(
        key=abs, ascending=False
    )
    print(corrs.round(4).to_string())


def group_target_stats(df: pd.DataFrame) -> None:
    """Print target statistics grouped by the grouping column (shed)."""
    print_section(f"Target Stats by '{GROUP_COL}'")
    grp = df.groupby(GROUP_COL)[TARGET_COL].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    print(grp.round(2).to_string())


def compare_train_later(train: pd.DataFrame, later: pd.DataFrame) -> None:
    """Compare numeric distributions between train and later.csv."""
    print_section("Distribution Comparison: train vs later")
    numeric_cols = [
        c
        for c in train.select_dtypes(include=np.number).columns
        if c not in (TARGET_COL, "metres_checked_at_folding")
    ]
    rows = []
    for col in numeric_cols:
        rows.append(
            {
                "column": col,
                "train_mean": train[col].mean(),
                "later_mean": later[col].mean() if col in later.columns else None,
                "train_std": train[col].std(),
                "later_std": later[col].std() if col in later.columns else None,
            }
        )
    cmp = pd.DataFrame(rows).set_index("column").round(2)
    print(cmp.to_string())


def data_sha256(path: Path) -> str:
    """Return the SHA-256 hash of the raw file (for MLflow logging)."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ─── Main ─────────────────────────────────────────────────────────────────────


def run_eda() -> None:
    """Execute the full EDA pipeline and print results to stdout."""
    # Load data
    if not TRAIN_PATH.exists():
        print(
            f"[EDA] train.csv not found at {TRAIN_PATH}. "
            "Run `dvc pull` to fetch data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    train = load_data(TRAIN_PATH)
    later = load_data(LATER_PATH) if LATER_PATH.exists() else None

    # SHA-256 for MLflow
    sha = data_sha256(TRAIN_PATH)
    print(f"\ntrain.csv SHA-256: {sha}")

    describe_dataset(train, name="train")
    categorical_summary(train)
    target_distribution(train)
    feature_target_correlation(train)
    group_target_stats(train)

    if later is not None:
        describe_dataset(later, name="later")
        compare_train_later(train, later)

    print("\n[EDA] Done.")


if __name__ == "__main__":
    run_eda()

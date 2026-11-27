"""Data drift (PSI) and slice MAE monitoring for G018 MP243.

Run:
    python -m src.powerloom.drift

Author: Eshaan Mishra (Ops Lead) — S24CSEU0795
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from powerloom.data import normalize_categories

ROOT = Path(__file__).resolve().parents[2]

NUMERIC_FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
]
CATEGORICAL_FEATURES = ["fabric", "shed"]


def load_params() -> dict:
    """Load centralized pipeline parameters from params.yaml."""
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_psi(
    reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-6
) -> float:
    """Compute Population Stability Index (PSI) using quantile bins from reference."""
    ref_vals = reference.dropna().to_numpy(dtype=float)
    cur_vals = current.dropna().to_numpy(dtype=float)

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref_vals, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_vals, bins=edges)
    cur_counts, _ = np.histogram(cur_vals, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(ref_vals), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(len(cur_vals), 1), eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_and_slice_metrics() -> dict:
    """Calculate 10-bin PSI for numeric features and v1 MAE by shed."""
    params = load_params()
    train_df = pd.read_csv(ROOT / params["paths"]["train_clean"])
    later_df = pd.read_csv(ROOT / params["paths"]["later_raw"])
    v1_model = joblib.load(ROOT / params["paths"]["model"])

    bins = int(params["data"].get("psi_bins", 10))
    target_col = params["data"]["target_col"]
    group_col = params["data"]["group_col"]

    psi_scores = {
        col: round(compute_psi(train_df[col], later_df[col], bins=bins), 4)
        for col in NUMERIC_FEATURES
    }
    most_drifted_feature = max(psi_scores, key=psi_scores.get)

    features = normalize_categories(
        later_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy(),
        CATEGORICAL_FEATURES,
    )
    later_df = later_df.copy()
    later_df["_v1_pred"] = v1_model.predict(features)

    mae_by_group = {}
    for group_val in sorted(later_df[group_col].dropna().unique()):
        subset = later_df[later_df[group_col] == group_val]
        mae_by_group[str(group_val)] = round(
            float(mean_absolute_error(subset[target_col], subset["_v1_pred"])), 4
        )

    worst_served_group = max(mae_by_group, key=mae_by_group.get)

    return {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted_feature,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served_group,
    }


def update_findings_monitoring(metrics: dict) -> None:
    """Update monitoring PSI and slice MAE sections in reports/findings.yaml."""
    findings_path = ROOT / "reports/findings.yaml"
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f) or {}

    monitoring = findings.get("monitoring") or {}
    monitoring["psi"] = metrics["psi"]
    monitoring["most_drifted_feature"] = metrics["most_drifted_feature"]
    monitoring["mae_by_group"] = metrics["mae_by_group"]
    monitoring["worst_served_group"] = metrics["worst_served_group"]
    findings["monitoring"] = monitoring

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


def main() -> None:
    metrics = compute_drift_and_slice_metrics()
    update_findings_monitoring(metrics)

    print("10-bin PSI (train_clean vs later):")
    for col, val in metrics["psi"].items():
        print(f"  {col}: {val:.4f}")
    print(f"Most drifted feature: {metrics['most_drifted_feature']}")

    print("\nv1 MAE by shed on later.csv:")
    for group_val, val in metrics["mae_by_group"].items():
        print(f"  {group_val}: {val:.4f}")
    print(f"Worst served group: {metrics['worst_served_group']}")


if __name__ == "__main__":
    main()

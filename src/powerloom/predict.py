"""Holdout inference pipeline using the @champion model for G018 MP243.

Run:
    python -m src.powerloom.predict

Author: Divyanshi Maurya (Data Lead) — S24CSEU0803
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from powerloom.data import normalize_categories

ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]
CATEGORICAL_FEATURES = ["fabric", "shed"]


def load_params() -> dict:
    """Load centralized pipeline parameters from params.yaml."""
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_holdout_predictions() -> pd.DataFrame:
    """Load champion model, score holdout_features.csv, and write CSV output."""
    params = load_params()
    holdout_path = ROOT / params["paths"]["holdout"]
    model_path = ROOT / params["paths"]["model"]
    output_path = ROOT / params["paths"]["predictions"]

    holdout_df = pd.read_csv(holdout_path)
    champion_model = joblib.load(model_path)

    features = normalize_categories(
        holdout_df[FEATURE_COLUMNS].copy(), CATEGORICAL_FEATURES
    )
    preds = champion_model.predict(features)

    result_df = pd.DataFrame(
        {
            "record_id": holdout_df["record_id"],
            "prediction": preds.round(4),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df


def main() -> None:
    result_df = generate_holdout_predictions()
    print(f"Generated {len(result_df)} holdout predictions.")
    print(f"Columns: {list(result_df.columns)}")
    print(
        f"Prediction summary: min={result_df['prediction'].min():.2f}, "
        f"mean={result_df['prediction'].mean():.2f}, "
        f"max={result_df['prediction'].max():.2f}"
    )


if __name__ == "__main__":
    main()

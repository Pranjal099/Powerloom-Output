"""Retraining pipeline (v2) and promotion gate for G018 MP243.

Run:
    python -m src.powerloom.retrain

Author: Gitali Kumari (Model Lead) — S24CSEU0775
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from powerloom.data import normalize_categories, remove_leaking_column

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
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_params() -> dict:
    """Load centralized pipeline parameters from params.yaml."""
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_candidates(random_state: int) -> dict[str, object]:
    """Return candidate scikit-learn regressors for v2 retraining."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        ),
    }


def retrain_and_evaluate() -> dict:
    """Train v2 on train_clean + first 70% of later.csv and check promotion rule."""
    params = load_params()

    train_clean_path = ROOT / params["paths"]["train_clean"]
    raw_train_path = ROOT / params["paths"]["train_raw"]
    later_path = ROOT / params["paths"]["later_raw"]
    model_path = ROOT / params["paths"]["model"]

    target_col = params["data"]["target_col"]
    leaking_col = params["data"]["leaking_col"]
    experiment_name = params["model"]["experiment_name"]
    registered_name = params["model"]["registered_name"]
    random_state = int(params["model"]["random_state"])
    later_split = float(params["model"]["later_split"])
    promote_threshold = float(params["model"]["promote_threshold"])

    data_sha256 = hashlib.sha256(raw_train_path.read_bytes()).hexdigest()

    train_clean = pd.read_csv(train_clean_path)
    later_df = pd.read_csv(later_path)

    # Sort production stream chronologically before 70/30 split
    later_df = later_df.sort_values(
        ["date", "record_id"], kind="mergesort"
    ).reset_index(drop=True)

    train_clean = normalize_categories(
        remove_leaking_column(train_clean, leaking_col), CATEGORICAL_FEATURES
    )
    later_df = normalize_categories(
        remove_leaking_column(later_df, leaking_col), CATEGORICAL_FEATURES
    )

    split_idx = int(len(later_df) * later_split)
    later_train = later_df.iloc[:split_idx].copy()
    later_check = later_df.iloc[split_idx:].copy()

    # Evaluate existing v1 model on the held-out last 30% of later.csv
    v1_model = joblib.load(model_path)
    v1_preds = v1_model.predict(later_check[FEATURE_COLUMNS])
    mae_v1_on_check = float(mean_absolute_error(later_check[target_col], v1_preds))

    combined_train = pd.concat(
        [
            train_clean[FEATURE_COLUMNS + [target_col]],
            later_train[FEATURE_COLUMNS + [target_col]],
        ],
        ignore_index=True,
    )
    X_train = combined_train[FEATURE_COLUMNS]
    y_train = combined_train[target_col]
    X_check = later_check[FEATURE_COLUMNS]
    y_check = later_check[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    results = []
    for model_name, estimator in build_candidates(random_state).items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        with mlflow.start_run(run_name=f"{model_name}-v2") as run:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_check)
            mae = float(mean_absolute_error(y_check, preds))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("stage", "v2_retrain")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("later_split", later_split)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metric("mae_later", mae)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "model_name": model_name,
                    "mae": mae,
                    "run_id": run.info.run_id,
                    "pipeline": pipeline,
                }
            )
            print(f"v2 {model_name} MAE (last 30%): {mae:.4f}")

    best_v2 = min(results, key=lambda item: item["mae"])
    mae_v2_on_check = best_v2["mae"]

    registered_v2 = mlflow.register_model(
        model_uri=f"runs:/{best_v2['run_id']}/model",
        name=registered_name,
    )
    v2_version = int(registered_v2.version)

    improvement = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    if improvement >= promote_threshold:
        decision = "promote"
        client = MlflowClient()
        client.set_registered_model_alias(
            name=registered_name,
            alias="champion",
            version=registered_v2.version,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_v2["pipeline"], model_path)
    else:
        decision = "keep"

    return {
        "mae_v1_on_check": round(mae_v1_on_check, 4),
        "mae_v2_on_check": round(mae_v2_on_check, 4),
        "improvement_pct": round(improvement * 100.0, 2),
        "decision": decision,
        "best_v2_model": best_v2["model_name"],
        "best_v2_run_id": best_v2["run_id"],
        "v2_version": v2_version,
    }


def update_findings(summary: dict) -> None:
    """Update monitoring retrain results and champion version in reports/findings.yaml."""
    findings_path = ROOT / "reports/findings.yaml"
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f) or {}

    monitoring = findings.get("monitoring") or {}
    monitoring["mae_v1_on_check"] = summary["mae_v1_on_check"]
    monitoring["mae_v2_on_check"] = summary["mae_v2_on_check"]
    monitoring["decision"] = summary["decision"]
    findings["monitoring"] = monitoring

    if summary["decision"] == "promote":
        tracking = findings.get("tracking") or {}
        tracking["champion_version"] = summary["v2_version"]
        findings["tracking"] = tracking

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


def main() -> None:
    summary = retrain_and_evaluate()
    update_findings(summary)
    print(f"\nmae_v1_on_check: {summary['mae_v1_on_check']:.4f}")
    print(f"mae_v2_on_check: {summary['mae_v2_on_check']:.4f}")
    print(f"Improvement:     {summary['improvement_pct']:.2f}%")
    print(f"Decision:        {summary['decision']}")
    print(f"Registered v2:   version {summary['v2_version']} ({summary['best_v2_model']})")


if __name__ == "__main__":
    main()

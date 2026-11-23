
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Project configuration

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml", "r") as file:
    config = yaml.safe_load(file)

train_path = ROOT / config["paths"]["train_clean"]
raw_train_path = ROOT / config["paths"]["train_raw"]
model_path = ROOT / config["paths"]["model"]

target_col = config["data"]["target_col"]
experiment_name = config["model"]["experiment_name"]
registered_name = config["model"]["registered_name"]

random_state = config["model"]["random_state"]
test_size = config["model"]["test_size"]


# 2. Load dataset and calculate SHA256

df = pd.read_csv(train_path)

sha256 = hashlib.sha256(raw_train_path.read_bytes()).hexdigest()

print("Dataset shape:", df.shape)
print("SHA256:", sha256)


# 3. Features and target

numeric_features = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
]

categorical_features = [
    "fabric",
    "shed",
]

X = df[numeric_features + categorical_features]
y = df[target_col]


# 4. Train-validation split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))


# 5. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# 6. Models

models = {
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


# 7. MLflow setup

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(experiment_name)

results = []


# 8. Train and log models

for model_name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)

        mlflow.log_param("data_sha256", sha256)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("validation_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Prefix model parameters to avoid duplicate parameter names
        model_params = {
            f"model_{key}": value
            for key, value in estimator.get_params().items()
            if value is not None
        }

        mlflow.log_params(model_params)

        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        results.append({
            "model_name": model_name,
            "mae": mae,
            "run_id": run.info.run_id,
            "pipeline": pipeline,
        })

        print(f"{model_name} MAE: {mae:.4f}")


# 9. Select best model

best_result = min(results, key=lambda item: item["mae"])

best_model = best_result["pipeline"]
best_run_id = best_result["run_id"]

print("\nBest model:", best_result["model_name"])
print("Best MAE:", best_result["mae"])


# 10. Register best model

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_name,
)

client = MlflowClient()

client.set_registered_model_alias(
    name=registered_name,
    alias="champion",
    version=registered_model.version,
)


# 11. Export model

model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_path)

print("\nModel exported:", model_path)
print("Registered version:", registered_model.version)
print("MLflow alias: @champion")


# 12. Update findings.yaml tracking section

findings_path = ROOT / "reports/findings.yaml"
if findings_path.exists():
    with open(findings_path, "r") as f:
        findings_data = yaml.safe_load(f)

    if findings_data is None:
        findings_data = {}

    findings_data["tracking"] = {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": registered_name,
        "champion_version": int(registered_model.version),
    }

    with open(findings_path, "w") as f:
        yaml.safe_dump(findings_data, f, sort_keys=False)
    print("Updated reports/findings.yaml tracking section.")


# 12. Display results

print("\nMODEL COMPARISON")

for result in sorted(results, key=lambda item: item["mae"]):
    print(
        f"{result['model_name']}: "
        f"MAE = {result['mae']:.4f}"
    )

print("\nTRAINING COMPLETED SUCCESSFULLY!")

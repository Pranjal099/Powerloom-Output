# G018 — MP243 Power Loom Output (Guwahati)

## Group Information

| Field | Value |
|---|---|
| **Group ID** | G018 |
| **Problem ID** | MP243 |
| **Course** | SCSE3040 — MLOps Mini Project |
| **Faculty** | Amit Soni |
| **Batch** | 52 |

## Members

| Name | Enrolment No. | GitHub Username | Role |
|---|---|---|---|
| Divyanshi Maurya | S24CSEU0803 | @DivyanshiMaurya | Data Lead |
| Gitali Kumari | S24CSEU0775 | @Gitalikumari | Model Lead |
| Pranjal | S24CSEU1027 | @Pranjal099 | Service Lead |
| Eshaan Mishra | S24CSEU0795 | @InsaneSoul207 | Ops Lead |

## Problem Description

The **Guwahati Weavers' Cluster** wants to predict how many metres of cloth each loom will produce in a shift (`metres_per_shift`). Accurate predictions enable realistic delivery dates and fewer penalties.

This is a **regression** problem evaluated with **MAE** (Mean Absolute Error).

### Input Features

| Feature | Description |
|---|---|
| `loom_age_yr` | Age of the loom (years) |
| `thread_count` | Threads per inch |
| `power_cuts_min` | Power cuts during the shift (minutes) |
| `weaver_exp_yr` | Weaver's experience (years) |
| `humidity_pct` | Shed humidity (%) |
| `shift_hours` | Length of the shift (hours) |
| `fabric` | Fabric type (cotton / silk / polyester) |
| `shed` | Shed location (Shed 1 / Shed 2 / Shed 3) |

## Repository Layout

```
README.md                         # This file
requirements.txt                  # Pinned dependencies
params.yaml                       # Settings (paths, model config)
src/powerloom/__init__.py         # Python package
src/powerloom/data_checks.py      # Seven data quality checks
src/powerloom/eda.py              # Exploratory data analysis
tests/test_*.py                   # Pytest tests (3+)
data/raw/train.csv.dvc            # DVC pointer — training data
data/raw/later.csv.dvc            # DVC pointer — monitoring data
data/raw/holdout_features.csv     # Holdout (committed — no target)
.dvc/config                       # DVC configuration
mlflow.db                         # MLflow SQLite database
app/main.py                       # FastAPI prediction service
models/model.joblib               # Champion model
Dockerfile                        # Container build instructions
.dockerignore                     # Docker ignore list
.github/workflows/ci.yml          # GitHub Actions CI
k8s/deployment.yaml               # Kubernetes Deployment
k8s/service.yaml                  # Kubernetes Service
reports/findings.yaml             # Completed findings form
predictions/holdout_predictions.csv # Final holdout predictions
```

## Pipeline Stages

| Stage | Focus | Status |
|---|---|---|
| 1 | Repo setup, DVC, data quality checks | ✅ In progress |
| 2 | Model training, MLflow tracking, registry | 🔜 |
| 3 | FastAPI service, Docker, CI, Kubernetes | 🔜 |
| 4 | Monitoring, PSI, retrain v2, holdout predict | 🔜 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data quality checks
python -m src.powerloom.data_checks

# Run EDA
python -m src.powerloom.eda

# Run tests
pytest tests/
```

## Submission Deadline

**Sunday 29 November 2026, 11:59 PM**

# G018 — MP243: Power Loom Output Prediction (Guwahati Weavers' Cluster)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLOps](https://img.shields.io/badge/MLOps-Pipeline-orange.svg?logo=git&logoColor=white)](#pipeline-architecture)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Prediction%20API-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5.svg?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%20Pipeline-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/features/actions)

> **Course:** SCSE3040 — MLOps Mini Project  
> **Group ID:** `G018`  
> **Problem ID:** `MP243`  
> **Faculty:** Amit Soni  
> **Batch:** 52  
> **Submission Deadline:** Sunday, 29 November 2026, 11:59 PM  

---

## 👥 Group Members & Persona Roles

Every member commits from their own verified GitHub account, driving end-to-end ownership of their assigned component:

| Member | Enrolment No. | Student Email | GitHub Account | Assigned Role | Core Responsibilities |
|---|---|---|---|---|---|
| **Pranjal** | `S24CSEU1027` | `pranjalofficial32@gmail.com` | [@Pranjal099](https://github.com/Pranjal099) | **Service Lead (Repo Manager)** | FastAPI microservice, Pydantic schema validation rules, Pytest test suites, Dockerfile, Repository coordination |
| **Divyanshi Maurya** | `S24CSEU0803` | `divyanshimaurya716@gmail.com` | [@DivyanshiMaurya](https://github.com/DivyanshiMaurya) | **Data Lead** | DVC data versioning, 7 data quality checks, data cleaning pipeline (`clean.py`), `findings.yaml` data issues |
| **Gitali Kumari** | `S24CSEU0775` | `gitalikashyap14@gmail.com` | [@Gitalikumari](https://github.com/Gitalikumari) | **Model Lead** | Exploratory Data Analysis, model training (`train.py`), MLflow tracking, candidate evaluations, Model Registry & `@champion` |
| **Eshaan Mishra** | `S24CSEU0795` | `s24cseu0795@bennett.edu.in` | [@InsaneSoul207](https://github.com/InsaneSoul207) | **Ops Lead** | CI/CD GitHub Actions workflows, Kubernetes manifests, DVC configuration, drift monitoring (PSI), retraining automation |

---

## 🏢 Business Context & Problem Statement

The **Guwahati Weavers' Cluster** operates commercial power looms producing textiles across multiple sheds in Assam. The cluster takes wholesale orders measured strictly by the metre, with tight delivery schedules requested by buyers.

### The Problem
Power-loom cloth output varies shift-to-shift depending on loom condition, environmental variables (shed humidity), operational disruptions (power cuts), weaver experience, and raw fabric properties. Inaccurate manual estimates result in missed delivery deadlines, commercial customer disputes, and severe contractual delay penalties.

### The Solution
Develop, deploy, and maintain an **end-to-end production MLOps pipeline** to predict:
$$\text{metres\_per\_shift} \in \mathbb{R}^+$$
the quantity of cloth (in metres) a loom will weave during a single shift.

- **Task Type:** Continuous Supervised Regression
- **Primary Optimization Metric:** **MAE (Mean Absolute Error)** in metres — directly reflecting the average commercial deviation per shift:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$

---

## 📊 Dataset & Feature Dictionary

The project utilizes three distinct datasets under strict temporal separation:

| Dataset | Split Period | Records | Purpose | Tracking Method |
|---|---|---|---|---|
| `data/raw/train.csv` | Jan – Jun 2026 | 1,035 | Training & validation (noisy, real-world errors) | Tracked by DVC (`train.csv.dvc`) |
| `data/raw/later.csv` | Jul – Aug 2026 | 300 | Post-deployment monitoring & retraining | Tracked by DVC (`later.csv.dvc`) |
| `data/raw/holdout_features.csv` | Sep 2026 | 250 | Final evaluation set (no target column) | Tracked in Git (no target) |

### Feature Attributes

| Column Name | Data Type | Unit / Format | Description | Used in Inference? |
|---|---|---|---|:---:|
| `record_id` | String | `R\d{5}` | Unique shift log identifier | ❌ (Identifier) |
| `date` | Date | `YYYY-MM-DD` | Date of shift | ❌ (Temporal index) |
| `loom_age_yr` | Float | Years | Operating age of the power loom | ✅ |
| `thread_count` | Integer | Threads/inch | Yarn density set on loom reed | ✅ |
| `power_cuts_min` | Float | Minutes | Total power outage duration in shift | ✅ |
| `weaver_exp_yr` | Float | Years | Years of experience of assigned weaver | ✅ |
| `humidity_pct` | Float | Percentage | Average relative humidity recorded in shed | ✅ |
| `shift_hours` | Float | Hours | Operational duration of shift | ✅ |
| `fabric` | Categorical | Text | `cotton`, `silk`, `polyester` | ✅ |
| `shed` | Categorical | Text | `Shed 1`, `Shed 2`, `Shed 3` | ✅ |
| `metres_checked_at_folding` | Float | Metres | Length measured at folding table post-shift | ❌ **Leaking Column (Dropped)** |
| **`metres_per_shift`** | Float | Metres | **Target Variable:** Total cloth woven | **Prediction Target** |

---

## 🔍 Stage 1: Data Quality Audit (The 7 Data Issues)

As mandated by Student Handbook §7, `train.csv` was subjected to rigorous empirical auditing via `src/powerloom/data_checks.py`. All answers are verified and recorded in [`reports/findings.yaml`](reports/findings.yaml):

| # | Data Issue Kind | Status | Detailed Empirical Finding | Remediation Strategy |
|:---:|---|:---:|---|---|
| **1** | **`leaking_column`** | **PRESENT** ✅ | `metres_checked_at_folding` is measured *after* the shift at folding inspection. It is absent from `holdout_features.csv` and carries artificial correlation ($r = 0.334$). | Excluded from feature set $X$ during preprocessing. |
| **2** | **`unit_mixup`** | **ABSENT** ❌ | No shed displays $\ge 8\times$ magnitude differences. Maximum cross-shed ratio is $1.10\times$ on `power_cuts_min` (Shed 1 vs Shed 2). | None required. |
| **3** | **`sentinel_value`** | **PRESENT** ✅ | Value `99` appears repeatedly (4 times) in `thread_count`. | Handled gracefully via median imputation during preprocessing. |
| **4** | **`duplicate_rows`** | **PRESENT** ✅ | Exactly **35 duplicate records** identified via `df.duplicated().sum()`. | Dropped via `df.drop_duplicates()`, reducing train size to exactly 1,000 unique records. |
| **5** | **`stuck_sensor`** | **PRESENT** ✅ | `thread_count` sensor was frozen at constant value `147` from **2026-03-14** to **2026-03-28** (93 consecutive records). | Masked during the frozen date window and imputed with the true median. |
| **6** | **`target_typos`** | **PRESENT** ✅ | **7 records** have target values between $416.0\text{m}$ and $578.0\text{m}$ ($>5\times p_{95} = 65\text{m}$), typed with an accidental extra trailing zero ($10\times$ typo). | Divided by $10.0$ ($\rightarrow 41.6\text{--}57.8\text{m}$), restoring true statistical distribution. |
| **7** | **`spelling_variants`** | **ABSENT** ❌ | Categorical features `fabric` (`cotton`, `silk`, `polyester`) and `shed` (`Shed 1`, `Shed 2`, `Shed 3`) perfectly match `later.csv`. | None required. |

---

## 🛠️ Complete MLOps Pipeline Lifecycle

```mermaid
flowchart TD
    subgraph S1["Stage 1: Versioning & Quality Audit"]
        A["Raw Data (train.csv, later.csv)"] -->|DVC Tracking| B["data/raw/*.dvc"]
        A -->|data_checks.py| C["reports/findings.yaml"]
        A -->|eda.py| D["Exploratory Data Analysis"]
    end

    subgraph S2["Stage 2: Cleaning, Training & Registry"]
        B -->|clean.py| E["Cleaned Train Data (1,000 rows)"]
        E -->|train.py| F["MLflow Tracking (sqlite:///mlflow.db)"]
        F -->|Select Best Val MAE| G["Model Registry: powerloom-regressor"]
        G -->|Set Alias| H["@champion (v1)"]
        H -->|Export| I["models/model.joblib"]
        E -->|pytest| J["Unit Test Suite (tests/)"]
    end

    subgraph S3["Stage 3: Service, Container & CI/CD"]
        I -->|FastAPI| K["app/main.py (POST /predict, GET /health)"]
        K -->|Dockerfile| L["Docker Container: mp243-api:1.0"]
        K -->|GitHub Actions| M[".github/workflows/ci.yml"]
        L -->|Manifests| N["k8s/deployment.yaml & service.yaml (4 Replicas)"]
    end

    subgraph S4["Stage 4: Monitoring, Retrain & Holdout Inference"]
        O["later.csv Monitoring Data"] -->|PSI Calculation| P["Feature Drift Detection"]
        O -->|Group MAE| Q["Worst-Served Shed Analysis"]
        O -->|Retrain v2 (70/30)| R["Promotion Rule: >= 10% MAE drop"]
        R -->|Final Predict| S["predictions/holdout_predictions.csv"]
    end

    S1 --> S2 --> S3 --> S4
```

---

## 📁 Repository Directory Structure

The repository follows the strict layout specification defined in Student Handbook §5:

```text
Powerloom-Output/
├── .dvc/
│   ├── .gitignore                       # Excludes DVC cache from Git
│   └── config                           # DVC configuration (autostage enabled)
├── .dvcignore                           # DVC ignore rules
├── .gitignore                           # Excludes raw CSVs, caches, pycache
├── README.md                            # Comprehensive project documentation
├── requirements.txt                     # Fully pinned production dependencies
├── params.yaml                          # Centralized configuration (paths, hyperparameters)
├── data/
│   └── raw/
│       ├── train.csv.dvc                # DVC pointer for training dataset
│       ├── later.csv.dvc                # DVC pointer for monitoring dataset
│       └── holdout_features.csv         # Holdout test features (tracked in Git)
├── src/
│   └── powerloom/
│       ├── __init__.py                  # Package entry point
│       ├── data_checks.py               # Empirical checks for all 7 data issues
│       ├── eda.py                       # Exploratory Data Analysis & statistical summaries
│       ├── clean.py                     # Production data cleaning pipeline
│       └── train.py                     # MLflow experiment tracking & model training
├── tests/
│   ├── test_clean.py                    # Unit tests for data cleaning logic
│   └── test_model.py                    # Unit tests for model inference & contracts
├── models/
│   └── model.joblib                     # Serialized champion model artifact
├── mlflow.db                            # SQLite MLflow tracking database (committed)
├── app/
│   └── main.py                          # FastAPI service with Pydantic validation (Stage 3)
├── Dockerfile                           # Container definition based on python:3.12-slim
├── .dockerignore                        # Container build ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml                       # Automated CI workflow running pytest
├── k8s/
│   ├── deployment.yaml                  # Kubernetes Deployment (mp243-api, 4 replicas)
│   └── service.yaml                     # Kubernetes Service (mp243-svc, port 80:8000)
├── reports/
│   └── findings.yaml                    # Automated evaluation findings form
└── predictions/
    └── holdout_predictions.csv          # Champion predictions for holdout dataset (Stage 4)
```

---

## ⚡ Quick Start & Development Setup

### 1. Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/Pranjal099/Powerloom-Output.git
cd Powerloom-Output

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all pinned dependencies
pip install -r requirements.txt
```

### 2. Pulling Raw Data via DVC
The raw datasets are tracked via DVC to keep large data out of Git history:

```bash
# Pull raw datasets from remote storage
dvc pull
```

### 3. Running Data Quality Checks & EDA
Execute the empirical audit scripts to verify the dataset condition:

```bash
# Run the 7 data quality checks
python3 src/powerloom/data_checks.py

# Run Exploratory Data Analysis
python3 src/powerloom/eda.py
```

### 4. Running Data Cleaning & Model Training
Execute the cleaning and MLflow tracking pipeline:

```bash
# Clean raw training data
python3 src/powerloom/clean.py

# Train candidate models, log to MLflow, register champion
python3 src/powerloom/train.py

# Launch the MLflow UI to inspect runs and artifacts
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 5. Running Automated Unit Tests
Run the test suite with `pytest`:

```bash
PYTHONPATH=src pytest -v tests/
```

---

## 🔒 Business Validation Rules for Serving (Stage 3)

The client specifies two strict operational constraints that are enforced directly at the **Pydantic schema layer** of the `POST /predict` endpoint, ensuring that invalid inputs return HTTP `422 Unprocessable Entity` and never touch the model:

1. **Shift Length Constraint:** `shift_hours <= 20` (Shifts exceeding 20 hours are physically impossible in the cluster).
2. **Power Cuts Constraint:** `power_cuts_min >= 0` (Negative outage durations represent input typing errors).

---

## ☸️ Production Deployment Specifications

| Deployment Parameter | Required Specification |
|---|---|
| **Docker Image Name & Tag** | `mp243-api:1.0` |
| **Kubernetes Deployment Name** | `mp243-api` |
| **Kubernetes Service Name** | `mp243-svc` |
| **Pod Replica Count** | `4` |
| **Container Port** | `8000` |
| **Service Port** | `80` (forwarding to targetPort `8000`) |
| **Health Probes** | Liveness (`/health`, initial delay: 15s) and Readiness (`/health`, initial delay: 5s) |

---

## 📜 Academic Integrity & Submission Compliance

- **Fingerprint Verification:** All experiments log the verified SHA-256 hash of `train.csv` (`9cfff4b5ae21b59840b44a5eadf311542526f993f3bab7df6525815ebaa6c957`).
- **Individual Contribution:** Every persona commits their respective modules exclusively using their verified student / personal GitHub email address.
- **Reproducibility:** All random states are pinned (`random_state=42`), package versions pinned with `==`, and parameters centrally configured in [`params.yaml`](params.yaml).

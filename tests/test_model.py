"""Unit tests for model loading and prediction inference contract (MP243).

Authored by: Pranjal (Service Lead) & Gitali Kumari (Model Lead)
Verifies:
  - Champion model loads from models/model.joblib.
  - Generates valid numeric predictions matching input length.
  - Predicts physically reasonable cloth production (0 < metres < 150m).
  - Handles unseen categorical levels or typical production feature inputs.
No raw CSV dependencies (pure synthetic fixtures).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def champion_model():
    """Load the champion model pipeline from disk."""
    model_path = ROOT / "models/model.joblib"
    assert model_path.exists(), f"Model file not found at {model_path}"
    return joblib.load(model_path)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a sample feature DataFrame matching the inference contract."""
    return pd.DataFrame(
        {
            "loom_age_yr": [4.5, 12.0, 7.5],
            "thread_count": [128.0, 115.0, 132.0],
            "power_cuts_min": [25.0, 60.0, 10.0],
            "weaver_exp_yr": [14.0, 8.0, 18.0],
            "humidity_pct": [62.0, 58.0, 65.0],
            "shift_hours": [8.0, 10.0, 9.0],
            "fabric": ["cotton", "silk", "polyester"],
            "shed": ["Shed 1", "Shed 2", "Shed 3"],
        }
    )


def test_model_predict_shape(champion_model, sample_feature_df):
    """Test that model predictions match the input batch size."""
    preds = champion_model.predict(sample_feature_df)
    assert len(preds) == len(sample_feature_df)
    assert isinstance(preds, np.ndarray)


def test_model_predict_range(champion_model, sample_feature_df):
    """Test that predictions are positive and within realistic physical output range."""
    preds = champion_model.predict(sample_feature_df)
    assert (preds > 0.0).all()
    assert (preds < 150.0).all()


def test_model_unseen_category_handling(champion_model):
    """Test that OneHotEncoder handles unknown categories gracefully without error."""
    df_unknown = pd.DataFrame(
        {
            "loom_age_yr": [5.0],
            "thread_count": [120.0],
            "power_cuts_min": [30.0],
            "weaver_exp_yr": [10.0],
            "humidity_pct": [60.0],
            "shift_hours": [8.0],
            "fabric": ["linen"],  # Unseen fabric type
            "shed": ["Shed 4"],   # Unseen shed
        }
    )
    preds = champion_model.predict(df_unknown)
    assert len(preds) == 1
    assert not np.isnan(preds[0])

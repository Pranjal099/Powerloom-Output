"""Integration tests for the FastAPI prediction service (MP243).

Authored by: Gitali Kumari (Model Lead) — S24CSEU0775
Verifies:
  - GET /health returns status 'ok' with loaded champion model.
  - POST /predict returns a realistic numeric cloth output prediction.
  - Client Rule 1: shift_hours > 20 returns HTTP 422.
  - Client Rule 2: power_cuts_min < 0 returns HTTP 422.
  - Boundary values shift_hours == 20 and power_cuts_min == 0 are accepted.
No raw CSV dependencies.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Provide a TestClient running the FastAPI app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_payload(client: TestClient):
    """Verify POST /predict returns a valid cloth output prediction."""
    payload = {
        "loom_age_yr": 5.0,
        "thread_count": 128,
        "power_cuts_min": 25.0,
        "weaver_exp_yr": 12.0,
        "humidity_pct": 62.0,
        "shift_hours": 8.5,
        "fabric": "cotton",
        "shed": "Shed 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
    assert 0.0 < body["prediction"] < 150.0


@pytest.mark.parametrize("invalid_shift_hours", [20.1, 24.0])
def test_predict_rejects_shift_hours_above_20(
    client: TestClient, invalid_shift_hours: float
):
    """Client Rule 1: shift_hours > 20 must be rejected with HTTP 422."""
    payload = {
        "loom_age_yr": 5.0,
        "thread_count": 120,
        "power_cuts_min": 15.0,
        "weaver_exp_yr": 8.0,
        "humidity_pct": 60.0,
        "shift_hours": invalid_shift_hours,
        "fabric": "silk",
        "shed": "Shed 2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_power_cuts(client: TestClient):
    """Client Rule 2: power_cuts_min < 0 must be rejected with HTTP 422."""
    payload = {
        "loom_age_yr": 6.0,
        "thread_count": 115,
        "power_cuts_min": -5.0,
        "weaver_exp_yr": 10.0,
        "humidity_pct": 58.0,
        "shift_hours": 8.0,
        "fabric": "polyester",
        "shed": "Shed 3",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_accepts_boundary_values(client: TestClient):
    """Boundary values shift_hours == 20 and power_cuts_min == 0 are valid."""
    payload = {
        "loom_age_yr": 4.0,
        "thread_count": 130,
        "power_cuts_min": 0.0,
        "weaver_exp_yr": 15.0,
        "humidity_pct": 64.0,
        "shift_hours": 20.0,
        "fabric": "cotton",
        "shed": "Shed 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)

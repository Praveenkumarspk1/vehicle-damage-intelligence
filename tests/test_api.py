"""Integration tests for the Flask REST API endpoints."""

import pytest
from api.flask_api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_index(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "models" in data


def test_api_pricing_full(client):
    res = client.get("/pricing")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_records"] >= 80
    assert "Maruti Suzuki" in data["makes"]


def test_api_pricing_filter_by_make(client):
    res = client.get("/pricing?make=Hyundai")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["count"] >= 12
    assert all("Hyundai" in r["make"] for r in data["records"])


def test_api_pricing_query_specific(client):
    res = client.get("/pricing?make=Tata&class=bonnet-dent")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["record"]["damage_class"] == "bonnet-dent"


def test_api_predict_missing_image_returns_400(client):
    res = client.post("/predict", json={})
    assert res.status_code == 400


def test_api_compare_missing_images_returns_400(client):
    res = client.post("/compare", json={})
    assert res.status_code == 400


def test_api_report_generation(client):
    payload = {"results": {"inspection_test": {"damage_count": 2, "cost": 4500}}}
    res = client.post("/report", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "report_path" in data

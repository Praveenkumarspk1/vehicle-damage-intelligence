import importlib
from pathlib import Path

import numpy as np
import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    import yaml
    config_path = ROOT / "config.yaml"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def test_get_panel_material_bumper():
    module = _import_module("cost_engine.indian_repair_quote")
    assert module.get_panel_material("front-bumper-dent") == "Detachable Plastic / Composite"
    assert module.get_panel_material("bonnet-dent") == "Bolt-On Sheet Metal Panel"
    assert module.get_panel_material("Front-windscreen-damage") == "Laminated Safety Glass"


def test_estimate_severity_thresholds():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    minor, rank_minor, pct_minor = module.estimate_severity(0.01, "scratch", config)
    assert minor == "Minor"
    assert rank_minor == 1
    moderate, rank_mod, pct_mod = module.estimate_severity(0.05, "dent", config)
    assert moderate == "Moderate"
    assert rank_mod == 2
    severe, rank_sev, pct_sev = module.estimate_severity(0.10, "crack", config)
    assert severe == "Severe"
    assert rank_sev == 3


def test_calculate_repair_quote_minor():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    model_name = list(config["vehicle_models"].keys())[0]
    tier = list(config["location_tiers"].keys())[0]
    workshop = list(config["workshop_types"].keys())[0]
    quote = module.calculate_indian_repair_quote("front-bumper-dent", "Minor", 0.01, model_name, tier, workshop, config)
    assert quote.total_cost > 0
    assert quote.action == "Paintless Dent Repair (PDR) / Spot Touch-up"
    assert quote.gst_amount == pytest.approx(quote.total_cost * 0.18 / 1.18, rel=0.01)


def test_calculate_repair_quote_severe_structural():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    model_name = list(config["vehicle_models"].keys())[0]
    tier = list(config["location_tiers"].keys())[0]
    workshop = list(config["workshop_types"].keys())[0]
    quote = module.calculate_indian_repair_quote("roof-dent", "Severe", 0.15, model_name, tier, workshop, config)
    assert "Structural" in quote.panel_material
    assert "Frame Pull" in quote.action or "Replacement" in quote.action


def test_calculate_repair_quote_glass():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    model_name = list(config["vehicle_models"].keys())[0]
    tier = list(config["location_tiers"].keys())[0]
    workshop = list(config["workshop_types"].keys())[0]
    quote = module.calculate_indian_repair_quote("Front-windscreen-damage", "Severe", 0.20, model_name, tier, workshop, config)
    assert "Glass" in quote.panel_material
    assert "Replacement" in quote.action


def test_extract_visual_embedding_returns_128d():
    module = _import_module("cost_engine.indian_repair_quote")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    emb = module.extract_visual_embedding(img, [5, 5, 50, 50])
    assert len(emb) == 128


def test_cosine_similarity_identical():
    module = _import_module("cost_engine.indian_repair_quote")
    vec = [1.0, 0.0, 0.0]
    assert module.cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    module = _import_module("cost_engine.indian_repair_quote")
    assert module.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_hybrid_reid_match_new_damage():
    module = _import_module("cost_engine.indian_repair_quote")
    baseline = []
    current = []
    result = module.hybrid_reid_match(baseline, current)
    assert len(result[0]) == 0
    assert len(result[1]) == 0
    assert result[2] == 0.0


def test_project_to_3d_coordinates():
    module = _import_module("cost_engine.indian_repair_quote")
    coords = module.project_to_3d_coordinates("front", "front-bumper-dent", [0.5, 0.5])
    assert len(coords) == 3
    assert coords[0] >= 0.0
    assert coords[2] >= 0.0


def test_build_vehicle_mesh_traces():
    module = _import_module("cost_engine.indian_repair_quote")
    traces = module.build_vehicle_mesh_traces()
    assert len(traces) > 0
    assert "xs" in traces[0]
    assert "ys" in traces[0]
    assert "zs" in traces[0]


def test_validate_image_quality_accepts_good_image():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    img = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    result = module.validate_image_quality(img, config)
    assert isinstance(result.is_acceptable, bool)
    assert result.blur_score > 0
    assert len(result.issues) >= 0


def test_check_specular_glare_detects_highlights():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    result = module.check_specular_glare(img, [0, 0, 64, 64], config)
    assert isinstance(result, bool)


def test_estimate_turnaround_and_downtime():
    module = _import_module("cost_engine.indian_repair_quote")
    config = _load_config()
    model_name = list(config["vehicle_models"].keys())[0]
    quote = module.RepairQuote("test", "Steel", 2.0, 720.0, 0.0, 5000.0, 936.0, 6656.0, "test")
    from cost_engine.indian_repair_quote import DamageRecord
    record = DamageRecord(
        damage_id="test_1", class_name="bonnet-dent", slot="front",
        rel_center=[0.5, 0.5], rel_area=0.05, size_bucket="Moderate",
        size_rank=2, confidence=0.9, bbox=[0, 0, 100, 100], quote=quote,
    )
    result = module.estimate_turnaround_and_downtime([record], model_name, 1800.0, config)
    assert result["tat_days"] > 0
    assert result["downtime_loss"] > 0


def test_evaluate_insurance_claim():
    module = _import_module("cost_engine.indian_repair_quote")
    result = module.evaluate_insurance_claim(total_repair_quote=5000.0, vehicle_cc=1500, current_ncb_pct=35.0, annual_od_premium=14000.0)
    assert "decision" in result
    assert "claim_penalty" in result
    assert result["claim_penalty"] > 0
    big_result = module.evaluate_insurance_claim(total_repair_quote=50000.0, vehicle_cc=1500, current_ncb_pct=35.0, annual_od_premium=14000.0)
    assert "FILE" in big_result["decision"]

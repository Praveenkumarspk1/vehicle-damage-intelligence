"""Unit tests for the Indian Bodyshop Parts & Pricing Dataset and Loader."""

from pathlib import Path
import pytest

from data.indian_price_loader import IndianPricingCatalog, get_indian_pricing_catalog
from cost_engine.indian_repair_quote import calculate_indian_repair_quote


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_file_exists_and_loads():
    catalog = get_indian_pricing_catalog()
    assert catalog.total_records >= 80
    makes = catalog.get_makes()
    assert "Maruti Suzuki" in makes
    assert "Tata Motors" in makes
    assert "Mahindra" in makes
    assert "Hyundai" in makes
    assert "Toyota" in makes
    assert "German Luxury" in makes


def test_catalog_query_exact_and_partial():
    catalog = get_indian_pricing_catalog()
    record = catalog.query("Maruti Suzuki", "bonnet-dent")
    assert record is not None
    assert record.make == "Maruti Suzuki"
    assert record.oem_part_price_inr == 4200.0
    assert record.gst_rate_pct == 18.0
    assert record.irda_labor_hours == 4.0
    assert "Immediate Stock" in record.stock_availability


def test_catalog_filter_by_make():
    catalog = get_indian_pricing_catalog()
    tata_records = catalog.filter_by_make("Tata Motors")
    assert len(tata_records) >= 12
    for r in tata_records:
        assert r.make == "Tata Motors"
        assert r.oem_part_price_inr > 0


def test_cost_engine_uses_indian_catalog():

    empty_config = {
        "location_tiers": {"Tier 1": {"denting_labor_hr": 480.0, "paint_booth_panel": 2800.0, "multiplier": 1.15}},
        "workshop_types": {"Authorized OEM Dealer (ASS)": {"labor_markup": 1.20, "parts_markup": 1.0, "gst_rate": 0.18}},
        "vehicle_models": {"Maruti Suzuki": {"parts": {}}},
    }
    quote = calculate_indian_repair_quote(
        class_name="bonnet-dent",
        severity="Severe",
        rel_area=0.15,
        model_name="Maruti Suzuki",
        location_tier="Tier 1",
        workshop_type="Authorized OEM Dealer (ASS)",
        config=empty_config,
    )
    assert quote.part_cost == 4200.0
    assert quote.total_cost > 4200.0
    assert quote.gst_amount > 0

"""Indian Automotive Parts & Bodyshop Pricing Dataset Loader.

Standardized against:
- IRDAI (Insurance Regulatory and Development Authority of India) Motor Surveyor Norms
- GIPSA (General Insurance Public Sector Association) Schedule of Labor & Paint Charges
- SIAM (Society of Indian Automobile Manufacturers) Parts Classification Standards

Data files bundled in this project:
- indian_bodyshop_parts_pricing.csv  — 84 parts across 6 Indian OEMs (IRDAI/GIPSA norms)
- gipsa_bodyshop_labor_schedule.csv  — GIPSA 4-tier labor rate schedule
- irdai_depreciation_norms.json      — IMT age-based depreciation schedule
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "indian_bodyshop_parts_pricing.csv"
GIPSA_SCHEDULE_PATH = PROJECT_ROOT / "data" / "gipsa_bodyshop_labor_schedule.csv"
IRDAI_DEPRECIATION_PATH = PROJECT_ROOT / "data" / "irdai_depreciation_norms.json"

OFFICIAL_DATA_SOURCES = {
    "data_gov_in": "https://data.gov.in/keywords/automobile-parts",
    "irdai_portal": "https://irdai.gov.in/",
    "siam_india": "https://www.siam.in/",
    "gipsa_bodyshop_norms": "https://irdai.gov.in/document-detail?documentId=1164993",
}


@dataclass
class PartPriceRecord:
    make: str
    model: str
    segment: str
    damage_class: str
    panel_name: str
    material: str
    oem_part_price_inr: float
    aftermarket_fng_price_inr: float
    irda_labor_hours: float
    denting_labor_tier1_inr: float
    denting_labor_tier2_inr: float
    denting_labor_tier3_inr: float
    paint_booth_inr: float
    lead_time_tier1_days: int
    lead_time_tier2_tier3_days: int
    stock_availability: str
    gst_rate_pct: float


@dataclass
class GipsaTierRecord:
    """Single row from gipsa_bodyshop_labor_schedule.csv."""
    city_category: str
    location_tier: str
    major_cities: str
    hourly_denting_rate_inr: float
    paint_booth_per_panel_inr: float
    oem_multiplier: float
    fng_multiplier: float
    gst_rate_pct: float


def load_gipsa_schedule() -> List[GipsaTierRecord]:
    """Load GIPSA bodyshop labor rate schedule from CSV.

    Returns a list of GipsaTierRecord for each city tier (Tier 1–4).
    Used by the IRDAI quote engine to derive location-specific labor rates.
    """
    records: List[GipsaTierRecord] = []
    if not GIPSA_SCHEDULE_PATH.exists():
        return records
    with open(GIPSA_SCHEDULE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(GipsaTierRecord(
                    city_category=row["city_category"].strip(),
                    location_tier=row["location_tier"].strip(),
                    major_cities=row["major_cities"].strip(),
                    hourly_denting_rate_inr=float(row["hourly_denting_rate_inr"]),
                    paint_booth_per_panel_inr=float(row["paint_booth_per_panel_inr"]),
                    oem_multiplier=float(row["oem_multiplier"]),
                    fng_multiplier=float(row["fng_multiplier"]),
                    gst_rate_pct=float(row["gst_rate_pct"]),
                ))
            except (ValueError, KeyError):
                continue
    return records


def load_irdai_depreciation() -> Dict[str, Any]:
    """Load IRDAI IMT depreciation norms from JSON.

    Returns the full depreciation schedule including:
    - material_depreciation: flat/age-based depreciation per material type
    - compulsory_deductibles_inr: mandatory excess per vehicle CC
    - no_claim_bonus_slabs_pct: NCB slab progression list

    Used by calculate_indian_repair_quote() to apply depreciation to part costs.
    """
    if not IRDAI_DEPRECIATION_PATH.exists():
        return {}
    with open(IRDAI_DEPRECIATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_irdai_depreciation_rate(material: str, vehicle_age_years: float = 0.0) -> float:
    """Get the IRDAI depreciation rate (0.0–1.0) for a given material and vehicle age.

    Args:
        material: Panel material string (e.g. 'Detachable Plastic / Composite')
        vehicle_age_years: Vehicle age in years (0 = brand new)

    Returns:
        Depreciation fraction (0.0 = no depreciation, 0.5 = 50% depreciation)
    """
    norms = load_irdai_depreciation()
    mat_norms = norms.get("material_depreciation", {})


    if any(k in material for k in ("Plastic", "Rubber", "Composite", "Nylon")):
        for key, val in mat_norms.items():
            if "Rubber" in key or "Plastic" in key:
                return float(val.get("depreciation_percent", 50.0)) / 100.0


    if "Glass" in material or "Acrylic" in material:
        return 0.0


    if "Fibre" in material or "Fiber" in material:
        for key, val in mat_norms.items():
            if "Fibre" in key or "Fiber" in key:
                return float(val.get("depreciation_percent", 30.0)) / 100.0


    age_schedule = {}
    for key, val in mat_norms.items():
        if "Metal" in key or "Sheet" in key:
            age_schedule = val.get("depreciation_schedule_by_vehicle_age", {})
            break
    if age_schedule:
        if vehicle_age_years < 0.5:
            return 0.0
        elif vehicle_age_years < 1:
            return age_schedule.get("6 months to 1 year", 5.0) / 100.0
        elif vehicle_age_years < 2:
            return age_schedule.get("1 year to 2 years", 10.0) / 100.0
        elif vehicle_age_years < 3:
            return age_schedule.get("2 years to 3 years", 15.0) / 100.0
        elif vehicle_age_years < 4:
            return age_schedule.get("3 years to 4 years", 25.0) / 100.0
        elif vehicle_age_years < 5:
            return age_schedule.get("4 years to 5 years", 35.0) / 100.0
        elif vehicle_age_years <= 10:
            return age_schedule.get("5 years to 10 years", 40.0) / 100.0
        else:
            return age_schedule.get("Over 10 years", 50.0) / 100.0

    return 0.0


def get_gipsa_labor_rate(tier_keyword: str = "Tier 1") -> Optional[GipsaTierRecord]:
    """Get GIPSA labor schedule record for a given tier keyword.

    Args:
        tier_keyword: e.g. 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'
                      Also accepts verbose keys like 'Tier 1 (Metros: ...)'

    Returns:
        Matching GipsaTierRecord or None
    """
    schedule = load_gipsa_schedule()
    if not schedule:
        return None
    keyword = tier_keyword.strip().lower()


    bare = keyword.replace(" ", "")
    for record in schedule:
        rec_tier = record.location_tier.strip().lower().replace(" ", "")
        if bare == rec_tier:
            return record
        if keyword in record.city_category.lower():
            return record

    for record in schedule:
        if record.location_tier.strip() == tier_keyword.strip():
            return record
    return schedule[0]


class IndianPricingCatalog:
    """In-memory queryable repository of Indian automotive bodyshop parts and pricing.

    Integrates:
    - Parts pricing from indian_bodyshop_parts_pricing.csv
    - GIPSA labor rates from gipsa_bodyshop_labor_schedule.csv
    - IRDAI depreciation norms from irdai_depreciation_norms.json
    """

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or DEFAULT_DATASET_PATH
        self._records: List[PartPriceRecord] = []
        self._by_key: Dict[str, PartPriceRecord] = {}
        self._gipsa_schedule: List[GipsaTierRecord] = []
        self._irdai_norms: Dict[str, Any] = {}
        self.load()

    def load(self) -> List[PartPriceRecord]:
        """Load all datasets: parts pricing, GIPSA schedule, IRDAI norms."""
        self._records.clear()
        self._by_key.clear()


        self._gipsa_schedule = load_gipsa_schedule()


        self._irdai_norms = load_irdai_depreciation()


        if not self.csv_path.exists():
            return []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = PartPriceRecord(
                        make=row["make"].strip(),
                        model=row["model"].strip(),
                        segment=row["segment"].strip(),
                        damage_class=row["damage_class"].strip(),
                        panel_name=row["panel_name"].strip(),
                        material=row["material"].strip(),
                        oem_part_price_inr=float(row["oem_part_price_inr"]),
                        aftermarket_fng_price_inr=float(row["aftermarket_fng_price_inr"]),
                        irda_labor_hours=float(row["irda_labor_hours"]),
                        denting_labor_tier1_inr=float(row["denting_labor_tier1_inr"]),
                        denting_labor_tier2_inr=float(row["denting_labor_tier2_inr"]),
                        denting_labor_tier3_inr=float(row["denting_labor_tier3_inr"]),
                        paint_booth_inr=float(row["paint_booth_inr"]),
                        lead_time_tier1_days=int(row["lead_time_tier1_days"]),
                        lead_time_tier2_tier3_days=int(row["lead_time_tier2_tier3_days"]),
                        stock_availability=row["stock_availability"].strip(),
                        gst_rate_pct=float(row["gst_rate_pct"]),
                    )
                    self._records.append(record)
                    key = f"{record.make.lower()}:{record.damage_class.lower()}"
                    self._by_key[key] = record
                except (ValueError, KeyError):
                    continue

        return self._records

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def gipsa_schedule(self) -> List[GipsaTierRecord]:
        """Return loaded GIPSA labor rate schedule."""
        return self._gipsa_schedule

    @property
    def irdai_norms(self) -> Dict[str, Any]:
        """Return loaded IRDAI depreciation norms."""
        return self._irdai_norms

    def get_makes(self) -> List[str]:
        return sorted(list(set(r.make for r in self._records)))

    def query(
        self,
        make: Optional[str] = None,
        damage_class: Optional[str] = None,
    ) -> Optional[PartPriceRecord]:
        """Lookup by make and damage class (partial match supported)."""
        if not self._records:
            return None
        clean_class = (damage_class or "").strip().lower()
        if make:
            make_clean = make.strip().lower()
            for r in self._records:
                if (r.make.lower() in make_clean or make_clean in r.make.lower()) and r.damage_class.lower() == clean_class:
                    return r
        for r in self._records:
            if r.damage_class.lower() == clean_class:
                return r
        return None

    def get_depreciated_price(
        self,
        make: str,
        damage_class: str,
        vehicle_age_years: float = 0.0,
        use_fng: bool = False,
    ) -> Optional[float]:
        """Get IRDAI-depreciated part price for insurance claim purposes.

        Applies the IMT material depreciation schedule to the OEM (or FNG)
        part price to derive the insured value for claim assessment.

        Args:
            make: Vehicle make (e.g. 'Maruti Suzuki')
            damage_class: Damage class key (e.g. 'front-bumper-dent')
            vehicle_age_years: Vehicle age in years (0 = brand new)
            use_fng: If True, use aftermarket FNG price instead of OEM

        Returns:
            Depreciated price in INR, or None if record not found
        """
        record = self.query(make, damage_class)
        if record is None:
            return None
        base_price = record.aftermarket_fng_price_inr if use_fng else record.oem_part_price_inr
        dep_rate = get_irdai_depreciation_rate(record.material, vehicle_age_years)
        return round(base_price * (1.0 - dep_rate), 2)

    def get_labor_rate_for_tier(
        self,
        damage_class: str,
        make: str,
        tier_number: int = 2,
    ) -> Optional[float]:
        """Get GIPSA-sourced labor cost for a damage class at a specific city tier.

        Args:
            damage_class: e.g. 'front-bumper-dent'
            make: Vehicle make
            tier_number: 1 = Metro, 2 = State Capital, 3 = District, 4 = Rural

        Returns:
            Labor cost in INR, or None
        """
        record = self.query(make, damage_class)
        if record is None:
            return None
        tier_map = {1: record.denting_labor_tier1_inr, 2: record.denting_labor_tier2_inr, 3: record.denting_labor_tier3_inr}
        return tier_map.get(tier_number, record.denting_labor_tier2_inr)

    def filter_by_make(self, make: str) -> List[PartPriceRecord]:
        make_lower = make.strip().lower()
        return [r for r in self._records if make_lower in r.make.lower()]

    def as_dict_list(self) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self._records]

    @staticmethod
    def download_dataset(url: str, target_path: Optional[Path] = None) -> bool:
        """Download updated pricing dataset from an authorized Indian registry URL."""
        dest = target_path or DEFAULT_DATASET_PATH
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception:
            return False


catalog = IndianPricingCatalog()


def get_indian_pricing_catalog() -> IndianPricingCatalog:
    return catalog

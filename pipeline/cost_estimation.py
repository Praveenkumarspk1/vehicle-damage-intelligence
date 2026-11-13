"""Cost Estimation Bridge — Korean KRW pipeline → Indian INR GBR engine.

The legacy Korean cost estimator referenced model artifacts that no longer
exist in this project (KRW-based). This bridge re-exports the trained
Indian Gradient Boosting Regressor from ``cost_engine.gbr_utils`` while
maintaining full backward compatibility with the original API surface
(``CostEstimator.estimate(part, repair_method, ...)``).

All estimates now return ``currency: "INR"`` and delegate to the
India-adapted GBR model or the IRDAI rule-based quote engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_COMPANY = "Maruti Suzuki"
DEFAULT_FIRSTDAY = 20180101
DEFAULT_REPAIRDAY = 20230101


_KOREAN_PART_TO_DAMAGE_CLASS = {
    "Bonnet": "bonnet-dent",
    "Frontbumper": "front-bumper-dent",
    "Rearbumper": "rear-bumper-dent",
    "Frontfender": "fender-dent",
    "Rearfender": "fender-dent",
    "Reardoor": "doorouter-dent",
    "Frontdoor": "doorouter-dent",
    "Trunklid": "quaterpanel-dent",
    "Sidemirror": "Sidemirror-Damage",
    "Headlights": "Headlight-damage",
    "FrontWheel": "flat_tire",
}


_KOREAN_METHOD_TO_SEVERITY = {
    "교체": "Severe",
    "교환": "Severe",
    "탈착": "Moderate",
    "수리": "Moderate",
    "도장": "Moderate",
    "판금": "Severe",
    "조정": "Minor",
    "overhaul": "Severe",
    "replace": "Severe",
    "detach": "Moderate",
}


def _normalize_part(part: Any) -> str:
    return _KOREAN_PART_TO_DAMAGE_CLASS.get(str(part).strip(), "bonnet-dent")


def _normalize_method(method: Any) -> str:
    if method is None:
        return "Moderate"
    for korean, severity in _KOREAN_METHOD_TO_SEVERITY.items():
        if korean in str(method):
            return severity
    return "Moderate"


def resolve_cost_artifacts() -> dict[str, Any]:
    """Return resolved artifacts — now delegates to the Indian GBR model path."""
    from cost_engine.gbr_utils import GBR_MODEL_PATH
    return {
        "model": GBR_MODEL_PATH if GBR_MODEL_PATH.is_file() else None,
        "currency": "INR",
        "source": "cost_engine.gbr_utils.IndiaGBREstimator",
    }


def normalize_company(company: Any) -> str:
    """Normalize Korean company strings to Indian makes (backward compat).

    Accepts Korean inputs but maps them to Indian OEM categories.
    """
    if company is None:
        return DEFAULT_COMPANY
    text = str(company).strip()
    mapping = {
        "현대": "Hyundai",
        "기아": "Kia",
        "한국GM": "Chevrolet",
        "쌍용": "Mahindra",
        "르노": "Renault",
        "국산차": "Maruti Suzuki",
        "외제차": "Hyundai",
    }
    for korean, indian in mapping.items():
        if korean in text:
            return indian
    return text if text else DEFAULT_COMPANY


def normalize_part_name(part: Any) -> str:
    return _normalize_part(part)


def normalize_repair_method(method: Any) -> str:
    return _normalize_method(method)


class CostEstimator:
    """Indian INR cost estimator backed by the trained GBR model.

    Bridges the legacy Korean API: CostEstimator().estimate(part, method, company)
    now returns an Indian repair quote in INR.
    """

    def __init__(self) -> None:
        self._gbr = None
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from cost_engine.gbr_utils import IndiaGBREstimator
            self._gbr = IndiaGBREstimator()
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            self._gbr = None

    @property
    def available(self) -> bool:
        return self._gbr is not None and self._gbr.available

    def load(self) -> "CostEstimator":
        return self

    def estimate(
        self,
        part: Any,
        repair_method: Any,
        company: Any = None,
        firstday: Any = DEFAULT_FIRSTDAY,
        repairday: Any = DEFAULT_REPAIRDAY,
        model_path: Any = None,
        location_tier: str = "Tier 2",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate Indian repair cost (INR) using GBR model.

        Args:
            part: Korean part name or Indian damage class
            repair_method: Korean method string or severity
            company: Vehicle make (Korean or Indian)
            location_tier: City tier for labor rates
            model_path: Ignored (kept for backward compat)
        """
        damage_class = _normalize_part(part)
        severity = _normalize_method(repair_method)
        make = normalize_company(company)
        if self._gbr is not None and self._gbr.available:
            result = self._gbr.estimate(damage_class, severity, make, location_tier)
            result["currency"] = "INR"
            result["source"] = "gbr_model"
            result["part"] = str(part)
            result["repair_method"] = str(repair_method)
            result["company"] = make
            result["model_base_cost"] = result["estimated_cost_inr"]
            result["replace_price"] = 0
            result["fallback"] = False
            return result

        return _heuristic_estimate(damage_class, severity, make, location_tier)


def estimate_cost(
    part: Any,
    repair_method: Any,
    company: Any = None,
    firstday: Any = DEFAULT_FIRSTDAY,
    repairday: Any = DEFAULT_REPAIRDAY,
    model_path: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return CostEstimator().estimate(part, repair_method, company, firstday, repairday, model_path=model_path, **kwargs)


def _heuristic_estimate(damage_class: str, severity: str, make: str, location_tier: str) -> dict[str, Any]:
    from data.indian_price_loader import get_indian_pricing_catalog
    cat = get_indian_pricing_catalog()
    rec = cat.query(make, damage_class) if cat else None
    base = float(rec.oem_part_price_inr) if rec else 4000.0
    sev_mult = {"Minor": 0.5, "Moderate": 1.0, "Severe": 1.8}.get(severity, 1.0)

    from cost_engine.gbr_utils import MAKE_MULTIPLIER
    make_lower = make.lower()
    mult = 1.0
    for key, val in MAKE_MULTIPLIER.items():
        if key in make_lower:
            mult = val
            break
    estimated = round(base * sev_mult * mult, 2)
    return {
        "estimated_cost": estimated,
        "currency": "INR",
        "available": False,
        "source": "heuristic_fallback",
        "part": str(damage_class),
        "repair_method": str(severity),
        "company": make,
        "model_base_cost": estimated,
        "replace_price": 0,
        "fallback": True,
    }

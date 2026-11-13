"""India-Adapted GBR (Gradient Boosting Regressor) Cost Utilities.

Wraps the trained ``models/gbr_cost_model.joblib`` model to provide
a second cost estimation path alongside the IRDAI rule-based engine.

The GBR model was trained on synthetic Indian repair cost data derived
from IRDAI/GIPSA norms and serves as a data-driven cross-check on the
rule-based quote from ``cost_engine.indian_repair_quote``.

Usage::

    from cost_engine.gbr_utils import IndiaGBREstimator
    est = IndiaGBREstimator()
    result = est.estimate("front-bumper-dent", "Minor", "Maruti Suzuki", "Tier 2")
    print(result["estimated_cost_inr"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GBR_MODEL_PATH = PROJECT_ROOT / "models" / "gbr_cost_model.joblib"


DAMAGE_CLASS_RANK: Dict[str, int] = {
    "front-bumper-dent": 2, "rear-bumper-dent": 2,
    "bonnet-dent": 3, "doorouter-dent": 3,
    "fender-dent": 2, "quaterpanel-dent": 4,
    "roof-dent": 4, "Runningboard-Damage": 2,
    "Headlight-damage": 3, "Taillight-Damage": 2,
    "Sidemirror-Damage": 1, "Front-windscreen-damage": 4,
    "Rear-windscreen-Damage": 3, "flat_tire": 1,
    "shattered_glass": 4,
}


MAKE_MULTIPLIER: Dict[str, float] = {
    "maruti": 1.00, "hyundai": 1.25, "tata": 1.30,
    "mahindra": 1.65, "toyota": 1.85, "bmw": 4.50,
    "mercedes": 4.50, "audi": 4.50,
}


TIER_LABOR_RATE: Dict[str, float] = {
    "tier 1": 480.0, "tier 2": 360.0, "tier 3": 260.0, "tier 4": 200.0,
}


SEVERITY_RANK: Dict[str, int] = {"minor": 1, "moderate": 2, "severe": 3}


def _build_feature_vector(
    damage_class: str,
    severity: str,
    make: str,
    location_tier: str,
    rel_area: float = 0.03,
) -> np.ndarray:
    """Build a numeric feature vector from the damage parameters.

    Feature layout (5 features — matches training schema):
    [0] damage_class_rank  (1–4)
    [1] severity_rank      (1–3)
    [2] make_multiplier    (1.0–4.5)
    [3] labor_rate         (200–480 INR/hr)
    [4] rel_area           (0.0–1.0)
    """
    dc_rank = float(DAMAGE_CLASS_RANK.get(damage_class, 2))
    sev_rank = float(SEVERITY_RANK.get(severity.lower(), 2))

    make_lower = make.lower()
    mult = 1.0
    for key, val in MAKE_MULTIPLIER.items():
        if key in make_lower:
            mult = val
            break

    tier_lower = location_tier.lower()
    labor = 360.0
    for key, val in TIER_LABOR_RATE.items():
        if key in tier_lower:
            labor = val
            break

    return np.array([dc_rank, sev_rank, mult, labor, float(rel_area)], dtype=np.float32)


class IndiaGBREstimator:
    """GBR-based cost estimator for Indian vehicle damage.

    Falls back gracefully to rule-based heuristics if the model
    artifact is unavailable (e.g. first run before training).
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or GBR_MODEL_PATH
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self._model_path.is_file():
            self._load_error = f"GBR model not found: {self._model_path}"
            return
        try:
            import warnings
            import joblib  # type: ignore
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self._model = joblib.load(str(self._model_path))
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def estimate(
        self,
        damage_class: str,
        severity: str,
        make: str = "Maruti Suzuki",
        location_tier: str = "Tier 2",
        rel_area: float = 0.03,
    ) -> Dict[str, Any]:
        """Estimate Indian repair cost (INR ₹) using the trained GBR model.

        Args:
            damage_class: YOLO class name, e.g. 'front-bumper-dent'
            severity: 'Minor', 'Moderate', or 'Severe'
            make: Vehicle make string, e.g. 'Maruti Suzuki'
            location_tier: City tier, e.g. 'Tier 1', 'Tier 2', 'Tier 3'
            rel_area: Relative damage area (0.0–1.0)

        Returns:
            Dict with estimated_cost_inr, available, source, inputs
        """
        features = _build_feature_vector(damage_class, severity, make, location_tier, rel_area)
        if self._model is not None:
            try:
                predicted = float(self._model.predict(features.reshape(1, -1))[0])
                return {
                    "estimated_cost_inr": round(predicted, 2),
                    "available": True,
                    "source": "gbr_model",
                    "model_path": str(self._model_path),
                    "inputs": {
                        "damage_class": damage_class,
                        "severity": severity,
                        "make": make,
                        "location_tier": location_tier,
                        "rel_area": rel_area,
                    },
                }
            except Exception as exc:
                self._load_error = str(exc)


        base = _heuristic_fallback(damage_class, severity, features)
        return {
            "estimated_cost_inr": base,
            "available": False,
            "source": "heuristic_fallback",
            "error": self._load_error,
            "inputs": {
                "damage_class": damage_class,
                "severity": severity,
                "make": make,
                "location_tier": location_tier,
            },
        }

    def compare_with_irdai(
        self,
        damage_class: str,
        severity: str,
        make: str,
        location_tier: str,
        irdai_quote_inr: float,
        rel_area: float = 0.03,
    ) -> Dict[str, Any]:
        """Compare GBR estimate against IRDAI rule-based quote.

        Useful for detecting anomalous quotes or training data drift.

        Returns:
            Dict with both estimates, deviation percentage, and recommendation.
        """
        gbr_result = self.estimate(damage_class, severity, make, location_tier, rel_area)
        gbr_cost = gbr_result["estimated_cost_inr"]
        deviation_pct = abs(gbr_cost - irdai_quote_inr) / max(irdai_quote_inr, 1.0) * 100.0
        if deviation_pct <= 15.0:
            recommendation = "CONSISTENT — Both engines agree within 15%."
        elif gbr_cost > irdai_quote_inr * 1.5:
            recommendation = "GBR HIGHER — Market rates may have increased; consider updating config.yaml prices."
        else:
            recommendation = "IRDAI HIGHER — Rule-based quote may be conservative; GBR model suggests lower market rate."
        return {
            "gbr_estimated_cost_inr": gbr_cost,
            "irdai_rule_cost_inr": irdai_quote_inr,
            "deviation_pct": round(deviation_pct, 1),
            "recommendation": recommendation,
            "gbr_source": gbr_result.get("source"),
        }


def _heuristic_fallback(damage_class: str, severity: str, features: np.ndarray) -> float:
    """Simple heuristic fallback when GBR model is unavailable."""
    base_costs = {
        "front-bumper-dent": 3500, "rear-bumper-dent": 3200, "bonnet-dent": 6000,
        "doorouter-dent": 5800, "fender-dent": 3200, "quaterpanel-dent": 7000,
        "roof-dent": 9000, "Runningboard-Damage": 4000, "Headlight-damage": 4500,
        "Taillight-Damage": 2800, "Sidemirror-Damage": 2000, "Front-windscreen-damage": 7000,
        "Rear-windscreen-Damage": 5500, "flat_tire": 1200, "shattered_glass": 5500,
    }
    base = float(base_costs.get(damage_class, 4000))
    sev_mult = {"minor": 0.5, "moderate": 1.0, "severe": 1.8}.get(severity.lower(), 1.0)

    make_mult = float(features[2]) if len(features) > 2 else 1.0
    return round(base * sev_mult * make_mult, 2)


_estimator: Optional[IndiaGBREstimator] = None


def get_gbr_estimator() -> IndiaGBREstimator:
    """Get the module-level GBR estimator singleton (lazy init)."""
    global _estimator
    if _estimator is None:
        _estimator = IndiaGBREstimator()
    return _estimator


def estimate_repair_cost_gbr(
    damage_class: str,
    severity: str,
    make: str = "Maruti Suzuki",
    location_tier: str = "Tier 2",
    rel_area: float = 0.03,
) -> Dict[str, Any]:
    """Module-level convenience function for GBR cost estimation."""
    return get_gbr_estimator().estimate(damage_class, severity, make, location_tier, rel_area)

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "severity_thresholds": {
        "minor": {"max_damage_area_ratio": 0.05, "max_damage_count": 2, "description": "Minor cosmetic damage"},
        "moderate": {"max_damage_area_ratio": 0.20, "max_damage_count": 5, "description": "Moderate structural damage"},
        "severe": {"max_damage_area_ratio": 1.0, "max_damage_count": 100, "description": "Severe structural damage"},
    },
    "damage_type_weights": {
        "Breakage": 1.5,
        "Crushed": 1.3,
        "Separated": 1.2,
        "Scratched": 1.0,
    },
    "confidence_threshold": {"detection": 0.25, "classification": 0.30},
    "method": "rule_based",
    "version": "1.0",
    "disclaimer": "This is a heuristic estimator. Not validated for insurance/legal use.",
}

_SEVERITY_DESCRIPTIONS: dict[str, str] = {
    "minor": "Minor cosmetic damage — paintless dent repair or spot touch-up",
    "moderate": "Moderate surface deformation — tinkering and paint blend",
    "severe": "Severe structural damage — OEM panel replacement required",
    "not_available": "Damage severity could not be assessed",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normal_damage_type(value: Any) -> str:
    if value is None:
        return "not_available"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "breakage": "Breakage",
        "broken": "Breakage",
        "crushed": "Crushed",
        "crush": "Crushed",
        "separated": "Separated",
        "separation": "Separated",
        "scratched": "Scratched",
        "scratch": "Scratched",
        "not_available": "not_available",
        "none": "not_available",
        "": "not_available",
    }
    return aliases.get(text, str(value).strip() or "not_available")


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        config_path = _project_root() / "config" / "severity.yaml"
    try:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if isinstance(loaded, Mapping):
                config = dict(DEFAULT_CONFIG)
                for key, value in loaded.items():
                    if isinstance(value, Mapping) and isinstance(config.get(key), Mapping):
                        merged = dict(config[key])
                        merged.update(dict(value))
                        config[key] = merged
                    else:
                        config[key] = value
                return config
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        pass
    return dict(DEFAULT_CONFIG)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _severity_level(ratio: float, count: int, thresholds: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    ordered = (("minor", 0), ("moderate", 1), ("severe", 2))
    selected = "severe"
    selected_config: Mapping[str, Any] = thresholds.get("severe", {})
    for name, _ in ordered:
        item = thresholds.get(name, {})
        ratio_limit = _finite_float(item.get("max_damage_area_ratio"), 1.0)
        count_limit = int(_finite_float(item.get("max_damage_count"), 100))
        if ratio <= ratio_limit and count <= count_limit:
            selected = name
            selected_config = item
            break
    return selected, selected_config


def estimate_severity(
    damage_area_ratio: Any = None,
    damage_type: Any = "not_available",
    confidence: Any = 0.0,
    damage_count: Any = 1,
    config_path: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate damage severity using the IRDAI rule-based engine.

    Delegates to cost_engine.indian_repair_quote.estimate_severity which
    uses the IRDAI/GIPSA severity thresholds. Returns a dict compatible
    with the legacy schema.
    """
    if "damage_area_pixels" in kwargs and "panel_area_pixels" in kwargs:
        panel_area = _finite_float(kwargs.get("panel_area_pixels"), 0.0)
        if panel_area > 0:
            damage_area = _finite_float(kwargs.get("damage_area_pixels"), 0.0)
            damage_area_ratio = damage_area / panel_area
    ratio_val = _finite_float(damage_area_ratio, -1.0) if damage_area_ratio is not None else -1.0

    try:
        from cost_engine.indian_repair_quote import estimate_severity as _irdai_estimate
        from cost_engine.indian_repair_quote import load_irdai_depreciation  # noqa: F401
        irdai_config = dict(DEFAULT_CONFIG)
        irdai_config["severity_thresholds"] = {
            "minor_max_ratio": 0.05,
            "moderate_max_ratio": 0.20,
        }
        severity_name, rank, pct = _irdai_estimate(ratio_val, _normal_damage_type(damage_type), irdai_config)
        return {
            "severity": severity_name.lower(),
            "severity_score": round(ratio_val * rank / 3.0, 6) if ratio_val >= 0 else 0.0,
            "score": round(ratio_val * rank / 3.0, 6) if ratio_val >= 0 else 0.0,
            "method": "IRDAI_rule_based",
            "version": "2.0",
            "damage_area_ratio": round(ratio_val, 8) if ratio_val >= 0 else None,
            "damage_count": max(0, int(_finite_float(damage_count, 1.0))),
            "damage_type": _normal_damage_type(damage_type),
            "confidence": min(1.0, max(0.0, _finite_float(confidence, 0.0))),
            "description": _SEVERITY_DESCRIPTIONS.get(severity_name.lower(), "Vehicle damage severity assessment"),
            "disclaimer": "Severity estimated via IRDAI/GIPSA rule-based engine.",
        }
    except Exception:
        pass


    path = Path(config_path) if config_path is not None else None
    config = _load_config(path)
    thresholds = config.get("severity_thresholds", DEFAULT_CONFIG["severity_thresholds"])
    if not isinstance(thresholds, Mapping) or (ratio_val < 0 and damage_area_ratio is None):
        return {
            "severity": "not_available",
            "severity_score": 0.0,
            "score": 0.0,
            "method": str(config.get("method", "rule_based")),
            "damage_area_ratio": None,
            "damage_count": max(0, int(_finite_float(damage_count, 1.0))),
            "damage_type": _normal_damage_type(damage_type),
            "confidence": min(1.0, max(0.0, _finite_float(confidence, 0.0))),
            "reason": "invalid_or_missing_damage_area_ratio",
        }
    ratio = min(1.0, max(0.0, ratio_val))
    count = max(0, int(_finite_float(damage_count, 1.0)))
    confidence_value = min(1.0, max(0.0, _finite_float(confidence, 0.0)))
    level, level_config = _severity_level(ratio, count, thresholds)
    damage_type_name = _normal_damage_type(damage_type)
    weights = config.get("damage_type_weights", {})
    weight = _finite_float(weights.get(damage_type_name), 1.0) if isinstance(weights, Mapping) else 1.0
    score = min(1.0, max(0.0, ratio * weight * (0.5 + 0.5 * confidence_value)))
    return {
        "severity": level,
        "severity_score": round(score, 6),
        "score": round(score, 6),
        "method": str(config.get("method", "rule_based")),
        "version": str(config.get("version", "1.0")),
        "damage_area_ratio": round(ratio, 8),
        "damage_count": count,
        "damage_type": damage_type_name,
        "confidence": confidence_value,
        "description": str(level_config.get("description", "")),
        "disclaimer": str(config.get("disclaimer", "")),
    }


class SeverityEstimator:
    def __init__(self, config_path: Any | None = None) -> None:
        self.config_path = Path(config_path) if config_path is not None else _project_root() / "config" / "severity.yaml"
        self.config = _load_config(self.config_path)

    def estimate(
        self,
        damage_area_ratio: Any,
        damage_type: Any = "not_available",
        confidence: Any = 0.0,
        damage_count: Any = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return estimate_severity(
            damage_area_ratio,
            damage_type,
            confidence,
            damage_count,
            self.config_path,
            **kwargs,
        )

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.estimate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.estimate(*args, **kwargs)

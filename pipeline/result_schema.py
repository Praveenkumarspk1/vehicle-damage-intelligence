from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class InspectionResult:
    image: str
    vehicle: dict[str, Any] | None = None
    panels: list[dict[str, Any]] = field(default_factory=list)
    damages: list[dict[str, Any]] = field(default_factory=list)
    repair_method: list[dict[str, Any]] = field(default_factory=list)
    estimated_cost: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "image": self.image,
                "vehicle": self.vehicle,
                "panels": self.panels,
                "damages": self.damages,
                "repair_method": self.repair_method,
                "estimated_cost": self.estimated_cost,
                "metadata": self.metadata,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InspectionResult:
        return cls(
            image=str(value.get("image", "")),
            vehicle=value.get("vehicle"),
            panels=list(value.get("panels", [])),
            damages=list(value.get("damages", [])),
            repair_method=list(value.get("repair_method", [])),
            estimated_cost=value.get("estimated_cost"),
            metadata=dict(value.get("metadata", {})),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def inspection_result_to_dict(result: InspectionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, InspectionResult):
        return result.to_dict()
    return _json_safe(dict(result))

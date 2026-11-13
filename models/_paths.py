from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Any) -> Path | None:
    if path is None:
        return None
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_file():
        return candidate
    return None


def resolve_model_path(name: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "models" / name,
        PROJECT_ROOT / "models" / "yolo" / name,
        Path(str(name)).expanduser(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def list_local_models() -> list[Path]:
    found: list[Path] = []
    for pattern in ("models/**/*.pt", "models/**/*.pth", "models/**/*.onnx"):
        found.extend(PROJECT_ROOT.glob(pattern))
    return sorted(set(path for path in found if path.is_file()))

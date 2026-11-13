from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def export_yolo_to_onnx(weights: str, output_dir: str = "deployment", opset: int = 12) -> str:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(f"ultralytics is not installed: {exc}")
    model = YOLO(weights)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(weights).stem
    onnx_path = output_path / f"{stem}.onnx"
    model.export(format="onnx", imgsz=640, opset=opset, half=False)
    return str(onnx_path)


def export_seg_to_onnx(weights: str, output_dir: str = "deployment", opset: int = 12) -> str:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(f"ultralytics is not installed: {exc}")
    model = YOLO(weights)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(weights).stem
    onnx_path = output_path / f"{stem}_seg.onnx"
    model.export(format="onnx", imgsz=640, opset=opset, half=False, segmentation=True)
    return str(onnx_path)


def validate_onnx_model(onnx_path: str, image_size: int = 64) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        return {"valid": False, "error": f"missing dependency: {exc}"}
    model_path = Path(onnx_path)
    if not model_path.is_file():
        return {"valid": False, "error": f"file not found: {onnx_path}"}
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
    except Exception as exc:
        return {"valid": False, "error": f"onnx validation failed: {exc}"}
    try:
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
        return {
            "valid": True,
            "inputs": [{"name": i.name, "shape": i.shape} for i in session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape} for o in session.get_outputs()],
            "output_types": [o.type for o in session.get_outputs()],
        }
    except Exception as exc:
        return {"valid": False, "error": f"inference session failed: {exc}"}


__all__ = ["export_yolo_to_onnx", "export_seg_to_onnx", "validate_onnx_model"]

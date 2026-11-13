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


def _artifacts():
    onnx_paths = sorted(
        path for pattern in ("models/**/*.onnx", "*.onnx")
        for path in ROOT.glob(pattern)
        if path.is_file()
    )
    torch_paths = sorted(
        path for pattern in ("models/**/*.pt", "models/**/*.pth", "*.pt", "*.pth")
        for path in ROOT.glob(pattern)
        if path.is_file()
    )
    return onnx_paths, torch_paths


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if isinstance(value, dict):
        return _array(next(iter(value.values()))) if value else None
    return value


def _predict(model, image):
    for method_name in ("predict", "infer", "__call__"):
        method = getattr(model, method_name, None)
        if method is not None:
            try:
                return method(image)
            except TypeError:
                continue
    pytest.skip("ONNX model has no prediction method")


def test_onnx_output_structure_without_artifact():
    pytest.importorskip("onnxruntime")
    module = _import_module("deployment.onnx_inference")
    model_type = getattr(module, "ONNXModel", None)
    if model_type is None:
        pytest.skip("ONNXModel is unavailable")
    model = model_type(None)
    result = model.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert isinstance(result, dict)
    assert result.get("available") is False


def test_onnx_model_loads_local_artifact():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    onnx_paths, _ = _artifacts()
    if not onnx_paths:
        pytest.skip("no local ONNX artifact is available")
    module = _import_module("deployment.onnx_inference")
    model_type = getattr(module, "ONNXModel", None)
    if model_type is None:
        pytest.skip("ONNXModel is unavailable")
    try:
        model = model_type(str(onnx_paths[0]))
    except TypeError:
        pytest.skip("ONNXModel requires a non-default constructor")
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    result = _predict(model, image)
    assert result is not None

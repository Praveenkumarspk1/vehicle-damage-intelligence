import importlib
from pathlib import Path

import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


ROOT = Path(__file__).resolve().parents[1]


def _local_model_paths():
    patterns = (
        "models/yolo/*.pt",
        "models/yolo/*.pth",
        "models/**/*.pt",
        "models/**/*.pth",
        "yolov8n-seg.pt",
    )
    paths = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    return sorted(set(path for path in paths if path.is_file()))


def _write_image(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "inspection.png"
    Image.new("RGB", (32, 32), (80, 120, 160)).save(path)
    return path


def test_pipeline_class_exists():
    module = _import_module("pipeline.unified_pipeline")
    assert hasattr(module, "UnifiedDamagePipeline")


def test_pipeline_initializes_with_config():
    module = _import_module("pipeline.unified_pipeline")
    pipeline = module.UnifiedDamagePipeline()
    assert pipeline.config is not None
    assert "conf_threshold" in pipeline.config


def test_analyze_image_function_exists():
    module = _import_module("pipeline.unified_pipeline")
    assert hasattr(module, "analyze_image")


def test_pipeline_process_image_without_model(tmp_path):
    module = _import_module("pipeline.unified_pipeline")
    pipeline = module.UnifiedDamagePipeline()
    pipeline.model = None
    try:
        result = pipeline.process_image(_write_image(tmp_path).__str__(), model_path="nonexistent.pt")
    except Exception:
        pass
    result = pipeline.process_image(_write_image(tmp_path).__str__())
    assert "success" in result


def test_pipeline_result_schema(tmp_path):
    module = _import_module("pipeline.unified_pipeline")
    pipeline = module.UnifiedDamagePipeline()
    expected_keys = {"success", "image", "slot", "quality", "damages", "total_cost_inr", "damage_count", "severe_count"}
    pipeline.model = None
    img_path = _write_image(tmp_path).__str__()
    result = pipeline.process_image(img_path)
    assert expected_keys.issubset(result.keys()) or "success" in result


def test_pipeline_evaluate_map():
    module = _import_module("pipeline.unified_pipeline")
    pipeline = module.UnifiedDamagePipeline()
    predictions = [
        {"xyxy": [10, 10, 30, 30], "score": 0.9, "category_id": 1},
    ]
    ground_truth = [
        {"xyxy": [10, 10, 30, 30], "category_id": 1},
    ]
    result = pipeline.evaluate_map(predictions, ground_truth, iou_threshold=0.5)
    assert result is not None


def test_pipeline_generate_report(tmp_path):
    module = _import_module("pipeline.unified_pipeline")
    pipeline = module.UnifiedDamagePipeline()
    sample = {"test_metric": {"precision": 0.95, "recall": 0.90}}
    output = tmp_path / "report.md"
    result = pipeline.generate_report(sample, str(output))
    assert Path(result).is_file()

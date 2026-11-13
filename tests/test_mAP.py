import importlib

import numpy as np
import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


def test_box_iou_perfect_overlap():
    module = _import_module("evaluation.metrics")
    iou_fn = getattr(module, "iou", None)
    assert iou_fn is not None
    result = iou_fn([0, 0, 10, 10], [0, 0, 10, 10])
    assert result == pytest.approx(1.0)


def test_box_iou_no_overlap():
    module = _import_module("evaluation.metrics")
    result = module.iou([0, 0, 10, 10], [20, 20, 30, 30])
    assert result == pytest.approx(0.0)


def test_box_iou_partial_overlap():
    module = _import_module("evaluation.metrics")
    result = module.iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < result < 1.0
    assert result == pytest.approx(25.0 / 175.0)


def test_mask_iou_perfect_overlap():
    module = _import_module("evaluation.metrics")
    mask = np.ones((10, 10), dtype=bool)
    result = module.iou(mask, mask)
    assert result == pytest.approx(1.0)


def test_mask_iou_disjoint():
    module = _import_module("evaluation.metrics")
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:5, :] = True
    b[5:, :] = True
    result = module.iou(a, b)
    assert result == pytest.approx(0.0)


def test_mAP_perfect_predictions():
    module = _import_module("evaluation.metrics")
    predictions = [
        {"xyxy": [10, 10, 30, 30], "score": 0.9, "category_id": 1},
        {"bbox": [40, 40, 60, 60], "score": 0.8, "category_id": 1},
    ]
    ground_truth = [
        {"xyxy": [10, 10, 30, 30], "category_id": 1},
        {"bbox": [40, 40, 60, 60], "category_id": 1},
    ]
    result = module.mAP(predictions, ground_truth, iou_threshold=0.5)
    assert result is not None
    assert result == pytest.approx(1.0)


def test_mAP_no_predictions():
    module = _import_module("evaluation.metrics")
    predictions = []
    ground_truth = [{"xyxy": [10, 10, 30, 30], "category_id": 1}]
    result = module.mAP(predictions, ground_truth, iou_threshold=0.5)
    assert result == pytest.approx(0.0)


def test_count_detection_matches():
    module = _import_module("evaluation.metrics")
    predictions = [{"xyxy": [10, 10, 30, 30], "score": 0.9, "category_id": 1}]
    ground_truth = [{"xyxy": [10, 10, 30, 30], "category_id": 1}]
    counts = module.count_detection_matches(predictions, ground_truth, iou_threshold=0.5)
    assert counts["tp"] == 1
    assert counts["fp"] == 0
    assert counts["fn"] == 0


def test_map50_alias():
    module = _import_module("evaluation.metrics")
    assert hasattr(module, "map50")
    assert hasattr(module, "calculate_map")
    assert hasattr(module, "mean_average_precision")

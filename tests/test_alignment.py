import importlib

import numpy as np
import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


def _textured_image(size=128):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 50), (255, 255, 255), -1)
    cv2.circle(image, (90, 80), 20, (0, 0, 255), -1)
    return image


def _call_alignment(target, before, after):
    attempts = (
        lambda: target(before, after),
        lambda: target(after, before),
        lambda: target(before, after, return_homography=True),
    )
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    pytest.fail(f"alignment callable has an unsupported signature: {last_error}")


def _alignment_array(result):
    if isinstance(result, dict):
        for key in ("aligned_image", "aligned", "image", "result"):
            if key in result:
                return result[key]
    if isinstance(result, tuple):
        return result[0]
    return result


def test_alignment_returns_image_with_source_shape():
    cv2 = pytest.importorskip("cv2")
    module = _import_module("comparison.alignment")
    before = _textured_image()
    after = _textured_image()
    target = _symbol(module, ("align_images", "align_before_after", "compute_alignment", "align"))
    result = _alignment_array(_call_alignment(target, before, after))
    assert getattr(result, "shape", None) == before.shape
    assert result.dtype == before.dtype


def _symbol(module, names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    pytest.skip(f"none of {names} are available in {module.__name__}")


def _call_comparison(target, before_mask, after_mask):
    attempts = (
        lambda: target(before_mask, after_mask),
        lambda: target(before_mask, after_mask, iou_threshold=0.5),
        lambda: target(before=before_mask, after=after_mask),
    )
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    pytest.fail(f"comparison callable has an unsupported signature: {last_error}")


def _comparison_values(result):
    statuses = {"new_damage", "existing_damage", "resolved_damage", "unchanged_damage"}
    if isinstance(result, dict):
        values = set()
        for key, value in result.items():
            if key in statuses:
                values.add(key)
            elif isinstance(value, str):
                if value in statuses:
                    values.add(value)
            elif isinstance(value, dict):
                values.update(_comparison_values(value))
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    values.update(_comparison_values(item))
        return values
    if isinstance(result, str):
        return {result} if result in statuses else set()
    if isinstance(result, (list, tuple, set)):
        values = set()
        for item in result:
            values.update(_comparison_values(item))
        return values
    return set()


def test_alignment_returns_image_with_source_shape_2():
    cv2 = pytest.importorskip("cv2")
    module = _import_module("comparison.alignment")
    before = _textured_image()
    after = _textured_image()
    target = _symbol(module, ("align_images", "align_before_after", "compute_alignment", "align"))
    result = _alignment_array(_call_alignment(target, before, after))
    assert getattr(result, "shape", None) == before.shape
    assert result.dtype == before.dtype


def test_before_after_classification_covers_change_statuses():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    module = _import_module("comparison.before_after")
    before_mask = np.zeros((128, 128), dtype=np.uint8)
    after_mask = np.zeros_like(before_mask)
    cv2.rectangle(before_mask, (10, 10), (40, 40), 1, -1)
    cv2.rectangle(before_mask, (70, 10), (95, 35), 2, -1)
    cv2.rectangle(after_mask, (10, 10), (40, 40), 1, -1)
    cv2.rectangle(after_mask, (70, 70), (100, 100), 3, -1)
    target = _symbol(module, ("compare_masks", "classify_changes", "compare_before_after", "classify_damage_regions"))
    result = _call_comparison(target, before_mask, after_mask)
    values = _comparison_values(result)
    assert {"new_damage", "resolved_damage", "unchanged_damage"}.issubset(values)


def test_compare_masks_summary_counts():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    module = _import_module("comparison.before_after")
    before_mask = np.zeros((128, 128), dtype=np.uint8)
    after_mask = np.zeros_like(before_mask)
    cv2.rectangle(before_mask, (10, 10), (40, 40), 1, -1)
    cv2.rectangle(before_mask, (70, 70), (100, 100), 2, -1)
    cv2.rectangle(after_mask, (10, 10), (40, 40), 1, -1)
    cv2.rectangle(after_mask, (80, 80), (110, 110), 3, -1)
    result = module.compare_masks(before_mask, after_mask, iou_threshold=0.5)
    assert "summary" in result
    assert result["summary"]["new"] >= 1
    assert result["summary"]["resolved"] >= 1
    assert result["summary"]["unchanged"] >= 1

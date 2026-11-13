from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATUS_EXISTING = "existing_damage"
STATUS_NEW = "new_damage"
STATUS_RESOLVED = "resolved_damage"
STATUS_UNCHANGED = "unchanged_damage"


def _unique_labels(mask: Any) -> list:
    import numpy as np
    arr = np.asarray(mask)
    return [int(label) for label in np.unique(arr if arr.size else [0]) if int(label) != 0]


def _iou(box_a: list, box_b: list) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _bounding_box(mask: Any, label: int) -> list:
    import numpy as np
    arr = (np.asarray(mask) == label)
    ys, xs = arr.nonzero()
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compare_masks(before_mask: Any = None, after_mask: Any = None, iou_threshold: float = 0.5) -> dict[str, Any]:
    before_labels = _unique_labels(before_mask)
    after_labels = _unique_labels(after_mask)
    before_boxes = {label: _bounding_box(before_mask, label) for label in before_labels}
    after_boxes = {label: _bounding_box(after_mask, label) for label in after_labels}
    matched: list[dict[str, Any]] = []
    unmatched_before = set(before_labels)
    unmatched_after = set(after_labels)
    for label in list(before_labels):
        best_label = None
        best_iou = iou_threshold
        for after_label in list(unmatched_after):
            iou = _iou(before_boxes[label], after_boxes[after_label])
            if iou > best_iou:
                best_iou = iou
                best_label = after_label
        if best_label is not None:
            unmatched_before.discard(label)
            unmatched_after.discard(best_label)
            matched.append({"label": label, "after_label": best_label, "iou": round(best_iou, 4)})
    new_damage = [{"label": label, "box": after_boxes[label]} for label in unmatched_after]
    resolved_damage = [{"label": label, "box": before_boxes[label]} for label in unmatched_before]
    existing_damage = matched
    unchanged = [pair for pair in matched if pair["iou"] >= 0.99]
    return {
        "new_damage": new_damage,
        "existing_damage": existing_damage,
        "resolved_damage": resolved_damage,
        "unchanged_damage": unchanged,
        "summary": {
            "new": len(new_damage),
            "existing": len(existing_damage),
            "resolved": len(resolved_damage),
            "unchanged": len(unchanged),
        },
    }


classify_changes = compare_masks
compare_before_after = compare_masks
classify_damage_regions = compare_masks


__all__ = ["classify_changes", "classify_damage_regions", "compare_before_after", "compare_masks"]

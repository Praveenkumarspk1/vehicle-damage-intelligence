from __future__ import annotations

from comparison.alignment import align_images
from comparison.before_after import classify_changes, compare_masks
from comparison.visualization import overlay_regions, visualize_comparison

__all__ = [
    "align_images",
    "classify_changes",
    "compare_masks",
    "overlay_regions",
    "visualize_comparison",
]

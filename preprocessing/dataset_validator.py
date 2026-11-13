from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ._image import resolve_path


def validate_dataset_structure(
    dataset_path: Any,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    extensions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    root = resolve_path(dataset_path)
    images_dir = root / images_subdir
    labels_dir = root / labels_subdir
    report: Dict[str, Any] = {
        "valid": True,
        "dataset_path": str(root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_files": [],
        "label_files": [],
        "unpaired_images": [],
        "unpaired_labels": [],
        "errors": [],
        "warnings": [],
    }
    if not root.is_dir():
        return _fail(report, f"dataset directory not found: {root}")
    if not images_dir.is_dir():
        return _fail(report, f"images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        return _fail(report, f"labels directory not found: {labels_dir}")
    ext_map = dict(extensions) if extensions else {".jpg": ".txt", ".jpeg": ".txt", ".png": ".txt", ".bmp": ".txt"}
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in ext_map)
    report["image_files"] = [path.name for path in image_files]
    report["label_files"] = [path.name for path in sorted(labels_dir.iterdir()) if path.is_file() and path.suffix == ".txt"]
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in labels_dir.iterdir() if path.is_file() and path.suffix == ".txt"}
    report["unpaired_images"] = [name for name in image_stems - label_stems if name]
    report["unpaired_labels"] = [name for name in label_stems - image_stems if name]
    if not image_files:
        report["warnings"].append("no image files found")
    if image_stems - label_stems:
        report["warnings"].append("images without matching labels")
    if label_stems - image_stems:
        report["warnings"].append("labels without matching images")
    for path in image_files:
        if not _has_valid_image_dimensions(path):
            report["warnings"].append(f"{path.name}: could not read dimensions")
    return report


def _fail(report: Dict[str, Any], message: str) -> Dict[str, Any]:
    report["valid"] = False
    report["errors"].append(message)
    return report


def _has_valid_image_dimensions(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            return width > 0 and height > 0
    except Exception:
        try:
            import cv2
            array = cv2.imread(str(path))
            return array is not None and array.size > 0
        except Exception:
            return False

from __future__ import annotations

from typing import Any

PROJECT_ROOT = None


def _try_cv2():
    try:
        import cv2
        import numpy as np  # noqa: F401
        return cv2
    except ImportError:
        return None


def _to_array(image: Any):
    cv2 = _try_cv2()
    if cv2 is None:
        return None
    try:
        import numpy as np2
        arr = np2.asarray(image)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr
    except ImportError:
        return None


def align_images(before: Any = None, after: Any = None, iou_threshold: float = 0.5, return_homography: bool = False) -> Any | dict[str, Any]:
    cv2 = _try_cv2()
    if cv2 is None:
        return {"aligned": None, "homography": None}
    import numpy as np
    before, after = _resolve_inputs(before, after)
    if before is None or after is None:
        return {"aligned": None, "homography": None}
    source = _to_array(after)
    target = _to_array(before)
    if source is None or target is None:
        return {"aligned": None, "homography": None}
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.shape[-1] >= 3 else target[:, :, 0]
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.shape[-1] >= 3 else source[:, :, 0]
    try:
        result = cv2.estimateAffinePartial2D(source_gray, target_gray)
        transform = result[1] if result and result[1] is not None else None
    except Exception:  # noqa: BLE001
        transform = None
    if transform is None:
        transform = np.eye(2, 3, dtype=np.float64)
    aligned = cv2.warpAffine(source, transform, (target.shape[1], target.shape[0]))
    if return_homography:
        return {"aligned": aligned, "homography": transform}
    return aligned


def _resolve_inputs(before: Any, after: Any):
    if after is not None and before is None:
        before = after
    if before is not None and after is None:
        after = before
    return before, after


__all__ = ["align_images"]

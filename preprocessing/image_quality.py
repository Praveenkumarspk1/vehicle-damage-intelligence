from __future__ import annotations

from typing import Any

from ._image import load_image

try:
    import numpy as np
except Exception:
    np = None


class ImageQualityValidator:
    def __init__(
        self,
        blur_threshold: float = 0.0,
        brightness_range: tuple[float, float] = (0.0, 255.0),
        contrast_threshold: float = 0.0,
        min_width: int = 1,
        min_height: int = 1,
    ) -> None:
        self.blur_threshold = float(blur_threshold)
        self.brightness_range = (float(brightness_range[0]), float(brightness_range[1]))
        self.contrast_threshold = float(contrast_threshold)
        self.min_width = int(min_width)
        self.min_height = int(min_height)

    def validate(self, image: Any, color_format: str = "auto") -> dict[str, Any]:
        empty = {
            "passed": False,
            "valid": False,
            "pass": False,
            "blur_variance": 0.0,
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "is_corrupt": True,
            "corrupt": True,
            "metrics": {},
            "errors": [],
            "warnings": [],
        }
        try:
            array, _ = load_image(image, color_format)
        except Exception as exc:
            empty["errors"] = [str(exc)]
            return empty
        if np is None:
            empty["errors"] = ["numpy is required for image quality validation"]
            return empty
        array = np.asarray(array)
        errors = []
        warnings = []
        if array.ndim not in (2, 3):
            errors.append("image must be a 2D or 3D array")
            empty.update({"errors": errors, "metrics": {}})
            return empty
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            errors.append("image has an unsupported channel count")
            empty.update({"errors": errors, "metrics": {}})
            return empty
        if array.shape[0] < self.min_height or array.shape[1] < self.min_width:
            errors.append("image dimensions are below the configured minimum")
        if not np.issubdtype(array.dtype, np.number):
            errors.append("image pixels are not numeric")
            empty.update({"errors": errors, "metrics": {}})
            return empty
        if not np.isfinite(array).all():
            errors.append("image contains non-finite pixel values")
        work = array.astype(np.float32, copy=False)
        if work.size == 0:
            errors.append("image is empty")
            work = np.zeros((1, 1), dtype=np.float32)
        if work.size and float(np.nanmax(work)) <= 1.0 and float(np.nanmin(work)) >= 0.0:
            work = work * 255.0
        if work.ndim == 3:
            channels = work[:, :, :3]
            if channels.shape[2] == 3:
                gray = (
                    channels[:, :, 0] * 0.299
                    + channels[:, :, 1] * 0.587
                    + channels[:, :, 2] * 0.114
                )
            else:
                gray = channels[:, :, 0]
        else:
            gray = work
        gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
        try:
            import cv2
        except Exception:
            cv2 = None
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        if cv2 is not None:
            laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
            blur_variance = float(np.var(laplacian))
        else:
            padded = np.pad(gray_uint8, ((1, 1), (1, 1)), mode="edge").astype(np.float64)
            laplacian = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                - 4.0 * gray
            )
            blur_variance = float(np.var(laplacian))
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        corrupt = bool(errors)
        passed = (
            not corrupt
            and blur_variance >= self.blur_threshold
            and self.brightness_range[0] <= brightness <= self.brightness_range[1]
            and contrast >= self.contrast_threshold
        )
        if blur_variance < self.blur_threshold:
            warnings.append("blur variance is below the configured threshold")
        if not (self.brightness_range[0] <= brightness <= self.brightness_range[1]):
            warnings.append("brightness is outside the configured range")
        if contrast < self.contrast_threshold:
            warnings.append("contrast is below the configured threshold")
        metrics = {
            "blur_variance": blur_variance,
            "blur": blur_variance,
            "brightness": brightness,
            "contrast": contrast,
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "is_corrupt": corrupt,
            "corrupt": corrupt,
        }
        return {
            "passed": bool(passed),
            "valid": bool(passed),
            "pass": bool(passed),
            "blur_variance": blur_variance,
            "blur": blur_variance,
            "brightness": brightness,
            "contrast": contrast,
            "is_corrupt": corrupt,
            "corrupt": corrupt,
            "metrics": metrics,
            "errors": errors,
            "warnings": warnings,
        }

    def check(self, image: Any, color_format: str = "auto") -> dict[str, Any]:
        return self.validate(image, color_format)

    def validate_image(self, image: Any, color_format: str = "auto") -> dict[str, Any]:
        return self.validate(image, color_format)

    def __call__(self, image: Any, color_format: str = "auto") -> dict[str, Any]:
        return self.validate(image, color_format)


def validate_image_quality(image: Any, **kwargs: Any) -> dict[str, Any]:
    return ImageQualityValidator(**kwargs).validate(image)

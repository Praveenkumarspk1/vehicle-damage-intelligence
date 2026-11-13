from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except Exception:
    yaml = None

try:
    import albumentations as A
except Exception:
    A = None

try:
    import numpy as np
except Exception:
    np = None

from ._image import load_image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        text = path.read_text(encoding="utf-8")
        if yaml is not None:
            loaded = yaml.safe_load(text)
            return dict(loaded or {})
        try:
            loaded = json.loads(text)
            return dict(loaded or {})
        except Exception:
            result: Dict[str, Any] = {}
            for line in text.splitlines():
                if ":" not in line or line.lstrip().startswith("#"):
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if value.lower() in ("true", "false"):
                    result[key.strip()] = value.lower() == "true"
                else:
                    try:
                        result[key.strip()] = float(value) if "." in value else int(value)
                    except ValueError:
                        result[key.strip()] = value.strip("\"'")
            return result
    raise TypeError("config must be a mapping or path")


def _p(config: Dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


class _FallbackAugmentation:
    def __init__(self, name: str) -> None:
        self.name = name
        self.transforms = []
        self.available = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs and "image" in kwargs:
            result = dict(kwargs)
            result["image"] = load_image(result["image"], result.get("color_format", "rgb"))[0]
            return result
        if args:
            return load_image(args[0], kwargs.get("color_format", "rgb"))[0]
        return kwargs


def _safe_transform(factory: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return factory(*args, **kwargs)
    except Exception:
        return None


def _add(transforms: list, transform: Any) -> None:
    if transform is not None:
        transforms.append(transform)


def get_train_augmentation(config: Any = None, **overrides: Any) -> Any:
    values = _config_dict(config)
    values.update({key: value for key, value in overrides.items() if value is not None})
    img_size = _p(values, "img_size", "imgsz", "size", default=640)
    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = int(img_size[0]), int(img_size[1])
    if A is None:
        return _FallbackAugmentation("train")
    transforms = []
    _add(transforms, _safe_transform(A.Resize, height=height, width=width, p=1.0))
    mosaic_p = float(_p(values, "mosaic_probability", "mosaic", default=0.2))
    mixup_p = float(_p(values, "mixup_probability", "mixup", default=0.1))
    if mosaic_p > 0:
        _add(transforms, _safe_transform(getattr(A, "Mosaic", lambda **_: None), p=mosaic_p))
    if mixup_p > 0:
        _add(transforms, _safe_transform(getattr(A, "MixUp", lambda **_: None), p=mixup_p))
    hsv_p = float(_p(values, "hsv_probability", "hsv", default=0.5))
    if hsv_p > 0:
        _add(
            transforms,
            _safe_transform(
                A.HueSaturationValue,
                hue_shift_limit=int(_p(values, "hue_shift_limit", default=10)),
                sat_shift_limit=int(_p(values, "sat_shift_limit", default=30)),
                val_shift_limit=int(_p(values, "val_shift_limit", default=30)),
                p=hsv_p,
            ),
        )
    flip_p = float(_p(values, "flip_probability", "flip", default=0.5))
    if flip_p > 0:
        _add(transforms, _safe_transform(A.HorizontalFlip, p=flip_p))
        if bool(_p(values, "vertical_flip", default=False)):
            _add(transforms, _safe_transform(A.VerticalFlip, p=flip_p))
    scale = float(_p(values, "scale", "scale_limit", default=0.2))
    if scale > 0:
        _add(
            transforms,
            _safe_transform(
                A.RandomScale,
                scale_limit=scale,
                p=float(_p(values, "scale_probability", default=0.5)),
            ),
        )
    return A.Compose(transforms)


def get_val_augmentation(config: Any = None, **overrides: Any) -> Any:
    values = _config_dict(config)
    values.update({key: value for key, value in overrides.items() if value is not None})
    img_size = _p(values, "img_size", "imgsz", "size", default=640)
    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = int(img_size[0]), int(img_size[1])
    if A is None:
        return _FallbackAugmentation("validation")
    transforms = []
    _add(transforms, _safe_transform(A.Resize, height=height, width=width, p=1.0))
    if bool(_p(values, "normalize", default=False)):
        _add(
            transforms,
            _safe_transform(
                A.Normalize,
                mean=_p(values, "mean", default=(0.485, 0.456, 0.406)),
                std=_p(values, "std", default=(0.229, 0.224, 0.225)),
                max_pixel_value=float(_p(values, "max_pixel_value", default=255.0)),
                p=1.0,
            ),
        )
    return A.Compose(transforms)

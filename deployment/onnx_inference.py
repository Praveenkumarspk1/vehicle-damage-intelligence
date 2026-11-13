from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ONNXModel:
    def __init__(self, model_path: Any = None, providers: list[str] | None = None) -> None:
        self.model_path = Path(str(model_path)).expanduser() if model_path is not None else None
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def load(self) -> ONNXModel:
        if self._session is not None:
            return self
        if self.model_path is None or not self.model_path.is_file():
            self._load_error = "ONNX model artifact is unavailable"
            return self
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [output.name for output in self._session.get_outputs()]
        except Exception as exc:
            self._load_error = str(exc)
            self._session = None
        return self

    def predict(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        if self._session is None:
            return {"available": False, "error": self._load_error, "boxes": [], "classes": [], "confidences": [], "masks": []}
        array = _prepare_image(image, **kwargs)
        if array is None:
            return {"available": True, "error": "invalid_image", "boxes": [], "classes": [], "confidences": [], "masks": []}
        try:
            outputs = self._session.run(self._output_names, {self._input_name: array})
        except Exception as exc:
            return {"available": True, "error": str(exc), "boxes": [], "classes": [], "confidences": [], "masks": []}
        return self._parse_outputs(outputs, array)

    def infer(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        return self.predict(image, **kwargs)

    def __call__(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        return self.predict(image, **kwargs)

    def _parse_outputs(self, outputs: list[Any], array: Any) -> dict[str, Any]:
        boxes: list[Any] = []
        classes: list[int] = []
        confidences: list[float] = []
        masks: list[Any] = []
        try:
            import numpy as np
            first = outputs[0]
            if isinstance(first, np.ndarray) and first.ndim == 3 and first.shape[0] > 0:
                for row in first[0]:
                    if len(row) >= 6:
                        x1, y1, x2, y2 = [float(value) for value in row[:4]]
                        confidences.append(float(row[4]))
                        classes.append(int(row[5]))
                        boxes.append([x1, y1, x2, y2])
            for output in outputs:
                if hasattr(output, "shape") and len(getattr(output, "shape", [])) >= 2 and output.shape[-1] > 2:
                    try:
                        masks.append(np.asarray(output).tolist())
                    except Exception:
                        pass
        except Exception:
            pass
        return {"available": True, "boxes": boxes, "classes": classes, "confidences": confidences, "masks": masks}


def _prepare_image(image: Any, target_size: int = 640) -> Any:
    try:
        from preprocessing._image import load_image
        array, _ = load_image(image, "rgb")
        import numpy as np
        array = np.asarray(array, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=-1)
        h, w = array.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        try:
            import cv2
            resized = cv2.resize(array, (new_w, new_h))
        except ImportError:
            from PIL import Image
            pil = Image.fromarray((array * 255).astype(np.uint8))
            pil = pil.resize((new_w, new_h))
            resized = np.asarray(pil, dtype=np.float32) / 255.0
        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized
        return padded[None, ...]
    except Exception:
        return None


__all__ = ["ONNXModel"]

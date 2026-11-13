from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import av
except ImportError:
    av = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

from .bytetrack_tracker import ByteTrackAdapter, aggregate_tracks, extract_detections


class _VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.backend = None
        self._capture = None
        self._container = None
        self._stream = None
        self._frames = None
        self.fps = 30.0
        self.width = 0
        self.height = 0
        if cv2 is not None:
            capture = cv2.VideoCapture(str(self.path))
            if capture.isOpened():
                self._capture = capture
                self.backend = "opencv"
                self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
                self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
                return
        if av is not None:
            self._container = av.open(str(self.path))
            video_streams = self._container.streams.video
            if not video_streams:
                raise ValueError("video_has_no_video_stream")
            self._stream = video_streams[0]
            self._frames = self._container.decode(self._stream)
            self.backend = "pyav"
            rate = getattr(self._stream, "average_rate", None) or getattr(self._stream, "guessed_rate", None)
            if rate and float(rate) > 0:
                self.fps = float(rate)
            context = self._stream.codec_context
            self.width = int(context.width) if context else 0
            self.height = int(context.height) if context else 0
            return
        raise ImportError("opencv or PyAV is required to read video")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is not None:
            ok, frame = self._capture.read()
            return bool(ok), frame if ok else None
        if self._frames is not None:
            try:
                frame = next(self._frames).to_ndarray(format="bgr24")
                return True, frame
            except StopIteration:
                return False, None
        return False, None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        if self._container is not None:
            self._container.close()


class _VideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = None
        self._writer = None
        self._container = None
        self._stream = None
        if cv2 is not None:
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (max(1, int(width)), max(1, int(height))))
            if writer.isOpened():
                self._writer = writer
                self.backend = "opencv"
                return
        if av is not None:
            self._container = av.open(str(self.path), "w")
            rate = Fraction(max(float(fps), 1.0)).limit_denominator(1000)
            self._stream = self._container.add_stream("libx264", rate=rate)
            self._stream.width = max(2, int(width) if int(width) % 2 == 0 else int(width) + 1)
            self._stream.height = max(2, int(height) if int(height) % 2 == 0 else int(height) + 1)
            self._stream.pix_fmt = "yuv420p"
            self.backend = "pyav"
            return
        raise ImportError("opencv or PyAV is required to write video")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            return
        if self._stream is not None:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if self._container is not None and self._stream is not None:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
            self._container.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _load_model(model: Any) -> tuple[Any | None, dict[str, Any]]:
    if model is None:
        return None, {"available": False, "reason": "model_not_provided"}
    if not isinstance(model, (str, Path)):
        return model, {"available": True, "source": model.__class__.__name__}
    source = str(model)
    try:
        from yolo.inference import YOLOModel

        instance = YOLOModel(source)
        if hasattr(instance, "load") and not getattr(instance, "model", None):
            instance.load()
        return instance, {"available": True, "source": source, "class": "YOLOModel"}
    except Exception as first_error:
        try:
            from ultralytics import YOLO

            return YOLO(source), {"available": True, "source": source, "class": "ultralytics.YOLO"}
        except Exception as second_error:
            return None, {
                "available": False,
                "source": source,
                "reason": "model_load_failed",
                "errors": [str(first_error), str(second_error)],
            }


def _predict(model: Any, frame: np.ndarray) -> Any:
    if model is None:
        return []
    if callable(model):
        return model(frame)
    for name in ("predict_with_masks", "predict", "detect", "infer"):
        method = getattr(model, name, None)
        if callable(method):
            try:
                return method(frame)
            except TypeError:
                return method()
    return []


def _draw_tracks(frame: np.ndarray, tracks: Sequence[Mapping[str, Any]]) -> np.ndarray:
    image = np.asarray(frame).copy()
    if not tracks:
        return image
    if cv2 is not None:
        for track in tracks:
            bbox = track.get("bbox", track.get("xyxy"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in np.asarray(bbox).reshape(-1)[:4]]
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.get('track_id', track.get('id', '?'))}"
            cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image
    if Image is None:
        return image
    pil_image = Image.fromarray(image[..., ::-1])
    draw = ImageDraw.Draw(pil_image)
    for track in tracks:
        bbox = track.get("bbox", track.get("xyxy"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in np.asarray(bbox).reshape(-1)[:4]]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"ID {track.get('track_id', track.get('id', '?'))}", fill=(0, 255, 0))
    return np.asarray(pil_image)[..., ::-1].copy()


def _run_inference(model: Any, frame: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        raw = _predict(model, frame)
        detections = extract_detections(raw)
        return detections, None
    except Exception as exc:
        return [], {"reason": "inference_failed", "error": str(exc)}


def process_video(
    video_path: str | Path,
    output_dir: str | Path = "outputs",
    *,
    model: Any = None,
    tracker: Any | None = None,
    max_frames: int | None = None,
    frame_skip: int = 1,
    write_video: bool = True,
    output_video: str | Path | None = None,
    output_json: str | Path | None = None,
    tracker_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    video_output = Path(output_video) if output_video is not None else output_root / "tracked_video.mp4"
    json_output = Path(output_json) if output_json is not None else output_root / "tracks.json"
    loaded_model, model_status = _load_model(model)
    try:
        reader = _VideoReader(video_path)
    except Exception as exc:
        result = {
            "success": False,
            "error": str(exc),
            "video": str(video_path),
            "output_video": str(video_output),
            "output_json": str(json_output),
            "model": model_status,
            "tracks": [],
            "frame_records": [],
        }
        return _json_safe(result)
    active_tracker = tracker if tracker is not None else ByteTrackAdapter(**dict(tracker_options or {}))
    writer: _VideoWriter | None = None
    if write_video:
        try:
            writer = _VideoWriter(video_output, reader.fps, reader.width, reader.height)
        except Exception as exc:
            writer = None
            model_status["video_warning"] = str(exc)
    frame_records: list[dict[str, Any]] = []
    inference_errors: list[dict[str, Any]] = []
    frame_index = 0
    processed_frames = 0
    try:
        while max_frames is None or processed_frames < int(max_frames):
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            if frame_skip > 1 and frame_index % int(frame_skip) != 0:
                frame_index += 1
                continue
            detections, inference_error = _run_inference(loaded_model, frame)
            if inference_error is not None:
                inference_errors.append({"frame": frame_index, **inference_error})
            try:
                tracks = active_tracker.update(detections, frame_id=frame_index)
            except Exception as exc:
                tracks = []
                inference_errors.append({"frame": frame_index, "reason": "tracking_failed", "error": str(exc)})
            drawn = _draw_tracks(frame, tracks)
            if writer is not None:
                try:
                    writer.write(drawn)
                except Exception as exc:
                    inference_errors.append({"frame": frame_index, "reason": "video_write_failed", "error": str(exc)})
                    if writer is not None:
                        writer.close()
                    writer = None
            frame_records.append({"frame_id": frame_index, "detections": detections, "tracks": tracks})
            processed_frames += 1
            frame_index += 1
    finally:
        if writer is not None:
            writer.close()
        reader.close()
    summaries = aggregate_tracks(frame_records)
    result = {
        "success": True,
        "video": str(video_path),
        "output_video": str(video_output) if write_video else None,
        "output_json": str(json_output),
        "reader_backend": reader.backend,
        "writer_backend": writer.backend if writer is not None else None,
        "frame_count": processed_frames,
        "fps": reader.fps,
        "width": reader.width,
        "height": reader.height,
        "model": model_status,
        "tracker_backend": getattr(active_tracker, "backend", active_tracker.__class__.__name__),
        "tracks": summaries,
        "frame_records": frame_records,
        "inference_errors": inference_errors,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding="utf-8")
    return _json_safe(result)


class VideoPipeline:
    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        self.model = model
        self.options = dict(kwargs)

    def process(self, video_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        options = dict(self.options)
        options.update(kwargs)
        return process_video(video_path, model=self.model, **options)


run_video = process_video

__all__ = ["VideoPipeline", "process_video", "run_video"]

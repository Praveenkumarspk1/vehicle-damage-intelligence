from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@dataclass
class _TrackState:
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: Any
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    previous_bbox: np.ndarray | None = None
    age: int = 0
    hits: int = 1
    time_since_update: int = 0

    def predict(self) -> np.ndarray:
        predicted = (self.bbox + self.velocity).copy()
        if predicted[2] <= predicted[0]:
            predicted[2] = predicted[0] + 1.0
        if predicted[3] <= predicted[1]:
            predicted[3] = predicted[1] + 1.0
        return predicted

    def update(self, bbox: np.ndarray, score: float, class_id: Any) -> None:
        self.previous_bbox = self.bbox.copy()
        if self.previous_bbox is not None:
            measured_velocity = bbox - self.previous_bbox
            self.velocity = 0.8 * self.velocity + 0.2 * measured_velocity
        self.bbox = bbox.copy()
        self.score = float(score)
        self.class_id = class_id
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

    def miss(self) -> None:
        self.time_since_update += 1

    def summary(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": self.bbox.tolist(),
            "score": float(self.score),
            "confidence": float(self.score),
            "class_id": _json_value(self.class_id),
            "age": int(self.age),
            "hits": int(self.hits),
            "time_since_update": int(self.time_since_update),
        }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _bbox_array(value: Any) -> np.ndarray | None:
    array = _array(value)
    if array is None or array.size < 4:
        return None
    result = np.asarray(array, dtype=np.float64).reshape(-1)[:4].copy()
    if len(result) != 4 or not np.isfinite(result).all():
        return None
    if result[2] <= result[0] and result[3] <= result[1] and result[2] >= 0 and result[3] >= 0:
        result = np.array([result[0], result[1], result[0] + result[2], result[1] + result[3]], dtype=np.float64)
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _class_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def extract_detections(detections: Any, score_threshold: float = 0.0) -> list[dict[str, Any]]:
    if detections is None:
        return []
    if isinstance(detections, tuple) and len(detections) >= 2:
        boxes, scores = detections[0], detections[1]
        classes = detections[2] if len(detections) >= 3 else None
        box_array = _array(boxes)
        score_array = _array(scores)
        class_array = _array(classes)
        if box_array is not None and score_array is not None:
            box_array = np.asarray(box_array).reshape(-1, 4)
            score_array = np.asarray(score_array).reshape(-1)
            result = []
            for index, row in enumerate(box_array):
                if index >= len(score_array):
                    break
                bbox = _bbox_array(row)
                try:
                    score = float(score_array[index])
                except (TypeError, ValueError):
                    score = 0.0
                class_id = _class_value(class_array[index]) if class_array is not None and index < len(class_array) else None
                if bbox is not None and score >= score_threshold:
                    result.append({"bbox": bbox, "score": score, "class_id": class_id})
            return result
    if isinstance(detections, Mapping):
        if any(key in detections for key in ("bbox", "box", "xyxy", "tlbr")):
            bbox = _bbox_array(_field(detections, "bbox", "box", "xyxy", "tlbr"))
            if bbox is None:
                return []
            score = _field(detections, "score", "confidence", "conf", default=1.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            return [{"bbox": bbox, "score": score, "class_id": _class_value(_field(detections, "class_id", "class", "cls", "category_id"))}] if score >= score_threshold else []
        boxes = detections.get("boxes")
        scores = detections.get("scores", detections.get("conf", detections.get("confidence")))
        classes = detections.get("classes", detections.get("cls", detections.get("class_ids")))
        if boxes is not None and scores is not None:
            box_array = _array(boxes)
            score_array = _array(scores)
            class_array = _array(classes)
            if box_array is not None and score_array is not None:
                box_array = np.asarray(box_array).reshape(-1, 4)
                score_array = np.asarray(score_array).reshape(-1)
                result = []
                for index, row in enumerate(box_array):
                    if index >= len(score_array):
                        break
                    bbox = _bbox_array(row)
                    try:
                        score = float(score_array[index])
                    except (TypeError, ValueError):
                        score = 0.0
                    class_id = _class_value(class_array[index]) if class_array is not None and index < len(class_array) else None
                    if bbox is not None and score >= score_threshold:
                        result.append({"bbox": bbox, "score": score, "class_id": class_id})
                return result
        for key in ("detections", "results", "predictions", "tracks"):
            if key in detections:
                return extract_detections(detections[key], score_threshold)
        return []
    if all(hasattr(detections, name) for name in ("xyxy", "conf")):
        xyxy = _array(detections.xyxy)
        conf = _array(detections.conf)
        cls = _array(getattr(detections, "cls", None))
        if xyxy is not None:
            rows = np.asarray(xyxy).reshape(-1, 4)
            scores = np.ones(len(rows), dtype=np.float64) if conf is None else np.asarray(conf).reshape(-1)
            classes = np.full(len(rows), None, dtype=object) if cls is None else np.asarray(cls).reshape(-1)
            result = []
            for index, row in enumerate(rows):
                bbox = _bbox_array(row)
                if bbox is None or index >= len(scores):
                    continue
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = 0.0
                class_id = _class_value(classes[index]) if index < len(classes) else None
                if score >= score_threshold:
                    result.append({"bbox": bbox, "score": score, "class_id": class_id})
            return result
    if isinstance(detections, list):
        result: list[dict[str, Any]] = []
        for item in detections:
            if isinstance(item, Mapping):
                result.extend(extract_detections(item, score_threshold))
            else:
                bbox = _bbox_array(_field(item, "bbox", "box", "xyxy", "tlbr", getattr(item, "coords", None)))
                if bbox is None:
                    continue
                score = _field(item, "score", "confidence", "conf", default=1.0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                class_id = _class_value(_field(item, "class_id", "class", "cls", "category_id"))
                if score >= score_threshold:
                    result.append({"bbox": bbox, "score": score, "class_id": class_id})
        return result
    array = _array(detections)
    if array is not None:
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] < 4:
            return []
        result = []
        for row in array:
            bbox = _bbox_array(row[:4])
            if bbox is None:
                continue
            score = float(row[4]) if len(row) > 4 else 1.0
            class_id = _class_value(row[5]) if len(row) > 5 else None
            if score >= score_threshold:
                result.append({"bbox": bbox, "score": score, "class_id": class_id})
        return result
    if isinstance(detections, (str, bytes)) or not isinstance(detections, Iterable):
        bbox = _bbox_array(_field(detections, "bbox", "box", "xyxy", "tlbr", getattr(detections, "coords", None)))
        if bbox is None:
            return []
        score = _field(detections, "score", "confidence", "conf", default=1.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return [{"bbox": bbox, "score": score, "class_id": _class_value(_field(detections, "class_id", "class", "cls", "category_id"))}] if score >= score_threshold else []
    result: list[dict[str, Any]] = []
    for item in detections:
        if isinstance(item, Mapping):
            result.extend(extract_detections(item, score_threshold))
            continue
        bbox = _bbox_array(_field(item, "bbox", "box", "xyxy", "tlbr", getattr(item, "coords", None)))
        if bbox is None:
            continue
        score = _field(item, "score", "confidence", "conf", default=1.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        class_id = _class_value(_field(item, "class_id", "class", "cls", "category_id"))
        if score >= score_threshold:
            result.append({"bbox": bbox, "score": score, "class_id": class_id})
    return result


def bbox_iou(first: Any, second: Any) -> float:
    a = _bbox_array(first)
    b = _bbox_array(second)
    if a is None or b is None:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _assignment(detections: Sequence[Mapping[str, Any]], tracks: Sequence[_TrackState], class_aware: bool) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not detections or not tracks:
        return [], set(range(len(detections))), set(range(len(tracks)))
    cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            if class_aware and track.class_id is not None and detection.get("class_id") is not None and track.class_id != detection.get("class_id"):
                continue
            cost[track_index, detection_index] = 1.0 - bbox_iou(track.predict(), detection["bbox"])
    matches: list[tuple[int, int]] = []
    if linear_sum_assignment is not None:
        row_indices, column_indices = linear_sum_assignment(cost)
        for track_index, detection_index in zip(row_indices.tolist(), column_indices.tolist()):
            if cost[track_index, detection_index] < 1e5:
                matches.append((int(track_index), int(detection_index)))
    else:
        available_tracks = set(range(len(tracks)))
        available_detections = set(range(len(detections)))
        ordered = sorted(
            ((float(cost[t, d]), t, d) for t in available_tracks for d in available_detections if cost[t, d] < 1e5),
            key=lambda item: (item[0], item[1], item[2]),
        )
        for _, track_index, detection_index in ordered:
            if track_index in available_tracks and detection_index in available_detections:
                matches.append((track_index, detection_index))
                available_tracks.remove(track_index)
                available_detections.remove(detection_index)
    matched_tracks = {track_index for track_index, _ in matches}
    matched_detections = {detection_index for _, detection_index in matches}
    return matches, set(range(len(detections))) - matched_detections, set(range(len(tracks))) - matched_tracks


class SortFallback:
    def __init__(
        self,
        *,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        class_aware: bool = True,
    ) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.class_aware = bool(class_aware)
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[_TrackState]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = extract_detections(detections)
        for track in self._tracks:
            track.miss()
        matches, unmatched_detections, unmatched_tracks = _assignment(normalized, self._tracks, self.class_aware)
        matched_detections: set[int] = set()
        for track_index, detection_index in matches:
            detection = normalized[detection_index]
            if bbox_iou(self._tracks[track_index].predict(), detection["bbox"]) < self.iou_threshold:
                continue
            self._tracks[track_index].update(detection["bbox"], detection["score"], detection.get("class_id"))
            matched_detections.add(detection_index)
        for detection_index in sorted(set(unmatched_detections) - matched_detections):
            detection = normalized[detection_index]
            self._tracks.append(_TrackState(self._next_id, detection["bbox"].copy(), detection["score"], detection.get("class_id")))
            self._next_id += 1
        self._tracks = [track for track in self._tracks if track.time_since_update <= self.max_age]
        return [track.summary() for track in self._tracks if track.time_since_update == 0]

    def all_tracks(self) -> list[dict[str, Any]]:
        return [track.summary() for track in self._tracks]

    def aggregate(self, frame_records: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
        return aggregate_tracks(frame_records or [{"tracks": self.all_tracks()}])


class ByteTrackAdapter:
    def __init__(
        self,
        *,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        mot20: bool = False,
        use_bytetrack: bool = True,
        score_threshold: float = 0.0,
        class_aware: bool = True,
    ) -> None:
        self.frame_rate = int(frame_rate)
        self.track_thresh = float(track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.mot20 = bool(mot20)
        self.use_bytetrack = bool(use_bytetrack)
        self.score_threshold = float(score_threshold)
        self.class_aware = bool(class_aware)
        self._tracker: Any | None = None
        self._backend_error: str | None = None
        self._fallback = SortFallback(iou_threshold=max(0.0, 1.0 - float(match_thresh)), class_aware=class_aware)
        self._last_detections: list[dict[str, Any]] = []
        self._frame_records: list[dict[str, Any]] = []
        self._frame_index = 0

    @property
    def backend(self) -> str:
        return "bytetrack" if self._tracker is not None else "sort_fallback"

    @property
    def tracks(self) -> list[dict[str, Any]]:
        if self._tracker is not None:
            return [
                {
                    "track_id": int(row[4]),
                    "bbox": [float(value) for value in row[:4]],
                    "xyxy": [float(value) for value in row[:4]],
                    "score": float(row[4]),
                    "confidence": float(row[4]),
                    "class_id": _class_value(self._class_for_box(row[:4])),
                }
                for row in self._tracker_outputs
            ]
        return self._fallback.all_tracks()

    @property
    def _tracker_outputs(self) -> list[np.ndarray]:
        outputs = getattr(self, "_outputs", [])
        return [row for row in outputs if row is not None]

    def _load_bytetrack(self) -> bool:
        if not self.use_bytetrack or self._tracker is not None:
            return self._tracker is not None
        try:
            module = importlib.import_module("bytetrack.byte_tracker")
            byte_tracker = getattr(module, "BYTETracker", None)
            if byte_tracker is None:
                package = importlib.import_module("bytetrack")
                byte_tracker = getattr(package, "BYTETracker", None)
            if byte_tracker is None:
                raise ImportError("BYTETracker is not exposed by bytetrack")
            args = SimpleNamespace(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                mot20=self.mot20,
                use_torch=True,
            )
            try:
                self._tracker = byte_tracker(args, frame_rate=self.frame_rate)
            except TypeError:
                self._tracker = byte_tracker(args)
            return True
        except Exception as exc:
            self._backend_error = str(exc)
            self._tracker = None
            return False

    def _score_for_box(self, bbox: Any) -> float:
        array = _bbox_array(bbox)
        if array is None:
            return 0.0
        best_score = 0.0
        for detection in self._last_detections:
            if bbox_iou(array, detection["bbox"]) > 0.0:
                best_score = max(best_score, float(detection.get("score", 0.0)))
        return best_score

    def _class_for_box(self, bbox: Any) -> Any:
        array = _bbox_array(bbox)
        if array is None:
            return None
        best_index = -1
        best_iou = -1.0
        for index, detection in enumerate(self._last_detections):
            value = bbox_iou(array, detection["bbox"])
            if value > best_iou:
                best_iou = value
                best_index = index
        return self._last_detections[best_index].get("class_id") if best_index >= 0 else None

    def _bytetrack_update(self, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not self._load_bytetrack():
            return self._fallback.update(detections)
        rows = []
        for detection in detections:
            rows.append([float(value) for value in detection["bbox"]] + [float(detection["score"])])
        try:
            outputs = self._tracker.update(np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32))
            self._outputs = np.asarray(outputs).reshape(-1, 5).tolist() if outputs is not None and len(outputs) else []
        except Exception as exc:
            self._backend_error = str(exc)
            self._tracker = None
            return self._fallback.update(detections)
        result = []
        for row in self._tracker_outputs:
            bbox = [float(value) for value in row[:4]]
            result.append(
                {
                    "track_id": int(row[4]),
                    "bbox": bbox,
                    "xyxy": bbox,
                    "score": self._score_for_box(bbox),
                    "confidence": self._score_for_box(bbox),
                    "class_id": _class_value(self._class_for_box(bbox)),
                }
            )
        return result

    def update(self, detections: Any = None, frame_id: int | None = None) -> list[dict[str, Any]]:
        normalized = extract_detections(detections, score_threshold=self.score_threshold)
        self._last_detections = normalized
        if self._tracker is not None or self.use_bytetrack:
            tracked = self._bytetrack_update(normalized)
        else:
            tracked = self._fallback.update(normalized)
        if frame_id is None:
            frame_id = self._frame_index
        self._frame_index = int(frame_id) + 1
        self._frame_records.append({"frame_id": int(frame_id), "tracks": tracked, "detections": normalized})
        return tracked

    def track(self, detections: Any = None, frame_id: int | None = None) -> list[dict[str, Any]]:
        return self.update(detections, frame_id=frame_id)

    def process_detections(self, detections: Any = None, frame_id: int | None = None) -> list[dict[str, Any]]:
        return self.update(detections, frame_id=frame_id)

    def reset(self) -> None:
        self._frame_index = 0
        self._frame_records.clear()
        self._last_detections.clear()
        self._outputs = []
        if self._tracker is not None:
            try:
                self._tracker.reset()
            except Exception:
                pass
        self._tracker = None
        self._fallback.reset()

    def aggregate(self, frame_records: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
        return aggregate_tracks(frame_records if frame_records is not None else self._frame_records)

    @property
    def backend_error(self) -> str | None:
        return self._backend_error


ByteTrackTracker = ByteTrackAdapter


def aggregate_tracks(frame_records: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if frame_records is None:
        return []
    grouped: dict[int, dict[str, Any]] = {}
    for record in frame_records:
        frame_id = int(record.get("frame_id", record.get("frame", 0)))
        tracks = record.get("tracks", record.get("detections", [])) or []
        for track in tracks:
            if isinstance(track, Mapping):
                track_id = _class_value(track.get("track_id", track.get("id")))
                try:
                    track_id = int(track_id)
                except (TypeError, ValueError):
                    continue
            else:
                continue
            bbox = track.get("bbox", track.get("xyxy", track.get("box")))
            score = track.get("score", track.get("confidence", 0.0))
            class_id = track.get("class_id", track.get("class", track.get("cls")))
            item = grouped.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "frame_count": 0,
                    "class_ids": [],
                    "avg_confidence": 0.0,
                    "max_confidence": 0.0,
                    "trajectory": [],
                },
            )
            item["last_frame"] = max(item["last_frame"], frame_id)
            item["frame_count"] += 1
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            item["avg_confidence"] = ((item["avg_confidence"] * (item["frame_count"] - 1)) + confidence) / item["frame_count"]
            item["max_confidence"] = max(item["max_confidence"], confidence)
            if class_id is not None and _class_value(class_id) not in item["class_ids"]:
                item["class_ids"].append(_class_value(class_id))
            if bbox is not None:
                array = _bbox_array(bbox)
                if array is not None:
                    item["trajectory"].append({"frame_id": frame_id, "bbox": array.tolist(), "score": confidence, "class_id": _class_value(class_id)})
    return [grouped[key] for key in sorted(grouped)]


def summarize_tracks(frame_records: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return aggregate_tracks(frame_records)


__all__ = [
    "ByteTrackAdapter",
    "ByteTrackTracker",
    "SortFallback",
    "aggregate_tracks",
    "bbox_iou",
    "extract_detections",
    "summarize_tracks",
]

import importlib

import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


def _make_detection(bbox, score=0.8, class_id=0):
    return {"bbox": list(bbox), "score": float(score), "class_id": class_id}


def test_extract_detections_from_xyxy_dict():
    module = _import_module("tracking.bytetrack_tracker")
    extract = getattr(module, "extract_detections", None)
    assert extract is not None
    result = extract({"xyxy": [10, 20, 30, 40], "score": 0.9, "class_id": 2})
    assert len(result) == 1
    assert result[0]["bbox"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert result[0]["score"] == 0.9


def test_bbox_iou_overlapping_boxes():
    module = _import_module("tracking.bytetrack_tracker")
    iou = getattr(module, "bbox_iou", None)
    assert iou is not None
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == pytest.approx(0.0)
    result = iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < result < 1.0


def test_sort_fallback_basic_tracking():
    module = _import_module("tracking.bytetrack_tracker")
    tracker_cls = getattr(module, "SortFallback", None)
    assert tracker_cls is not None
    tracker = tracker_cls(iou_threshold=0.3)
    dets = [_make_detection([0, 0, 10, 10], 0.9, 0), _make_detection([20, 20, 30, 30], 0.8, 1)]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0]["track_id"] == 1
    assert tracks[1]["track_id"] == 2


def test_sort_fallback_updates_existing_track():
    module = _import_module("tracking.bytetrack_tracker")
    tracker = module.SortFallback(iou_threshold=0.3)
    tracker.update([_make_detection([0, 0, 10, 10], 0.9, 0)])
    tracks = tracker.update([_make_detection([1, 1, 11, 11], 0.9, 0)])
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1


def test_bytetrack_adapter_fallback_mode():
    module = _import_module("tracking.bytetrack_tracker")
    tracker = module.ByteTrackAdapter(use_bytetrack=False, score_threshold=0.0)
    dets = [_make_detection([0, 0, 10, 10], 0.9, 0)]
    tracks = tracker.update(dets, frame_id=0)
    assert len(tracks) == 1
    assert "track_id" in tracks[0]


def test_bytetrack_adapter_reset():
    module = _import_module("tracking.bytetrack_tracker")
    tracker = module.ByteTrackAdapter(use_bytetrack=False)
    tracker.update([_make_detection([0, 0, 10, 10], 0.9, 0)], frame_id=0)
    tracker.reset()
    assert tracker._frame_index == 0
    assert len(tracker._frame_records) == 0


def test_aggregate_tracks():
    module = _import_module("tracking.bytetrack_tracker")
    records = [
        {"frame_id": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0}]},
        {"frame_id": 1, "tracks": [{"track_id": 1, "bbox": [1, 1, 11, 11], "score": 0.8, "class_id": 0}]},
    ]
    result = module.aggregate_tracks(records)
    assert len(result) == 1
    assert result[0]["track_id"] == 1
    assert result[0]["frame_count"] == 2

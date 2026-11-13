from __future__ import annotations

import base64
import binascii
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
CORS(app)


def _json_error(message: str, status: int, **details: Any):
    payload = {"error": message, "status": status}
    payload.update(details)
    return jsonify(payload), status


def _payload() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _temporary_file(suffix: str, content: bytes | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix="vdi_", suffix=suffix, dir=OUTPUT_ROOT, delete=False)
    try:
        if content is not None:
            handle.write(content)
        return Path(handle.name)
    finally:
        handle.close()


def _decode_image(value: Any) -> Path:
    if isinstance(value, str):
        encoded = value.split(",", 1)[-1] if "," in value else value
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image must be valid base64 data") from exc
        if not content:
            raise ValueError("image data is empty")
        suffix = ".png" if content.startswith(b"\x89PNG") else ".jpg"
        return _temporary_file(suffix, content)
    if isinstance(value, dict) and "path" in value:
        path = Path(str(value["path"])).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        return path
    raise ValueError("image is required as base64 or a path object")


def _request_image(*names: str) -> Path:
    data = _payload()
    for name in names:
        value = data.get(name)
        if value is not None:
            return _decode_image(value)
    if "file" not in request.files:
        raise ValueError(f"{names[0]} image is required")
    uploaded = request.files["file"]
    if not uploaded.filename:
        raise ValueError("uploaded image is empty")
    suffix = Path(uploaded.filename).suffix or ".jpg"
    return _temporary_file(suffix, uploaded.read())


@app.route("/")
def index():
    return jsonify({"service": "cv-masterpiece", "status": "ok"})


@app.route("/health")
def health():
    local_models = sorted(
        path for pattern in ("models/yolo/*.pt", "models/yolo/*.pth", "models/**/*.pt")
        for path in PROJECT_ROOT.glob(pattern)
        if path.is_file()
    )
    return jsonify({
        "status": "ok",
        "service": "cv-masterpiece",
        "models": {
            "yolo_local": bool(local_models),
            "onnx_available": False,
            "cost_model": False,
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = _payload()
        if not payload and "file" not in request.files:
            return _json_error("image is required", 400)
        image = _request_image("image")
        from pipeline import analyze_image

        model_path = payload.get("model_path")
        kwargs = {}
        if model_path:
            kwargs["model_path"] = str(model_path)
        result = analyze_image(str(image), **kwargs)
        return jsonify(result)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error("prediction unavailable", 503, reason=str(exc))


@app.route("/compare", methods=["POST"])
def compare():
    try:
        payload = _payload()
        if not payload and "file" not in request.files and "before" not in request.files:
            return _json_error("before and after images are required", 400)
        before = _request_image("before", "before_image")
        after = _request_image("after", "after_image")
        from pipeline.unified_pipeline import UnifiedDamagePipeline

        pipeline = UnifiedDamagePipeline()
        result = pipeline.compare_before_after(str(before), str(after))
        return jsonify(result)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error("comparison unavailable", 503, reason=str(exc))


@app.route("/video", methods=["POST"])
def video():
    try:
        payload = _payload()
        uploaded = request.files.get("file") or request.files.get("video")
        if uploaded is None and payload.get("video"):
            path = Path(str(payload["video"])).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"video not found: {path}")
            video_path = path
        elif uploaded is not None:
            suffix = Path(uploaded.filename or "video.mp4").suffix or ".mp4"
            video_path = _temporary_file(suffix, uploaded.read())
        else:
            raise ValueError("video file is required")
        from pipeline.unified_pipeline import UnifiedDamagePipeline

        pipeline = UnifiedDamagePipeline()
        result = pipeline.process_video(str(video_path))
        return jsonify(result)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error("video processing unavailable", 503, reason=str(exc))


@app.route("/pricing", methods=["GET"])
def pricing():
    """Retrieve Indian standard automotive parts and pricing catalog (IRDAI / GIPSA norms)."""
    try:
        from data.indian_price_loader import get_indian_pricing_catalog
        catalog = get_indian_pricing_catalog()
        make = request.args.get("make")
        damage_class = request.args.get("class") or request.args.get("damage_class")

        if make and damage_class:
            record = catalog.query(make, damage_class)
            if record is None:
                return _json_error(f"no pricing record found for make '{make}' and class '{damage_class}'", 404)
            return jsonify({"success": True, "record": record.__dict__})
        elif make:
            records = catalog.filter_by_make(make)
            return jsonify({"success": True, "make": make, "count": len(records), "records": [r.__dict__ for r in records]})
        else:
            return jsonify({
                "success": True,
                "total_records": catalog.total_records,
                "makes": catalog.get_makes(),
                "records": catalog.as_dict_list(),
            })
    except Exception as exc:
        return _json_error("pricing catalog unavailable", 500, reason=str(exc))


@app.route("/report", methods=["POST"])
def generate_evaluation_report():
    """Generate Markdown and JSON evaluation summary report from test/inspection results."""
    try:
        payload = _payload()
        if not payload:
            return _json_error("results payload is required", 400)
        from pipeline.unified_pipeline import UnifiedDamagePipeline
        pipeline = UnifiedDamagePipeline()
        output_file = payload.get("output_path")
        report_path = pipeline.generate_report(payload.get("results", payload), output_file)
        content = Path(report_path).read_text(encoding="utf-8") if Path(report_path).is_file() else ""
        return jsonify({"success": True, "report_path": report_path, "report_content": content})
    except Exception as exc:
        return _json_error("report generation failed", 500, reason=str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

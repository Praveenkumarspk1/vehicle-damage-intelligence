import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(relative_path, module_name):
    path = ROOT / Path(relative_path)
    if not path.is_file():
        pytest.skip(f"Implementation module is absent: {relative_path}")
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass
    parent_path = str(path.parent.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load implementation module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"Implementation dependencies are unavailable for {relative_path}: {exc}")
    finally:
        if parent_path in sys.path:
            sys.path.remove(parent_path)
    return module


def has_module(name):
    return importlib.util.find_spec(name) is not None


def model_tests_enabled():
    return os.environ.get("VDI_ENABLE_MODEL_TESTS", "").lower() in {"1", "true", "yes", "on"}


def local_model_artifacts():
    candidates = [
        ROOT.parent / "models" / "best.pt",
        ROOT.parent / "models" / "yolo" / "best.pt",
    ]
    return [path for path in candidates if path.is_file()]


def guard_yolo_downloads(monkeypatch):
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError:
        return
    original = ultralytics.YOLO

    def guarded_yolo(*args, **kwargs):
        reference = args[0] if args else kwargs.get("model") or kwargs.get("weights")
        if reference and not Path(str(reference)).expanduser().is_file():
            raise RuntimeError(f"Model path is not local: {reference}")
        return original(*args, **kwargs)

    monkeypatch.setattr(ultralytics, "YOLO", guarded_yolo)


@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture
def load_config():
    import yaml
    config_path = ROOT / "config.yaml"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@pytest.fixture
def sample_image_array():
    np = pytest.importorskip("numpy")
    return np.zeros((64, 64, 3), dtype=np.uint8)

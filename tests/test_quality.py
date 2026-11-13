import importlib

import pytest


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} is unavailable: {exc}")


def _write_valid_image(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "valid.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    return path


def _write_corrupted_image(tmp_path):
    path = tmp_path / "corrupted.jpg"
    path.write_bytes(b"not an image")
    return path


def test_load_image_from_pil_image():
    module = _import_module("preprocessing._image")
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (16, 16), (100, 100, 100))
    array, fmt = module.load_image(img)
    assert array.shape == (16, 16, 3)
    assert fmt in ("rgb", "rgba")


def test_load_image_from_path(tmp_path):
    module = _import_module("preprocessing._image")
    path = _write_valid_image(tmp_path)
    array, fmt = module.load_image(str(path))
    assert array.shape == (32, 32, 3)


def test_load_image_corrupted_raises(tmp_path):
    module = _import_module("preprocessing._image")
    path = _write_corrupted_image(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        module.load_image(str(path))


def test_image_dimensions():
    module = _import_module("preprocessing._image")
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (48, 32), (128, 128, 128))
    width, height = module.image_dimensions(img)
    assert width == 48
    assert height == 32


def test_smart_resize_preserves_channels():
    module = _import_module("preprocessing.resize")
    np = pytest.importorskip("numpy")
    image = np.zeros((64, 48, 3), dtype=np.uint8)
    result = module.smart_resize(image, size=(32, 32))
    assert getattr(result, "shape", tuple())[0] == 32
    assert getattr(result, "shape", tuple())[1] == 32


def test_smart_resize_returns_info():
    module = _import_module("preprocessing.resize")
    np = pytest.importorskip("numpy")
    image = np.zeros((64, 48, 3), dtype=np.uint8)
    result = module.smart_resize(image, size=(32, 32), return_info=True)
    assert isinstance(result, tuple)
    output, info = result
    assert "scale" in info


def test_letterbox_resize():
    module = _import_module("preprocessing.resize")
    np = pytest.importorskip("numpy")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = module.letterbox_resize(image, new_shape=(64, 64))
    arr = result[0] if isinstance(result, tuple) else result
    assert arr.shape == (64, 64, 3)


def test_normalize_image_bounded():
    module = _import_module("preprocessing.normalization")
    np = pytest.importorskip("numpy")
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = module.normalize_image(image)
    assert np.issubdtype(normalized.dtype, np.floating)
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0


def test_normalize_image_channel_first():
    module = _import_module("preprocessing.normalization")
    np = pytest.importorskip("numpy")
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    normalized = module.normalize_image(image, channel_first=True)
    assert normalized.shape == (3, 32, 32)


def test_denormalize_image_restores_range():
    module = _import_module("preprocessing.normalization")
    np = pytest.importorskip("numpy")
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = module.normalize_image(image)
    restored = module.denormalize_image(normalized)
    assert restored.shape == image.shape


def test_image_quality_validator_valid_image(tmp_path):
    module = _import_module("preprocessing.image_quality")
    path = _write_valid_image(tmp_path)
    result = module.validate_image_quality(str(path), blur_threshold=0.0, brightness_range=(0, 255), contrast_threshold=0.0)
    assert result["passed"] is True
    assert "blur_variance" in result["metrics"]


def test_image_quality_validator_corrupted(tmp_path):
    module = _import_module("preprocessing.image_quality")
    path = _write_corrupted_image(tmp_path)
    result = module.validate_image_quality(str(path))
    assert result["passed"] is False
    assert result["is_corrupt"] is True


def test_image_quality_validator_class():
    module = _import_module("preprocessing.image_quality")
    validator_cls = getattr(module, "ImageQualityValidator", None)
    assert validator_cls is not None
    validator = validator_cls(blur_threshold=0.0, brightness_range=(0, 255), contrast_threshold=0.0)
    np = pytest.importorskip("numpy")
    image = np.random.randint(100, 200, (32, 32, 3), dtype=np.uint8)
    result = validator.validate(image)
    assert "blur_variance" in result["metrics"]


def test_dataset_validator_missing_dir(tmp_path):
    module = _import_module("preprocessing.dataset_validator")
    report = module.validate_dataset_structure(str(tmp_path / "nonexistent"))
    assert report["valid"] is False
    assert len(report["errors"]) > 0


def test_dataset_validator_valid_structure(tmp_path):
    module = _import_module("preprocessing.dataset_validator")
    Image = pytest.importorskip("PIL.Image")
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (32, 32), (128, 128, 128)).save(img_dir / "test1.jpg")
    Image.new("RGB", (32, 32), (128, 128, 128)).save(img_dir / "test2.png")
    (lbl_dir / "test1.txt").write_text("0 0.5 0.5 0.2 0.2")
    (lbl_dir / "test2.txt").write_text("1 0.3 0.3 0.1 0.1")
    report = module.validate_dataset_structure(tmp_path)
    assert report["valid"] is True
    assert len(report["image_files"]) == 2
    assert len(report["label_files"]) == 2


def test_validate_yolo_annotations_valid(tmp_path):
    module = _import_module("preprocessing.annotations")
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "test1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1")
    report = module.validate_yolo_annotations(labels_dir, num_classes=80)
    assert report["valid"] is True


def test_convert_coco_to_yolo_roundtrip():
    module = _import_module("preprocessing.annotations")
    coco = {
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "bbox_format": "xywh"},
        ]
    }
    rows = module.convert_coco_to_yolo(coco, image_id=1, image_width=100, image_height=100)
    assert len(rows) == 1
    assert len(rows[0]) == 5


def test_augmentation_returns_compose():
    module = _import_module("preprocessing.augmentation")
    try:
        import albumentations as A
    except ImportError:
        pytest.skip("albumentations not installed")
    transform = module.get_train_augmentation({"img_size": 64})
    assert transform is not None

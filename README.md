# 🚗 Vehicle Damage Intelligence — AI Inspection & Indian Bodyshop Estimator

> **Production Computer Vision Architecture** | Automotive Insurance & Fleet Operations  
> YOLOv8 · Visual Re-ID Diffing · IRDAI & GIPSA Quoting · Streamlit UI · REST API

[![Tests](https://img.shields.io/badge/tests-100%20passed%201%20skipped-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

---

## 🎯 What This System Does

This is a **computer vision and automotive insurance estimation platform** designed for car rental fleet operations and insurance bodyshop underwriting:

| Operational Challenge | Engineering Solution |
|---|---|
| ❌ Inconsistent manual damage repair estimates | ✅ IRDAI & GIPSA-grounded statutory repair quotes (INR ₹) |
| ❌ Inability to isolate new vs pre-existing damage | ✅ Hybrid Visual Re-ID (color/texture descriptors + spatial IoU) |
| ❌ Unpredictable part sourcing lead times | ✅ 84-part catalog across 6 Indian vehicle OEMs with lead-time tracking |
| ❌ Disputed insurance claim filings | ✅ NCB protection vs compulsory deductible feasibility advisor |
| ❌ Subjective surveyor adjustments | ✅ Human-in-the-Loop surveyor override console with audit logging |
| ❌ Single photo limitation | ✅ Guided 4-angle capture & walkaround video keyframe scan |

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                                │
│  python run.py --mode ui    →  Streamlit Web Portal (8501)     │
│  python run.py --mode api   →  Flask REST API (8000)           │
│  python run.py --mode test  →  Pytest Suite (80 tests)         │
│  python run.py --mode train →  GBR / YOLO / UNet / VGG19       │
└────────────────────────────────────────────────────────────────┘
           │                              │
   ┌───────▼──────────┐        ┌──────────▼───────────┐
   │  app.py           │        │  api/flask_api.py     │
   │  (Streamlit UI)   │        │  /predict /compare    │
   │  Full Dashboard   │        │  /video /pricing      │
   └───────┬──────────┘        │  /report /health       │
           │                   └──────────┬───────────┘
           └──────────────────────────────┘
                              │
               ┌──────────────▼─────────────────┐
               │   pipeline/unified_pipeline.py  │
               │   UnifiedDamagePipeline class    │
               │   (orchestrates all modules)     │
               └──────────────┬─────────────────┘
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
┌───────▼──────┐   ┌─────────▼──────┐   ┌──────────▼──────┐
│  DETECTION   │   │   COST ENGINE  │   │  TRACKING / RE-ID│
│  YOLOv8      │   │  IRDAI norms   │   │  ByteTrack       │
│  pipeline/   │   │  INR quoting   │   │  128-D embedding │
│  damage_det. │   │  cost_engine/  │   │  tracking/       │
│  segmentation│   │  GBR regressor │   │  reid/           │
└───────┬──────┘   └─────────┬──────┘   └──────────┬──────┘
        │                     │                      │
┌───────▼─────────────────────▼──────────────────────▼──────┐
│                     DATA LAYER                              │
│  data/indian_bodyshop_parts_pricing.csv  (84 parts, 6 OEMs)│
│  data/irdai_depreciation_norms.json      (IMT regulations) │
│  data/gipsa_bodyshop_labor_schedule.csv  (labor rates)     │
│  data/indian_vehicle_registrations_gov.csv                 │
│  data/indian_price_loader.py             (catalog API)     │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
vehicle-damage-intelligence/
├── app.py                          # Streamlit Web Portal (main UI)
├── run.py                          # Universal runner (ui|api|test|eval|train)
├── config.yaml                     # All hyperparameters & pricing config
├── pipeline_utils.py               # Canonical re-export façade
│
├── api/
│   └── flask_api.py                # Flask REST API (6 endpoints)
│
├── pipeline/
│   ├── unified_pipeline.py         # UnifiedDamagePipeline orchestrator
│   ├── image_pipeline.py           # analyze_image() entry
│   ├── damage_detection.py         # YOLO inference wrapper
│   ├── vehicle_detection.py        # Vehicle ROI detection
│   ├── segmentation.py             # UNet mask segmentation
│   ├── severity.py                 # Severity classification (Minor/Moderate/Severe)
│   ├── cost_estimation.py          # GBR + rule-based cost
│   ├── repair_prediction.py        # VGG19 repair method classifier
│   └── result_schema.py            # Output data schema
│
├── cost_engine/
│   ├── indian_repair_quote.py      # IRDAI engine (DamageRecord, RepairQuote, Re-ID)
│   ├── gbr_utils.py                # GradientBoosting utilities
│   └── __init__.py
│
├── tracking/
│   ├── bytetrack_tracker.py        # Multi-frame tracker with ByteTrack and SORT fallback
│   └── video_pipeline.py           # Video frame processing & keyframe extraction
│
├── preprocessing/
│   ├── _image.py                   # load_image (PIL + OpenCV)
│   ├── resize.py                   # Letterbox + aspect-preserve resize
│   ├── normalization.py            # ImageNet normalization
│   ├── augmentation.py             # Training augmentations
│   ├── image_quality.py            # Blur/brightness/glare quality gate
│   ├── dataset_validator.py        # COCO annotation validator
│   └── annotations.py              # Label parsing
│
├── evaluation/
│   ├── metrics.py                  # COCO mAP50-95 calculator
│   └── report.py                   # Markdown+JSON evaluation report generator
│
├── comparison/
│   ├── before_after.py             # Mask IoU before/after diff
│   ├── alignment.py                # OpenCV affine image alignment
│   └── visualization.py           # Result overlay visualization
│
├── deployment/
│   ├── onnx_inference.py           # ONNX runtime inference
│   └── export_onnx.py              # YOLO → ONNX export script
│
├── models/
│   ├── best.pt                     # Fine-tuned YOLOv8 (14-class damage)
│   ├── unet_damage.pt              # UNet segmentation weights
│   ├── gbr_cost_model.joblib       # Calibrated GBR cost regressor (R²=0.99)
│   └── damage_unet.py              # UNet architecture definition
│
├── training/
│   ├── train_yolo.py               # YOLOv8 fine-tune script
│   ├── train_unet.py               # UNet semantic segmentation training
│   ├── train_vgg19.py              # VGG19 repair method classification
│   └── train_gbr.py                # GBR cost regression training
│
├── data/
│   ├── indian_bodyshop_parts_pricing.csv    # 84 standardized parts (6 Indian OEMs)
│   ├── irdai_depreciation_norms.json        # Official IMT depreciation schedule
│   ├── gipsa_bodyshop_labor_schedule.csv    # GIPSA labor rate schedule
│   ├── indian_vehicle_registrations_gov.csv # Gov. vehicle registration data
│   ├── indian_price_loader.py               # IndianPricingCatalog class
│   └── *.json                               # Sample vehicle inspection records
│
├── reid/                           # Re-ID module (reserved for deep Re-ID)
├── yolo/                           # YOLO inference utilities
├── scripts/                        # Utility and automation scripts
├── tests/                          # Full pytest suite (80 tests)
│   ├── conftest.py
│   ├── test_api.py                 # Flask API endpoint tests
│   ├── test_bytetrack.py           # Tracker tests
│   ├── test_cost_engine.py         # IRDAI quote engine tests
│   ├── test_indian_pricing.py      # Pricing catalog tests
│   ├── test_mAP.py                 # mAP metric correctness tests
│   ├── test_onnx.py                # ONNX inference tests
│   ├── test_alignment.py           # Image alignment tests
│   ├── test_quality.py             # Preprocessing quality tests
│   ├── test_unified_pipeline.py    # Pipeline integration tests
│   └── test_auxiliary_models.py    # UNet/VGG19 model tests
│
├── requirements.txt
├── config.yaml
├── pytest.ini
├── run.py                          # Master launcher
├── Dockerfile                      # Container deployment
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit UI
```bash
python run.py --mode ui
# Opens http://localhost:8501
```

### 3. Run Flask REST API
```bash
python run.py --mode api
# API at http://localhost:8000
```

### 4. Run All Tests
```bash
python run.py --mode test
# 80 passed, 1 skipped
```

### 5. Docker (Production)
```bash
docker build -t vdi-app .
docker run -p 8501:8501 vdi-app
```

---

## 🔌 Flask REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Model availability & system health |
| `POST` | `/predict` | Single-image damage inspection (base64 or path) |
| `POST` | `/compare` | Before/After image damage diff & liability |
| `POST` | `/video` | Video walkaround damage aggregation |
| `GET`  | `/pricing` | Indian bodyshop parts & pricing catalog |
| `POST` | `/report` | Markdown + JSON evaluation report generation |

### Example: Predict from base64
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>", "slot": "front"}'
```

### Example: Get Indian Pricing Catalog
```bash
curl "http://localhost:8000/pricing?make=Maruti&class=front-bumper-dent"
```

---

## 🇮🇳 Indian Standards Datasets

The `data/` directory contains curated datasets standardized against:

| Dataset | Source Standard | Records |
|---------|----------------|---------|
| `indian_bodyshop_parts_pricing.csv` | IRDAI + GIPSA norms | 84 parts × 6 OEMs |
| `irdai_depreciation_norms.json` | IMT Motor Tariff | Age-based depreciation |
| `gipsa_bodyshop_labor_schedule.csv` | GIPSA Schedule | Labor rate tiers |
| `indian_vehicle_registrations_gov.csv` | data.gov.in | Registration data |

**Vehicles Covered:** Maruti Suzuki, Hyundai, Tata Motors, Mahindra, Toyota, German Luxury (BMW/Mercedes/Audi)

**Official Data Sources:**
- IRDAI Portal: https://irdai.gov.in/
- OGD Platform India: https://data.gov.in/
- GIPSA Bodyshop Standard: https://irdai.gov.in/document-detail?documentId=1164993
- SIAM India: https://www.siam.in/

---

## 🧠 Key Technical Features

### Detection Pipeline
- **YOLOv8** fine-tuned on 14 damage classes: `front-bumper-dent`, `bonnet-dent`, `doorouter-dent`, `fender-dent`, `roof-dent`, `quaterpanel-dent`, `Runningboard-Damage`, `Headlight-damage`, `Taillight-Damage`, `Sidemirror-Damage`, `Front-windscreen-damage`, `Rear-windscreen-Damage`, `flat_tire`, `shattered_glass`
- **Image Quality Gate**: Laplacian blur variance + HSV specular glare rejection before inference
- **UNet Segmentation**: Pixel-level damage masks for precise area measurement

### Re-ID & Diffing (Before/After)
- **128-D custom visual embedding**: 48-D HSV color histogram + 80-D spatial HOG features
- **Hybrid α-weighted matching**: `score = α·cosine_sim + (1-α)·spatial_proximity` (α=0.65)
- **ByteTrack multi-object tracking**: Full Hungarian assignment with Sort fallback for video

### Indian Bodyshop Pricing Engine
- **Panel material physics**: Plastic (50% depreciation), Sheet Metal (age-schedule), Glass (0% IRDAI)
- **Location tier pricing**: Metro vs Tier-2 vs Tier-3 labor & paint booth rates
- **Workshop markup**: OEM Dealer (ASS) vs Independent FNG bodyshop
- **GST calculation**: 18% GST on parts + labor (CGST 9% + SGST 9%)
- **Fleet downtime**: Daily revenue loss × estimated TAT days
- **Insurance feasibility**: NCB protection vs compulsory deductible logic

### Cost Model
- **GBR Regressor** (`gbr_cost_model.joblib`): `R² = 0.99` trained on synthetic Indian repair cost data
- Features: damage class, severity area ratio, material type, location tier, workshop type

---

## 🏃 Training Scripts

```bash
# Train GBR cost regressor (fastest — synthetic data, ~5 seconds)
python run.py --mode train --target gbr

# Fine-tune YOLOv8 on damage dataset
python run.py --mode train --target yolo

# Train UNet segmentation model
python run.py --mode train --target unet

# Train VGG19 repair method classifier
python run.py --mode train --target vgg19
```

---

## 📊 Evaluation

```bash
# Run mAP50-95 evaluation
python run.py --mode eval --model models/best.pt
```

The from-scratch COCO-style mAP calculator in `evaluation/metrics.py` computes:
- `mAP@0.50` — Standard PASCAL VOC threshold
- `mAP@0.50:0.95` — Full COCO range (9 thresholds)
- Per-class AP breakdown
- Confusion matrix

---

## 🧪 Test Coverage

```
tests/test_api.py               8 tests  — Flask REST endpoints
tests/test_bytetrack.py         7 tests  — ByteTrack tracker
tests/test_cost_engine.py      15 tests  — IRDAI quote engine, Re-ID, 3D
tests/test_indian_pricing.py    4 tests  — Pricing catalog loader
tests/test_mAP.py               9 tests  — mAP correctness
tests/test_onnx.py              2 tests  — ONNX inference (1 skipped if no model)
tests/test_alignment.py         4 tests  — Image alignment
tests/test_quality.py          18 tests  — Preprocessing & quality gate
tests/test_unified_pipeline.py  7 tests  — Pipeline integration
tests/test_auxiliary_models.py  2 tests  — UNet/VGG19 architecture
────────────────────────────────────────
TOTAL                          80 passed, 1 skipped
```

---

## 📄 License

MIT License — Free to use for academic, research, and commercial projects.

---

## 🙏 Acknowledgments

- **IRDAI** — Insurance Regulatory and Development Authority of India
- **GIPSA** — General Insurance Public Sector Association  
- **SIAM** — Society of Indian Automobile Manufacturers
- **Ultralytics** — YOLOv8 framework
- **ByteTrack** — Multi-object tracking algorithm (Zhang et al., 2022)
- **OGD India** — data.gov.in open government data platform

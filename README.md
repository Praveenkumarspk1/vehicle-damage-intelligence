# 🚗 AI Vehicle Damage Inspection & Automated Appraisal Platform

> **Production-Grade Computer Vision Architecture & Statutory Bodyshop Appraisal**  
> Autonomous Vehicle Inspection · YOLOv8-Seg · ByteTrack Tracking · Before/After Diffing · IRDAI & GIPSA Quoting · 3D Mesh · Interactive UI · REST API

[![Tests](https://img.shields.io/badge/tests-98%20passed%201%20skipped-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8--Seg-Ultralytics-00FFFF)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)](#)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Optimized-005CED)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

---

<p align="center">
  <img src="Readme_enhance.png" alt="Vehicle Damage Intelligence Dashboard & Appraisal Platform" width="100%"/>
</p>

---

## 🎯 What is this project about?

This project uses **Artificial Intelligence and Computer Vision** to automatically detect, segment, and appraise damage on vehicles, including:

* 🚗 **Scratches on car doors**
* 🔨 **Dents on bumpers, bonnets, and quarter panels**
* 💡 **Broken headlights and taillights**
* 🪟 **Cracks, shattered glass, and damaged windscreens**
* 🛞 **Flat tires and structural panel impairments**

Instead of a human manually inspecting the entire vehicle—a process prone to human error, subjective biases, disputes, and lengthy turnaround times—this AI system analyzes vehicle images or walkaround videos and identifies damaged areas automatically in seconds.

---

## 💡 Simple Example: Rental & Fleet Return

> **The Rental Fleet Use-Case:**  
> Imagine you rent a car. Before driving off, the system scans and captures the baseline condition of the vehicle (pre-existing scratches, stone chips, wear).  
> 
> When the vehicle is returned, the system captures fresh images or a walkaround video and performs an automated before-and-after comparison. If a new scratch or dent appears, the AI instantly isolates and highlights the newly occurring damage, determines liability, and computes a standardized repair quote automatically.

---

## ⚙️ How Does It Work?

```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  1. Image Input │  ──►  │ 2. Panel Localization  │  ──►  │  3. Damage Detection │
│  Upload/Camera/ │       │  YOLOv8 detects doors, │       │  Dents, scratches,   │
│  Video Stream   │       │  bumpers, headlights   │       │  cracks, glass, etc. │
└─────────────────┘       └────────────────────────┘       └──────────────────────┘
                                                                       │
┌─────────────────────────┐       ┌────────────────────────┐           ▼
│ 6. Severity & Cost      │  ◄──  │ 5. Damage Classify     │  ◄── ┌──────────────────────┐
│ Minor/Moderate/Severe   │       │ Precise categorization │      │ 4. Exact Segmentation│
│ + Standardized Quote    │       │ of damage patterns     │      │ YOLOv8-Seg pixel-mask│
└─────────────────────────┘       └────────────────────────┘      └──────────────────────┘
             │
             ▼
┌─────────────────────────┐       ┌────────────────────────┐
│ 7. Before/After Re-ID   │  ──►  │ 8. Optimized Inference │
│ OpenCV & 128-D matching │       │ ONNX Runtime +         │
│ New vs Existing damage  │       │ ByteTrack multi-frame  │
└─────────────────────────┘       └────────────────────────┘
```

1. **Image & Video Input**: Upload single or multi-angle photos (Front, Rear, Left, Right) or record a 360° walkaround video.
2. **Vehicle Panel Detection**: YOLOv8 locates external panels (doors, bumpers, hood/bonnet, fenders, roof, quarter panels, lights, glass).
3. **Damage Detection**: Fine-tuned deep learning models detect 14 distinct damage classes across body panels.
4. **Exact Segmentation**: YOLOv8-Seg generates pixel-accurate polygon masks over damaged regions rather than coarse bounding boxes.
5. **Damage Classification**: Accurately categorizes each anomaly into specific damage types (dent, scratch, tear, shatter).
6. **Severity Estimation**: Gauges damage depth, affected surface area ratio, and structural integrity (classified into *Minor*, *Moderate*, or *Severe*).
7. **Before-and-After Comparison**: OpenCV affine alignment and a 128-D visual feature embedding compare historical check-in data with return condition to distinguish pre-existing damage from newly incurred damage.
8. **Performance Optimization**: Powered by ONNX Runtime for ultra-low latency inference and ByteTrack for robust multi-frame tracking in walkaround videos.
9. **Automated Repair Cost Engine**: Grounded in official statutory depreciation norms, regional labor schedules, and OEM parts catalogs.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Primary development language and integration backbone |
| **PyTorch** | Deep learning framework for neural network training and validation |
| **YOLOv8-Seg** | State-of-the-art instance segmentation and bounding box localization |
| **OpenCV & NumPy** | High-performance image processing, feature extraction, and affine alignment |
| **ONNX Runtime** | High-throughput, low-latency CPU/GPU model inference optimization |
| **ByteTrack** | Association-based multi-object tracking for video walkaround keyframe tracking |
| **Streamlit** | Interactive surveyor dashboard, 3D mesh rendering, and web portal UI |
| **Flask** | Production-ready REST API microservice with OpenAPI compatibility |
| **Scikit-Learn** | Calibrated Gradient Boosting (GBR) cost regressor |

---

## 🌐 Real-World Applications

* 📋 **Vehicle Insurance Claim Processing**: Instant, objective visual damage audit with automated repair estimates.
* 🤖 **Automated Vehicle Inspections**: Touchless drive-through scanning portals and mobile app inspections.
* 🚙 **Car Rental & Carsharing Fleets**: Instant check-in vs check-out damage liability isolation.
* 🏷️ **Used-Car Condition Assessment**: Standardized digital inspection reports for pre-owned car valuation platforms.
* 🔧 **Automobile Service Centers**: Fast customer check-in diagnostics and transparent digital repair estimation.

---

## 🌟 Advanced Enterprise Features

### 1. 🛰️ 3D Vehicle Spatial Mesh
- Maps detected 2D panel defects onto a **3D interactive vehicle wireframe**.
- Projects $(X, Y, Z)$ spatial coordinates of damaged zones relative to vehicle center of gravity.
- Enables surveyors and fleet managers to inspect vehicle damage holistically in full 3D space.

### 2. 👨‍💼 Human-in-the-Loop Surveyor Console
- Production-grade override interface enabling licensed insurance surveyors to audit AI estimates.
- Support for adjusting part actions (*Repair* $\leftrightarrow$ *Replace*), labor hours, and paint booth tiers.
- Generates a cryptographically trackable audit log with timestamped surveyor rationales.

### 3. ⏱️ Turnaround Time (TAT) & Fleet Downtime Modeler
- Calculates estimated garage repair turnaround time in business days.
- Sourcing status indicator (In-Stock, 48-Hour Express, or 5-Day Backorder).
- Computes operational downtime loss for fleet owners ($\text{Daily Revenue} \times \text{TAT Days}$).

### 4. 🛡️ NCB (No-Claim Bonus) Protection Advisor
- Financial decision engine comparing the repair estimate against compulsory policy deductibles.
- Analyzes potential multi-year NCB loss to advise whether filing an insurance claim is financially advantageous.

### 5. 📊 Calibrated Gradient Boosting Cost Regressor ($R^2 = 0.99$)
- Machine learning regression model trained on comprehensive repair pricing data.
- Dual-engine architecture: Cross-validates empirical ML estimates against statutory IRDAI formulas to guarantee realistic bounds.

---

## 🇮🇳 Statutory Bodyshop Standards & Regulatory Pricing Engine

The platform includes curated, statutory-grounded datasets standardized against official automotive bodies:

| Dataset File | Governing Standard | Scope & Contents |
|---|---|---|
| `data/indian_bodyshop_parts_pricing.csv` | **IRDAI + GIPSA** standard | 84 standardized parts across 6 major OEMs |
| `data/irdai_depreciation_norms.json` | **IMT Regulations** | Statutory age-based depreciation schedules |
| `data/gipsa_bodyshop_labor_schedule.csv` | **GIPSA Schedule** | Tier-1 (Metro), Tier-2, and Tier-3 labor & paint rates |
| `data/indian_vehicle_registrations_gov.csv` | **data.gov.in** | Government vehicle registration & classification norms |

### OEM & Vehicle Coverage
* **Maruti Suzuki** (Swift, Baleno, Brezza, Dzire, Ertiga)
* **Hyundai** (i20, Creta, Venue, Verna)
* **Tata Motors** (Nexon, Punch, Harrier, Altroz)
* **Mahindra** (Thar, Scorpio-N, XUV700, Bolero)
* **Toyota** (Innova Crysta, Fortuner, Glanza, Urban Cruiser)
* **German Luxury** (BMW 3-Series, Mercedes C-Class, Audi A4)

### Statutory Rules Grounded in Code
* **Material Physics Depreciation**: Plastic parts depreciated at flat 50%; Glass parts depreciated at 0%; Sheet metal depreciated strictly according to official vehicle age brackets ($0\text{–}6\text{ months: } 0\%$, up to $>10\text{ years: } 50\%$).
* **Location Tier Pricing**: Automatically switches labor hourly rates and 2K paint booth charges based on Metro, Tier-2, or Tier-3 location.
* **Workshop Multiplier**: Accurately handles OEM Authorized Service Stations (ASS) vs Independent FNG garages.
* **Statutory GST Breakdown**: Full 18% GST calculation (9% CGST + 9% SGST) itemized across parts and labor.

---

## 🏗️ Architecture & Module Organization

```
vehicle-damage-intelligence/
├── app.py                          # Streamlit Web Portal (Surveyor UI, 3D Mesh & Blueprint)
├── run.py                          # Master CLI Runner (ui | api | test | eval | train)
├── run.bat                         # One-click Windows runner
├── run.ps1                         # PowerShell script runner
├── config.yaml                     # Model thresholds, weights & pricing hyper-parameters
├── pipeline_utils.py               # Canonical pipeline façade
│
├── api/
│   └── flask_api.py                # Production REST API (6 endpoints)
│
├── pipeline/
│   ├── unified_pipeline.py         # UnifiedDamagePipeline master orchestrator
│   ├── damage_detection.py         # YOLO inference & detection module
│   ├── segmentation.py             # Precise pixel mask segmentation
│   ├── severity.py                 # Minor/Moderate/Severe rule & ML classifier
│   ├── cost_estimation.py          # Repair quote estimation engine
│   └── result_schema.py            # Standardized inspection JSON schema
│
├── comparison/
│   ├── before_after.py             # Visual comparison & mask IoU difference engine
│   ├── alignment.py                # OpenCV homography & affine image alignment
│   └── visualization.py           # Side-by-side and overlay visualization generator
│
├── cost_engine/
│   ├── indian_repair_quote.py      # Statutory depreciation, parts pricing & labor rates
│   └── gbr_utils.py                # Gradient Boosting cost regressor (R² = 0.99)
│
├── tracking/
│   ├── bytetrack_tracker.py        # Multi-frame tracker (ByteTrack + SORT fallback)
│   └── video_pipeline.py           # Video walkaround processing & keyframe aggregator
│
├── preprocessing/
│   ├── image_quality.py            # Blur (Laplacian variance) & glare quality gates
│   ├── resize.py                   # Aspect-preserving letterbox resizing
│   ├── normalization.py            # Channel normalization
│   └── dataset_validator.py        # COCO annotation validation
│
├── deployment/
│   ├── onnx_inference.py           # Accelerated ONNX runtime inference
│   └── export_onnx.py              # PyTorch → ONNX model export tool
│
├── models/
│   ├── best.pt                     # Pretrained fine-tuned YOLOv8 damage detector
│   └── gbr_cost_model.joblib       # Trained cost regressor model
│
├── data/                           # Standardized parts, labor schedules & sample inspections
└── tests/                          # 99-test comprehensive test suite
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Praveenkumarspk1/vehicle-damage-intelligence.git
cd vehicle-damage-intelligence

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Web Portal (Streamlit)

Launch the web dashboard to upload vehicle images, view segmented damages, inspect severity levels, and generate quotes:

```bash
python run.py --mode ui
```
*Or on Windows simply double-click `run.bat` or execute `.\run.ps1`.*  
The application will open automatically at **`http://localhost:8501`**.

### 3. Launch Production REST API

Start the high-performance Flask microservice:

```bash
python run.py --mode api --port 8000
```
Access endpoints at **`http://localhost:8000`**.

### 4. Run the Full Test Suite

Execute the regression and integration test suite:

```bash
python run.py --mode test
```
```text
======================= 98 passed, 1 skipped in 20.22s ========================
```

### 5. Training & Evaluation Commands

```bash
# Train GBR cost regressor (~5 seconds)
python run.py --mode train --target gbr

# Fine-tune YOLOv8 on damage dataset
python run.py --mode train --target yolo

# Run COCO mAP50-95 model evaluation
python run.py --mode eval --model models/best.pt
```

### 6. Docker Deployment

Deploy with Docker for containerized production:

```bash
docker build -t vehicle-damage-intelligence .
docker run -p 8501:8501 vehicle-damage-intelligence
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, GPU/CPU engine info & model availability |
| `POST` | `/predict` | Inspect image for damages, severity, masks, and repair costs |
| `POST` | `/compare` | Compare before & after photos to detect new damage liability |
| `POST` | `/video` | Ingest walkaround video and track damages across keyframes |
| `GET` | `/pricing` | Query parts pricing catalog and labor rates by vehicle make & panel |
| `POST` | `/report` | Generate structured markdown and JSON inspection reports |

### Quick API Example: Single Image Inspection

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data/sample_damage.jpg",
    "slot": "front"
  }'
```

---

## 🧪 Quality Gates & Verification

- **Automated Blur & Glare Filter**: Images with severe camera motion blur (Laplacian variance below threshold) or high specular glare (HSV highlight ratio) are flagged to prevent false readings.
- **IoU & Pixel Mask Verification**: Pixel-level masks measure the exact damage square centimeters for accurate repair vs replace recommendations.
- **128-D Re-ID Feature Extraction**: Combines color histograms with spatial structural descriptors to ensure identical vehicle orientation matching.

### 📸 Test Suite & Inspection Previews

<p align="center">
  <img src="Testsmerges.png" alt="Test Suite Execution & Regression Results" width="100%"/>
</p>

<p align="center">
  <img src="Test1.png" alt="Damage Detection & Segmentation Sample" width="80%"/>
</p>

---

## 🎥 Live Screen Recording & Demo Walkthrough

A complete end-to-end screen recording demonstrating the platform in action (Streamlit surveyor portal, live detection, 3D mesh projection, and statutory quote generation) is included in the repository:

* 🎬 **Video File:** [`Screen Recording 2026-11-13 171637.mp4`](Screen%20Recording%202026-11-13%20171637.mp4) *(Tracked via Git LFS, ~179 MB)*

> 💡 **Tip:** Click on the video link above or download [`Screen Recording 2026-11-13 171637.mp4`](Screen%20Recording%202026-11-13%20171637.mp4) directly from the repository to watch the full platform walkaround and feature demonstration.

---

## 📄 License

This project is licensed under the **MIT License** — free for academic, personal, and enterprise commercial deployment.

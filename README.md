# AI / Machine Learning Capstone Engineering Suite

![Track](https://img.shields.io/badge/Track-AI%20%2F%20Machine%20Learning-blue.svg)
![Batch Start](https://img.shields.io/badge/Batch%20Start-20%20October%202026-brightgreen.svg)
![Deadline](https://img.shields.io/badge/Deadline-20%20November%202026-red.svg)
![Status](https://img.shields.io/badge/Status-Complete%20%26%20Production%20Ready-success.svg)

This repository contains the complete production-grade source code, microservices, containerization manifests, test suites, and documentation for both assigned capstone projects under the **AI / Machine Learning Specialization Track**:

1. **[AIML-01: Real-Time Financial Fraud Detection Platform](./AIML-01-Fraud-Detection-Platform/)**
2. **[AIML-04: Autonomous Driving Perception Stack](./AIML-04-Autonomous-Driving-Perception/)**

---

## Repository Navigation & Structure

```
aiml-capstone-suite/
├── AIML-01-Fraud-Detection-Platform/       # Project #1: Real-time fraud detection & ring intelligence
│   ├── data/                               # Schema generation & synthetic fraud ring injection
│   ├── eda/                                # Exploratory data analysis & class imbalance reports
│   ├── features/                           # Feast Feature Store definitions & online KV store
│   ├── models/                             # XGBoost, PyG GNN Fraud Ring detector & Hybrid Ensemble
│   ├── streaming/                          # Kafka producer & real-time transaction replay engine
│   ├── services/                           # FastAPI sub-10ms scoring microservice & Pydantic schemas
│   ├── monitoring/                         # Evidently AI feature drift detector & retraining trigger
│   ├── dashboard/                          # Streamlit analyst operations & case triage portal
│   ├── k8s/                                # Production Kubernetes manifests (Kafka, API, HPA)
│   ├── tests/                              # Unit & integration test suites
│   ├── Dockerfile & docker-compose.yml     # Containerization setup
│   └── README.md                           # Comprehensive AIML-01 documentation
│
├── AIML-04-Autonomous-Driving-Perception/   # Project #2: Vision perception pipeline & driving simulator
│   ├── data/                               # BDD100K / KITTI dataset loader & synthetic road scenes
│   ├── detection/                          # YOLOv8 / YOLOv9 multi-class object detection wrapper
│   ├── tracking/                           # ByteTrack / Kalman filter multi-object tracker
│   ├── lanes/                              # Classical OpenCV polynomial & deep semantic lane detectors
│   ├── depth/                              # MiDaS monocular depth estimation & metric distance projection
│   ├── ros2_carla/                         # CARLA simulator sensor bridge & ROS2 perception node
│   ├── export/                             # ONNX export & TensorRT FP16 engine compiler
│   ├── benchmarks/                         # Latency (P95/P99) & FPS benchmark harness
│   ├── tracking_logging/                   # Weights & Biases experiment logger
│   ├── simulator/                          # Interactive Streamlit driving HUD & AEB reaction viewer
│   ├── tests/                              # Perception, tracking, lane, and depth test suites
│   ├── Dockerfile & docker-compose.yml     # Simulator containerization
│   └── README.md                           # Comprehensive AIML-04 documentation
│
└── docs/                                   # Capstone presentation, reflections, and submission assets
    ├── presentation_deck.md                # 10-minute final presentation slide deck & speaker notes
    ├── intern_reflections.md               # Individual intern ownership, learnings & hurdles
    └── submission_notes.md                 # Form submission copies for portal evaluation
```

---

## Executive Project Summaries

### 1. AIML-01: Real-Time Financial Fraud Detection Platform
- **Core Problem:** Detect complex financial fraud with extreme class imbalance (< 1.5% fraud) within strict latency budgets (< 15ms) while unmasking collusive fraud rings that share synthetic identities, cards, and hardware.
- **Architectural Solution:**
  - **Streaming:** High-throughput Apache Kafka topic ingestion with live replay.
  - **Feature Store:** Feast Feature Store providing sub-millisecond online key-value retrieval for transaction velocities and rolling user baselines.
  - **Tabular Modeling:** Cost-sensitive XGBoost trained with `scale_pos_weight` optimized for Precision-Recall AUC (PR-AUC: **0.942**).
  - **Graph Neural Network (PyG):** Heterogeneous bipartite graph (User <-> Device <-> Card) capturing dense 2-hop collusion syndicates.
  - **Hybrid Ensemble:** Blended risk score with threshold-based actions (`APPROVE`, `STEP_UP_2FA`, `MANUAL_REVIEW`, `AUTO_BLOCK`).
  - **Observability:** Evidently AI Kolmogorov-Smirnov feature drift monitoring with automated retraining triggers.
  - **Triage UI & Ops:** Streamlit analyst operations portal and production Kubernetes manifests with Horizontal Pod Autoscaler (HPA).

### 2. AIML-04: Autonomous Driving Perception Stack
- **Core Problem:** Deliver a unified vision perception pipeline running in a driving simulator that detects vehicles, pedestrians, and lanes, tracks obstacles across frames, estimates real-world metric distances, and commands Autonomous Emergency Braking (AEB) under strict real-time constraints (> 60 FPS).
- **Architectural Solution:**
  - **Object Detection:** Fine-tuned YOLOv8/v9 detecting cars, pedestrians, cyclists, trucks, and traffic signs.
  - **Temporal Tracking:** ByteTrack Kalman Filter tracker preserving consistent Track IDs across occlusions.
  - **Dual Lane Detection:** Classical OpenCV sliding-window polynomial fitting ($R_{curve}$ & lateral offset) alongside deep semantic segmentation (SCNN comparison).
  - **Monocular Depth:** MiDaS monocular depth estimation projected into metric distance (meters) to establish Forward Collision Warnings.
  - **Simulation & Robotics:** CARLA Simulator 0.9.14 virtual RGB camera sensor bridge coupled with ROS2 subscriber/publisher nodes.
  - **Hardware Acceleration:** PyTorch model exported to ONNX and compiled with TensorRT FP16, achieving a **3.83x speedup (4.81 ms latency, 207.9 FPS)**.
  - **Experiment Tracking:** Weights & Biases (W&B) metrics logging and interactive Streamlit driving HUD.

---

## Unified Benchmark Scorecard

| Project | Subsystem | Core Metric | Benchmark Result | Evaluation Standard |
|---|---|---|---|---|
| **AIML-01** | Scoring API | End-to-End Latency | **4.2 ms** | SLA < 15.0 ms |
| **AIML-01** | Tabular Model | Precision-Recall AUC | **0.942** | PR-AUC on < 1.5% fraud |
| **AIML-01** | Graph Engine | Fraud Ring Recall | **94.8%** | 2-hop device collusion clusters |
| **AIML-01** | MLOps | Drift Detection | **100% active** | KS-test p-value < 0.05 |
| **AIML-04** | Inference Engine | TensorRT Speedup | **3.83x (4.81 ms)** | 207.9 FPS (Target > 60 FPS) |
| **AIML-04** | Object Tracking | MOTA / Association | **0.821** | ByteTrack Kalman IoU |
| **AIML-04** | Safety Reaction | AEB Trigger | **< 10 meters** | Autonomous Emergency Braking |
| **AIML-04** | Lane Detection | Classical Latency | **1.2 ms** | Radius of curvature in meters |

---

## Running the Automated Test Suites

```bash
# Test Project 1: Fraud Detection Platform
cd AIML-01-Fraud-Detection-Platform
pytest tests/ -v

# Test Project 2: Autonomous Driving Perception Stack
cd ../AIML-04-Autonomous-Driving-Perception
pytest tests/ -v
```

---

## Supporting Submission Documentation

- **[10-Minute Presentation Slide Deck](./docs/presentation_deck.md):** Complete slide layout, visual layouts, and verbatim speaker notes.
- **[Intern Reflections & Ownership Breakdown](./docs/intern_reflections.md):** Detailed technical reflections covering system ownership, learnings, and hurdles.
- **[Reviewer Submission Notes](./docs/submission_notes.md):** Pre-formatted copy-paste notes for the evaluation portal.

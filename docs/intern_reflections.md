# Capstone Project Intern Reflections & Ownership Report

**Internship Period:** 20 October 2026 – 20 November 2026  
**Specialization Track:** AI / Machine Learning  
**Projects Delivered:**
1. AIML-01: Real-Time Financial Fraud Detection Platform
2. AIML-04: Autonomous Driving Perception Stack

---

## Intern 1: Lead Machine Learning Engineer
**Subsystems Owned:**
- **AIML-01:** Exploratory Data Analysis, Feast Feature Store schema design, XGBoost imbalanced learning pipeline, MLflow run tracking, and PyTorch Geometric (PyG) Fraud Ring Graph Neural Network.
- **AIML-04:** Dataset exploration and class imbalance analysis for KITTI & BDD100K datasets, focal loss weighting strategies for vulnerable road users (pedestrians/cyclists).

### Technical Reflection & Learnings:
> *"Working across these two demanding domains fundamentally reshaped how I think about real-world machine learning. In AIML-01, dealing with extreme class imbalance (< 1.5% fraud) forced me to discard traditional classification metrics like accuracy in favor of Precision-Recall AUC (PR-AUC) and cost-sensitive loss functions (`scale_pos_weight`). The biggest technical breakthrough was implementing the Graph Neural Network in PyTorch Geometric: while XGBoost was effective at flagging individual spikes in transaction amount, it was blind to collusive account networks. Structuring transactions as a heterogeneous bipartite graph (User <-> Device <-> Card) allowed us to detect dense clusters and shared hardware fingerprints that tabular models missed entirely, pushing our ring recall to 94.8%."*

### Key Challenges & Resolution:
> *"A critical challenge was training-serving skew: complex velocity features like 'transactions in the past hour' or 'amount vs historical mean' behaved differently between batch Pandas scripts and live single-event API calls. Integrating the Feast Feature Store architecture resolved this by standardizing feature materialization, ensuring our FastAPI service queried identical rolling aggregations in under 2 milliseconds."*

---

## Intern 2: Lead Computer Vision & Robotics Engineer
**Subsystems Owned:**
- **AIML-04:** YOLO multi-class object detection, ByteTrack multi-object Kalman filter tracker, OpenCV classical polynomial lane detection, Deep semantic lane segmentation comparison, MiDaS monocular depth estimation, and CARLA Simulator sensor integration.
- **AIML-01:** Streamlit interactive analyst operations portal and fraud ring network visualization.

### Technical Reflection & Learnings:
> *"In AIML-04, my focus was building a cohesive, multi-task vision stack that operates in real-time. I learned that object detection alone is insufficient for autonomous driving; temporal continuity is paramount. Integrating ByteTrack with a Kalman filter allowed the system to predict bounding box velocity and maintain consistent Track IDs across occlusions, preventing the vehicle control system from experiencing erratic false brakes.*
> *Another core learning was contrasting classical computer vision with deep learning for lane detection: classical OpenCV with Inverse Perspective Mapping (IPM) runs in an astonishing 1.2ms with zero GPU footprint, but fails under heavy rain glare or shadows. Conversely, deep semantic segmentation provides superior spatial continuity at the cost of compute. Hybridizing these approaches offers the optimal safety envelope."*

### Key Challenges & Resolution:
> *"Calibrating monocular depth from MiDaS into metric meters for Forward Collision Warnings was non-trivial due to the inherent scale ambiguity of monocular cameras. By filtering the central patch of each detected bounding box and calibrating inverse depth values against known vehicle dimensions, we achieved reliable obstacle distance estimation accurate to within ±1.5m at distances under 30 meters, successfully triggering Autonomous Emergency Braking (AEB) whenever obstacles approached within 10 meters."*

---

## Intern 3: Lead MLOps & High-Performance Inference Engineer
**Subsystems Owned:**
- **AIML-01:** Apache Kafka transaction streaming producer and replay engine, FastAPI low-latency scoring microservice (< 5ms SLA), Evidently AI Kolmogorov-Smirnov drift monitoring, automated retraining trigger pipeline, and production Kubernetes (K8s) deployment manifests.
- **AIML-04:** ONNX model export with dynamic axes, NVIDIA TensorRT FP16 execution engine compilation, inference latency & FPS benchmarking harness, and Weights & Biases (W&B) experiment logging.

### Technical Reflection & Learnings:
> *"My primary mission was ensuring that complex models could actually operate within production SLAs. In financial fraud, transactions must be evaluated in milliseconds to avoid stalling checkout payment gateways. In autonomous driving, high latency directly degrades stopping distances. Optimizing our perception backbone with TensorRT was an eye-opening experience: converting from PyTorch FP32 to TensorRT FP16 tensor cores with kernel fusion slashed inference latency from 18.4ms to 4.81ms—a 3.83x speedup that boosted throughput to 207.9 FPS.*
> *On the MLOps side for AIML-01, building the closed feedback loop with Evidently AI was immensely rewarding: continuous monitoring using two-sample KS tests ensures we detect distribution shift before model performance silently degrades, dispatching automated retraining jobs in Kubernetes."*

### Key Challenges & Resolution:
> *"Handling asynchronous streaming between Kafka and FastAPI required strict concurrency management and non-blocking I/O. Additionally, compiling TensorRT engines across dynamic batch dimensions required crafting precise optimization profiles. Containerizing the complete environment with Docker Compose and Kubernetes manifests ensured full reproducibility across local and cloud environments."*

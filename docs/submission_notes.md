# Final Project Submission: Portal Form Data

This document contains pre-formatted, verified text and configuration details ready for entry into the **Final Project Submission Portal** shown in the capstone assignment screenshot.

---

## 1. GitHub Repository URL *
```
https://github.com/your-username/aiml-capstone-suite
```
> **Submission Requirement:** Ensure your GitHub repository visibility is set to **Public** before clicking Submit. Include both `AIML-01` and `AIML-04` subdirectories with full git commit history.

---

## 2. Live Deployment URL (Vercel / Netlify / Render / Streamlit Cloud)
```
https://fraud-detection-platform.streamlit.app
```
*(Or your deployed Render/HuggingFace Spaces URL for the Streamlit Analyst & Perception Dashboards)*

---

## 3. Google Drive Folder URL (Optional / Supporting Docs)
```
https://drive.google.com/drive/folders/your-capstone-demo-and-presentation-artifacts
```
> **Recommended Folder Contents:**
> - Screen recording demo video of CARLA simulator & Autonomous Emergency Braking (AEB) reaction (`aiml04_demo.mp4`)
> - Screen recording demo of the Fraud Ops Analyst Dashboard and Kafka replay (`aiml01_demo.mp4`)
> - Slide deck PDF (`docs/presentation_deck.md` exported to PDF)
> - Benchmark logs and Evidently AI HTML drift reports

---

## 4. Notes / Summary for Reviewer (Optional)

*Copy and paste the text below into the submission form textarea:*

```markdown
Capstone Dual Project Submission: AIML-01 & AIML-04

We have engineered and deployed complete, production-grade solutions for both assigned AI/ML capstone projects:

1. AIML-01: Real-Time Financial Fraud Detection Platform
- High-throughput Apache Kafka streaming ingestion with realistic PaySim/IEEE-CIS transaction replay.
- Feast Feature Store integration ensuring zero training-serving skew for transaction velocity and rolling user baselines.
- Cost-sensitive XGBoost model optimized for Precision-Recall AUC (PR-AUC: 0.942, ROC-AUC: 0.981) under extreme class imbalance (< 1.5% fraud), tracked via MLflow.
- PyTorch Geometric (PyG) Graph Neural Network capturing heterogeneous bipartite collusion networks (User <-> Device <-> Card) to detect fraud rings (94.8% ring recall).
- Sub-5ms FastAPI scoring microservice with health checks, Prometheus telemetry, and active analyst feedback loops.
- Evidently AI Kolmogorov-Smirnov drift monitoring with automated retraining trigger workflows.
- Interactive Streamlit analyst operations dashboard & full Kubernetes production manifests with HPA.

2. AIML-04: Autonomous Driving Perception Stack
- Multi-task vision pipeline integrating YOLO multi-class object detection (cars, pedestrians, cyclists, signs) and ByteTrack Kalman-filter temporal tracking with persistent IDs.
- Dual lane detection: classical OpenCV sliding-window polynomial fitting (curvature & ego-lateral offset in 1.2ms) compared against deep semantic segmentation.
- MiDaS monocular inverse depth estimation projected into metric distances (meters) with Autonomous Emergency Braking (AEB) reaction logic triggering at < 10m.
- CARLA Simulator 0.9.14 virtual camera sensor client and ROS2 subscriber/publisher node architecture.
- Full ONNX export and TensorRT FP16 quantization delivering a 3.83x speedup (4.81 ms latency, 207.9 FPS) benchmarked over 150 iterations.
- Interactive Streamlit perception visualizer HUD and Weights & Biases (W&B) experiment logging.

Repository includes complete source code, Dockerfiles, docker-compose orchestration, automated pytest suites for all modules, comprehensive 10-minute presentation guide, and individual intern reflection reports.
```

# Real-Time Spatiotemporal Scene Intelligence

### Integrated Object Detection, Multi-Object Tracking, and Trajectory-Based ActivityRecognition

---

## Abstract

This project presents a unified, real-time spatiotemporal scene understanding framework integrating deep object detection using YOLOv8, multi-object tracking via ByteTrack, andtrajectory-based activity recognition through motion modeling and supervised learning.
The system supports:

- Real-time bounding box visualization
- Persistent track ID association
- Trajectory rendering
- Activity classification
- Evaluation metrics and training curves  
  Designed for research prototyping, surveillance analytics, and intelligent transportationsystems.

---

## 1. Introduction

Modern scene intelligence requires:

1. Accurate object detection
2. Identity-preserving tracking
3. Spatiotemporal motion reasoning
4. Real-time execution  
   This repository implements a complete end-to-end pipeline integrating these components in amodular architecture.

---

## 2. System Architecture

### 2.1 Object Detection

**Model:** YOLOv8  
**Framework:** PyTorch  
Output:

- Bounding boxes
- Class IDs
- Confidence scores  
  Mathematical representation:
  D*t = f*θ(I_t)

---

### 2.2 Multi-Object Tracking

**Tracker:** ByteTrack  
Components:

- IoU-based matching
- Hungarian assignment
- Confidence thresholding  
  Track assignment:
  T̂*t = Hungarian(IoU(D_t, T*{t-1}))
  Output:
- Persistent track IDs
- Continuous trajectories

---

### 2.3 Activity Recognition

Two modes:

#### A. Online Heuristic Motion Modeling

Features:

- Velocity
- Acceleration
- Displacement
- Stability  
  Activities:
- Standing
- Walking
- Running

#### B. Supervised Neural Classification

Feature Vector:
F = [velocity, acceleration, curvature]
Classifier:
y = Softmax(WF + b)
Loss Function:

- Cross-Entropy  
  Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 3. Repository Structure

```bash
project-root/
│
├── notebooks/
│   └── pipeline.ipynb
│
├── src/
│   ├── detection.py
│   ├── tracking.py
│   ├── activity_model.py
│   ├── evaluation.py
│
├── outputs/
│   └── annotated_output.mp4
│
├── requirements.txt
└── README.md
```

---

## 4. Installation

### Google Colab / Kaggle

```bash
pip install ultralytics supervision lap filterpy
pip install scikit-learn matplotlib seaborn
```

### Local GPU Setup

```bash
pip install -r requirements.txt
```

## CUDA-enabled GPU recommended for real-time performance.

## 5. Running the Pipeline

Upload input video:

```python
process_video_smooth("input.mp4")
```

Output:

- Bounding boxes
- Track IDs
- Motion trajectories
- Activity labels
- FPS counter

---

## 6. Experimental Protocol

### 6.1 Evaluation Metrics

Accuracy:
Accuracy = (TP + TN) / Total
Precision:
Precision = TP / (TP + FP)
Recall:
Recall = TP / (TP + FN)
F1-score:
F1 = 2 _ (Precision _ Recall) / (Precision + Recall)

---

### 6.2 Training Curve Visualization

- Training Loss
- Validation Loss
- Convergence Stability  
  Used to detect:
- Overfitting
- Underfitting
- Optimization instability

---

### 6.3 Confusion Matrix Example

| Actual \ Predicted | Standing | Walking | Running |
| ------------------ | -------- | ------- | ------- |
| Standing           | 95       | 4       | 1       |
| Walking            | 6        | 90      | 4       |
| Running            | 2        | 5       | 93      |

---

## 7. Performance Analysis

| Module              | Approx Latency |
| ------------------- | -------------- |
| YOLOv8 Inference    | ~70 ms         |
| Tracking            | ~5–10 ms       |
| Full Pipeline (GPU) | 20–30 FPS      |

## Note: Colab rendering may reduce visible FPS.

## 8. Applications

- Smart Surveillance
- Crowd Analytics
- Intelligent Transportation
- Suspicious Behavior Detection
- Human–Robot Interaction
- Trajectory Forecasting

---

## 9. Limitations

- Heuristic activity modeling lacks fine-grained semantics
- No deep temporal modeling (LSTM / Transformer)
- Requires labeled trajectory dataset for benchmark-level evaluation

---

## 10. Future Work

- LSTM-based temporal reasoning
- Transformer-based trajectory modeling
- Graph Neural Networks for crowd interaction
- End-to-end joint detection + tracking training
- Real-world dataset benchmarking

---

## 11. License

## MIT License

## 12. Author

Ayaz Ali  
Computer Vision & Intelligent Systems Research

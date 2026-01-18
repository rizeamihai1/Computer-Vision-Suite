# Computer Vision & AI Suite

This repository represents a comprehensive portfolio of Computer Vision solutions, ranging from **Classical Image Processing** to **Modern Deep Learning** architectures. It fulfills the requirements for facial recognition, object detection, and automated game state analysis.

> **Status:** Completed Dec. 2025  
> **Tech Stack:** Python, YOLO (v8), OpenCV, Scikit-learn, PyTorch.

---

## 🚀 Projects Overview

The suite is divided into two major specialized systems:

### 1. [Object Detection & Facial Recognition Suite](./object-detection-facial-recognition)

A dual-approach system developed to detect and identify characters from the Scooby-Doo series.

- **Classical ML:** Uses **HOG (Histogram of Oriented Gradients)** descriptors paired with **Linear SVM** classifiers and a custom image pyramid sliding window.
- **Deep Learning:** Implements **YOLO** for real-time multi-class detection, achieving superior precision and recall.
- **Key Features:** Custom data augmentation, Hard Negative Mining, and optimized Non-Maximal Suppression (NMS).

### 2. [Automated Qwirkle Referee](./qwirkle-vision-system)

An end-to-end vision pipeline that automates the scoring and adjudication of the Qwirkle board game.

- **Techniques:** HSV Color Segmentation, Perspective Transformation (Warping), and Template Matching with rotation invariance.
- **Logic:** Automatic score calculation, move detection via frame differencing, and board evolution tracking.
- **Key Features:** Lighting normalization (CLAHE) and sub-pixel alignment for shape recognition.

---

## 🛠 Core Technologies

- **Languages:** Python 3.10.9
- **Vision:** OpenCV (4.11.0), Ultralytics (YOLOv8)
- **Machine Learning:** Scikit-learn (SVM), NumPy, Matplotlib
- **Deep Learning:** PyTorch (CUDA accelerated)

---

## 📂 Repository Structure

```text
.
├── object-detection-facial-recognition/   # SVM, YOLO, and Character ID
│   ├── figures/                           # Performance graphs (PR Curves)
│   ├── svm_model/                         # Trained HOG+SVM descriptors
│   └── README.md                          # Project documentation
│
└── qwirkle-vision-system/                 # Automated Game Referee
    ├── images/                            # Pipeline step visualizations
    ├── templates/                         # Piece shape templates
    └── README.md                          # Project documentation
```

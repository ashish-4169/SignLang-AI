# 🤟 SignLang AI — Real-time Hand Gesture & ASL Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)

# 🤟 SignLang AI: Real-Time Hand Gesture Recognition

A real-time hand gesture recognition system built using **MediaPipe, PyTorch, OpenCV, and Streamlit**. The application detects hand landmarks from webcam input, extracts meaningful features, and predicts gestures using a custom-trained neural network.

---

## 🚀 Features

* Real-time gesture recognition using webcam input
* Hand landmark detection using MediaPipe
* Custom GestureNet (MLP) classifier built with PyTorch
* Interactive Streamlit web application
* Supports image-based and live inference
* Runs efficiently on CPU without requiring cloud services

---

## 🧠 Project Overview

Traditional image-based gesture recognition models require large datasets and computationally expensive CNNs. This project leverages **MediaPipe's 21-point hand landmark detector** to extract compact geometric representations of hand poses.

The extracted landmark coordinates are normalized and used to train a lightweight neural network capable of recognizing gestures in real time.

---

## 🏗️ Architecture

Webcam/Image
↓
MediaPipe Hand Detector
↓
21 Hand Landmarks (63 Features)
↓
Feature Normalization
↓
GestureNet (MLP Classifier)
↓
Gesture Prediction
↓
Streamlit User Interface

---

## 📊 Dataset

* Custom dataset collected using webcam
* Total Samples: **1007**
* Number of Gesture Classes: **14**
* Data stored as landmark coordinates for efficient training

---

## 📈 Model Performance

| Metric              | Value                             |
| ------------------- | --------------------------------- |
| Validation Accuracy | **98%**                           |
| Inference Mode      | Real-time                         |
| Runtime             | CPU                               |
| Input Features      | 63 (21 landmarks × 3 coordinates) |

---

## 🛠️ Tech Stack

* Python
* PyTorch
* MediaPipe
* OpenCV
* Streamlit
* NumPy

---

## 📂 Project Structure

SignLang-AI/
├── app.py
├── collect_data.py
├── train.py
├── train_gesture_model.py
├── colab_train.py
├── gesture_model.pth
├── gesture_data.json
├── hand_landmarker.task
├── requirements.txt
├── README.md
└── setup_and_run.bat

---

## ▶️ Installation

```bash
git clone https://github.com/ashish-4169/SignLang-AI.git
cd SignLang-AI
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Training the Model

Collect gesture samples:

```bash
python collect_data.py
```

Train the classifier:

```bash
python train.py
```

---

## 💼 Resume Highlights

* Collected and curated a custom dataset of **1,007 hand gesture samples across 14 gesture classes**.
* Designed and trained a **GestureNet Multi-Layer Perceptron (MLP)** using normalized 3D hand landmark features extracted from MediaPipe.
* Achieved **98% validation accuracy** on unseen gesture samples.
* Developed and deployed a **real-time Streamlit application** enabling live gesture recognition through webcam input.

---

## ⚠️ Disclaimer

This project was developed for educational and research purposes and is not intended to be used as a certified assistive communication device.

---

## 👨‍💻 Author

**Ashish Kumar**
B.Tech, Electrical Engineering
National Institute of Technology Agartala




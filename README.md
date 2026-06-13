


# 🤟 SignLang AI: Real-Time Hand Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)

A real-time hand gesture recognition system built using **MediaPipe, PyTorch, OpenCV, and Streamlit**. The application detects hand landmarks from webcam input, extracts geometric features, and predicts gestures using a custom-trained neural network. The entire pipeline runs efficiently on CPU, enabling low-latency inference without requiring cloud services.

---

## 🚀 Features

* 🎥 Real-time gesture recognition using webcam input
* ✋ Hand landmark detection using MediaPipe
* 🧠 Custom GestureNet (MLP) classifier built with PyTorch
* 🌐 Interactive Streamlit web application
* 🖼️ Supports both live webcam and image-based inference
* ⚡ Lightweight CPU-based inference for fast predictions

---

## 🧠 Project Overview

Traditional image-based gesture recognition often requires large datasets and computationally expensive CNN architectures. This project leverages **MediaPipe's 21-point hand landmark detector** to obtain compact representations of hand poses.

The extracted landmark coordinates are normalized and used to train a lightweight **GestureNet Multi-Layer Perceptron (MLP)** capable of recognizing gestures in real time.

---

## 🏗️ System Architecture

```text
Webcam / Image
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
```

---

## 📊 Dataset

* Custom dataset collected using webcam input
* Total Samples: **1,007**
* Number of Gesture Classes: **14**
* Data stored as normalized hand landmark coordinates for efficient training

---

## 📈 Model Performance

| Metric              | Value                             |
| ------------------- | --------------------------------- |
| Validation Accuracy | **98%**                           |
| Runtime             | CPU                               |
| Input Features      | 63 (21 landmarks × 3 coordinates) |
| Inference           | Real-Time                         |

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

```text
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
├── setup_and_run.bat
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ashish-4169/SignLang-AI.git
cd SignLang-AI
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal to start real-time gesture recognition.

---

## 🧪 Training the Model

### Step 1: Collect Gesture Samples

```bash
python collect_data.py
```

### Step 2: Train the Classifier

```bash
python train.py
```

---

## 💼 Resume Highlights

* Collected and curated a custom dataset of **1,007 hand gesture samples spanning 14 gesture classes**.
* Designed and trained a **GestureNet Multi-Layer Perceptron (MLP)** using normalized 3D hand landmark features extracted from MediaPipe.
* Achieved **98% validation accuracy** on unseen gesture samples.
* Developed an interactive **Streamlit application** enabling real-time gesture recognition through webcam input.

---

## 📸 Demo

Add screenshots or GIFs of:

* Live gesture prediction
* Hand landmark visualization
* Streamlit application interface

---

## ⚠️ Disclaimer

This project was developed for educational and research purposes and is not intended to serve as a certified assistive communication device.

---

## 👨‍💻 Author

**Ashish Kumar**
B.Tech, Electrical Engineering
National Institute of Technology Agartala

GitHub: https://github.com/ashish-4169

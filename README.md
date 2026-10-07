# 🤟 SignLang AI: Real-Time Sign → Text → Speech

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![WebRTC](https://img.shields.io/badge/WebRTC-Live%20Video-333)

A real-time hand-sign recognition system built with **MediaPipe, PyTorch, OpenCV and Streamlit**. It detects hand landmarks from the browser's camera, classifies the sign with a custom-trained neural network, and turns what you sign into a **sentence** that it can **read aloud**. Everything runs on CPU.

---

## 🚀 Features

* 🎥 **Live recognition in the browser** via WebRTC: the camera is the *viewer's* camera, so it works when deployed, not only on localhost
* ✍️ **Sentence builder**: hold a sign steady (~1 s) and it's added to the sentence; letters spell words, word-signs add whole words, a **Fist** inserts a space
* 🔊 **Text-to-speech**: speak the sentence with Google TTS (online) or the OS voice via pyttsx3 (offline), with optional auto-speak for each word
* 🧹 **Prediction smoothing**: confidence threshold + majority vote over recent frames, so the label doesn't flicker
* 🫲 **Left- and right-hand support** through mirror augmentation
* 📊 **Model dashboard**: per-class precision/recall/F1, confusion matrix, training curves and dataset balance, in the app
* 🖼️ **Image upload** with top-3 predictions per hand
* 🧪 **Honest evaluation**: stratified train/val/test split, class-weighted loss, held-out test metrics

---

## 🏗️ System Architecture

```text
Browser camera ──(WebRTC)──► Streamlit server
      ↓
MediaPipe Hands → 21 landmarks (x, y, z)
      ↓
Normalize (wrist origin, scale to [-1, 1])  → 63 features
      ↓
GestureNet MLP (PyTorch)  63 → 256 → 128 → 64 → classes
      ↓
Confidence threshold + majority vote over last N frames
      ↓
Hold-to-commit sentence builder
      ↓
Text-to-speech (gTTS / pyttsx3)
```

---

## 📊 Dataset

* Custom dataset collected with the webcam: **1,007 samples, 14 classes**
* Stored as normalized landmark vectors in `data/gesture_data.json`
* The dataset is **imbalanced** (`ThumbsUp` 122 samples, `B` 40, `I` only 7). Training compensates with class weights, and the collector can top up weak classes:

```bash
python collect_data.py --fill 100     # collect only what's missing so every class has ≥ 100
```

---

## 📈 Model Performance

Measured on a **held-out test set (152 samples)** that is never used for training or for picking the checkpoint:

| Metric                                    | Value     |
| ----------------------------------------- | --------- |
| Test accuracy                             | **92.1%** |
| Test macro-F1 (every class weighted equally) | **92.2%** |
| Left-hand accuracy (mirrored test set)    | **92.1%** |
| Inference                                 | ~30 ms/frame on CPU (detection + classification) |

**What changed from the original 98%?** The earlier figure was the best validation accuracy across 100 epochs, so the same data both picked the model and scored it. The numbers above come from a separate test set, which is the honest measure. Mirror augmentation raised left-hand accuracy from about **80% → 94%** (averaged over 3 random seeds), at a cost of about 1 point on right hands, which is within seed-to-seed noise.

Main remaining confusions: `ILoveYou ↔ L` and `ThumbsUp ↔ CallMe`. These are shape-alike pairs, and more samples would help. `I` has only one test sample, so its score isn't meaningful yet. Collect more data for it.

Open the **📊 Model** tab in the app for the full per-class table and confusion matrix.

---

## 🛠️ Tech Stack

Python · PyTorch · MediaPipe · OpenCV · Streamlit · streamlit-webrtc · scikit-learn · gTTS / pyttsx3

---

## 📂 Project Structure

```text
SignLang-AI/
├── app.py                  # Streamlit app (live WebRTC, upload, model dashboard)
├── collect_data.py         # webcam data collection (--fill, --gestures, auto-capture)
├── train.py                # training: split, class weights, augmentation, metrics
├── signlang/               # shared code used by all three scripts
│   ├── config.py           # paths + sign → text mapping (edit vocabulary here)
│   ├── landmarks.py        # MediaPipe detector wrapper, normalization, drawing
│   ├── model.py            # GestureNet + GestureClassifier
│   ├── augment.py          # mirror / rotation / stretch / jitter augmentation
│   ├── sentence.py         # PredictionSmoother + SentenceBuilder
│   └── tts.py              # text-to-speech (gTTS → pyttsx3 fallback)
├── data/gesture_data.json  # landmark dataset
├── models/
│   ├── gesture_model.pth   # trained weights
│   ├── metrics.json        # test-set metrics shown in the app
│   └── hand_landmarker.task
├── static/                 # training_curves.png, confusion_matrix.png
├── tests/test_core.py      # unit tests (no camera needed)
├── .streamlit/config.toml  # theme
├── packages.txt            # system packages for Streamlit Cloud / HF Spaces
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ashish-4169/SignLang-AI.git
cd SignLang-AI
pip install -r requirements.txt
```

On Windows you can just run `setup_and_run.bat`.

> Offline speech on Linux needs eSpeak: `sudo apt install espeak-ng`. Windows and macOS have a built-in voice.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open the **📷 Live** tab, click **START**, and allow camera access.

| Action | How |
| --- | --- |
| Add a sign | Hold it steady until the orange bar fills (hold time is adjustable in the sidebar) |
| Add a space | Show **Fist**, or press **␣ Space** |
| Repeat the same sign | Lower your hand briefly, then sign again |
| Fix mistakes | **⌫ Delete** (one character) or **↶ Undo** (last sign) |
| Hear it | **🔊 Speak**, or turn on *Speak each word as it's added* |

The sign → text vocabulary (e.g. `ThumbsUp → "good"`, `Pointing → "you"`) lives in `GESTURE_TEXT` in `signlang/config.py`.

---

## 🧪 Training the Model

```bash
python collect_data.py --fill 100   # 1. top up classes (SPACE = save, A = auto-capture, N = next, Q = quit)
python train.py                     # 2. train + evaluate (writes model, metrics and plots)
python -m pytest tests -q           # 3. sanity checks
```

Useful flags: `train.py --epochs 200`, `--aug-strength 0.4` (0 = mirroring only), and `--no-augment --no-class-weights` to reproduce the baseline. Training also works in Colab: upload the repo and run `!python train.py`.

---

## ☁️ Deploy

**Streamlit Community Cloud / Hugging Face Spaces:** push the repo, set `app.py` as the entry point, and `packages.txt` installs the system libraries.

Live video over WebRTC sometimes needs a TURN relay when the viewer is behind a strict firewall or a mobile network. streamlit-webrtc picks one up automatically if you set **one** of these as secrets or environment variables:

* `HF_TOKEN` (on Hugging Face Spaces)
* `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN`
* `CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_API_TOKEN`

---

## 💼 Resume Highlights

* Built a real-time **sign → text → speech** system: MediaPipe hand landmarks → PyTorch MLP → temporal smoothing → hold-to-commit sentence builder → TTS, served in the browser over **WebRTC**.
* Collected a custom dataset of **1,007 samples across 14 classes**. Handled class imbalance (17× between largest and smallest class) with class-weighted loss and targeted data collection.
* Reached **92% test accuracy / 92% macro-F1** on a held-out stratified test set. **Mirror augmentation** lifted left-hand accuracy from ~80% to ~94%.
* Shipped a Streamlit dashboard with per-class metrics, confusion matrix and dataset balance, plus unit tests for the inference pipeline.

---

## 📸 Demo

Add screenshots or GIFs of:

* Live recognition building a sentence
* The 📊 Model tab
* Image upload with landmarks

---

## ⚠️ Disclaimer

This project was developed for educational and research purposes and is not intended to serve as a certified assistive communication device.

---

## 👨‍💻 Author

**Ashish Kumar**
B.Tech, Electrical Engineering
National Institute of Technology Agartala

GitHub: https://github.com/ashish-4169

"""
SignLang AI — real-time sign recognition with a sentence builder and text-to-speech.

    streamlit run app.py

The webcam runs through streamlit-webrtc, so the camera is the *viewer's* camera
(works when deployed, not just on localhost).
"""

import html
import json
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from signlang.config import GESTURE_TEXT, METRICS_PATH, MODEL_PATH, SPACE, STATIC_DIR
from signlang.landmarks import HandDetector, draw_hand
from signlang.model import GestureClassifier
from signlang.sentence import PredictionSmoother, SentenceBuilder
from signlang.tts import TTSError, synthesize

st.set_page_config(page_title="SignLang AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }

.navbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem; }
.nav-logo { font-size:1.45rem; font-weight:800; color:#1a1f36; letter-spacing:-0.5px; }
.nav-logo span { color:#5b5bd6; }
.nav-badge { background:linear-gradient(135deg,#5b5bd6,#7c6af7); color:#fff; border-radius:20px;
             padding:0.3rem 1rem; font-size:0.72rem; font-weight:600; letter-spacing:0.5px; }

.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.2rem; }
@media (max-width: 800px) { .stats-row { grid-template-columns:repeat(2,1fr); } }
.stat-card { background:#fff; border-radius:14px; padding:1rem 1.2rem; display:flex; align-items:center;
             gap:0.9rem; box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid #eef0f6; }
.stat-icon { width:42px; height:42px; border-radius:12px; display:flex; align-items:center;
             justify-content:center; font-size:1.15rem; flex-shrink:0; }
.i-purple{background:#ede9fe} .i-green{background:#dcfce7} .i-orange{background:#ffedd5} .i-blue{background:#dbeafe}
.stat-val { font-size:1.4rem; font-weight:800; color:#1a1f36; line-height:1; }
.stat-lbl { font-size:0.66rem; font-weight:600; color:#9ca3af; text-transform:uppercase; letter-spacing:1px; margin-top:0.25rem; }

.panel { background:#fff; border:1px solid #eef0f6; border-radius:16px; padding:1.1rem 1.3rem;
         box-shadow:0 1px 3px rgba(0,0,0,0.06); margin-bottom:0.9rem; }
.k { font-size:0.66rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#9ca3af;
     display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; }
.gesture-text { font-size:2.8rem; font-weight:800; color:#1a1f36; line-height:1.1; letter-spacing:-1px; }
.gesture-empty { color:#d1d5db; }
.sub { font-size:0.78rem; color:#9ca3af; font-weight:500; margin:0.15rem 0 0.8rem; }
.pct { font-size:0.9rem; font-weight:700; color:#16a34a; letter-spacing:0; }
.track { background:#f0f2f8; border-radius:6px; height:8px; overflow:hidden; margin-bottom:0.8rem; }
.fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#5b5bd6,#22c55e); }
.fill-hold { height:100%; border-radius:6px; background:#f59e0b; }
.sent-box { background:#f8f9fc; border:1px solid #e8eaf2; border-radius:10px; padding:0.9rem 1rem;
            font-size:1.35rem; font-weight:700; color:#1a1f36; min-height:3.4rem; word-break:break-word;
            white-space:pre-wrap; }
.sent-placeholder { color:#c4c9d4; font-size:0.9rem; font-weight:400; }
.cursor { display:inline-block; width:2px; height:1.2em; background:#5b5bd6; vertical-align:text-bottom;
          animation:blink 1s step-end infinite; margin-left:1px; }
@keyframes blink { 50% { opacity:0; } }
.chip { display:inline-block; background:#f5f3ff; border:1px solid #ddd6fe; color:#5b5bd6; border-radius:8px;
        padding:0.15rem 0.5rem; font-size:0.72rem; font-weight:600; margin:0 0.3rem 0.35rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Resources ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return GestureClassifier.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_resource
def load_image_detector():
    return HandDetector(max_hands=2, static_image=True, min_detection_confidence=0.5)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@st.cache_data(show_spinner=False, max_entries=64)
def speak(text: str, engine: str):
    return synthesize(text, engine=engine)


classifier = load_classifier()
metrics = load_metrics()
if classifier is None:
    st.error(f"No trained model found at `{MODEL_PATH}`. Run `python train.py` first.")
    st.stop()


# ── Live engine (runs in streamlit-webrtc's worker thread) ───────────────────
class LiveEngine:
    """Per-session state shared between the video thread and the Streamlit script."""

    def __init__(self, classifier: GestureClassifier):
        self.classifier = classifier
        self.lock = threading.Lock()
        self.builder = SentenceBuilder()
        self.settings = {"threshold": 0.6, "window": 8, "mirror": True, "show_skeleton": True}
        self._detector = None
        self._smoother = PredictionSmoother(window=8)
        self._fps, self._last_t = 0.0, None
        self.snapshot = {"label": None, "conf": 0.0, "raw": None, "raw_conf": 0.0,
                         "hand": None, "progress": 0.0, "fps": 0.0, "text": "", "n_chunks": 0}

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        s = dict(self.settings)
        if s["mirror"]:
            img = np.ascontiguousarray(img[:, ::-1])
        if self._detector is None:  # MediaPipe must be created in this thread
            self._detector = HandDetector(max_hands=2)
        if self._smoother.window != s["window"]:
            self._smoother = PredictionSmoother(window=s["window"])

        hands = self._detector.detect(img)
        preds = [(h, *self.classifier.predict(h.features)) for h in hands]
        best = max(preds, key=lambda p: p[2]) if preds else None
        raw, raw_conf = (best[1], best[2]) if best else (None, 0.0)
        vote = raw if raw_conf >= s["threshold"] else None
        stable, conf = self._smoother.update(vote, raw_conf)

        with self.lock:
            self.builder.update(stable)
            progress = self.builder.progress() if stable else 0.0
            now = time.monotonic()
            if self._last_t is not None:
                self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_t, 1e-3)
            self._last_t = now
            self.snapshot = {"label": stable, "conf": conf, "raw": raw, "raw_conf": raw_conf,
                             "hand": best[0].handedness if best else None, "progress": progress,
                             "fps": self._fps, "text": self.builder.text,
                             "n_chunks": len(self.builder.history)}

        if s["show_skeleton"]:
            for h, lbl, c in preds:
                draw_hand(img, h, f"{lbl} {c:.0%}" if c >= s["threshold"] else None)
        hgt, wid = img.shape[:2]
        if progress > 0:  # hold-to-add bar along the bottom
            cv2.rectangle(img, (0, hgt - 8), (int(wid * progress), hgt), (245, 158, 11), -1)
        cv2.rectangle(img, (8, 8), (92, 32), (0, 0, 0), -1)
        cv2.putText(img, f"FPS {self._fps:4.1f}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="rgb24")

    def get(self):
        with self.lock:
            h = self.builder.history
            return dict(self.snapshot, text=self.builder.text, n_chunks=len(h),
                        last_chunk=h[-1] if h else "")

    def edit(self, action: str):
        with self.lock:
            b = self.builder
            {"space": b.add_space, "backspace": b.backspace, "undo": b.undo, "clear": b.clear}[action]()

    def add_current(self):
        with self.lock:
            lbl = self.snapshot["label"]
            if lbl:
                self.builder.add_sign(lbl)


if "engine" not in st.session_state:
    st.session_state.engine = LiveEngine(classifier)
engine: LiveEngine = st.session_state.engine


# ── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    engine.settings["threshold"] = st.slider(
        "Minimum confidence", 0.30, 0.95, 0.60, 0.05,
        help="Predictions below this are ignored.")
    engine.settings["window"] = st.slider(
        "Smoothing window (frames)", 1, 20, 8,
        help="Majority vote over this many frames. Higher = steadier but slower to react.")
    engine.builder.hold_seconds = st.slider(
        "Hold time to add a sign (s)", 0.3, 3.0, 1.0, 0.1,
        help="How long a sign must be held steady before it's added to the sentence.")
    engine.settings["mirror"] = st.toggle("Mirror camera (selfie view)", True)
    engine.settings["show_skeleton"] = st.toggle("Draw hand skeleton", True)
    st.markdown("### 🔊 Speech")
    tts_engine = st.radio("Voice", ["auto", "gtts", "offline"], horizontal=True,
                          format_func={"auto": "Auto", "gtts": "Google", "offline": "Offline"}.get,
                          help="Google (gTTS) needs internet. Offline uses your OS voice via pyttsx3.")
    auto_speak = st.toggle("Speak each word as it's added", False)
    st.markdown("### 🧾 Sign → text")
    st.markdown("".join(
        f'<span class="chip">{html.escape(k)} → {"␣ space" if v == SPACE else html.escape(v)}</span>'
        for k, v in GESTURE_TEXT.items() if k in classifier.labels), unsafe_allow_html=True)
    st.caption("Edit `GESTURE_TEXT` in `signlang/config.py` to change these.")


# ── Header + stats ───────────────────────────────────────────────────────────
st.markdown("""<div class="navbar"><div class="nav-logo">Sign<span>Lang</span> AI</div>
<div class="nav-badge">🤟 REAL-TIME SIGN → TEXT → SPEECH</div></div>""", unsafe_allow_html=True)


def stat(icon, cls, val, lbl):
    return (f'<div class="stat-card"><div class="stat-icon {cls}">{icon}</div>'
            f'<div><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div></div>')


acc = f"{metrics['test_accuracy']*100:.0f}%" if metrics else "—"
left = f"{metrics['test_accuracy_mirrored']*100:.0f}%" if metrics else "—"
n_samples = sum(metrics["class_counts"].values()) if metrics else "—"
st.markdown('<div class="stats-row">' +
            stat("🎯", "i-purple", acc, "Test accuracy") +
            stat("🫲", "i-green", left, "Left-hand accuracy") +
            stat("🗂️", "i-orange", f"{n_samples:,}" if metrics else "—", "Training samples") +
            stat("🤟", "i-blue", len(classifier.labels), "Gestures") +
            "</div>", unsafe_allow_html=True)


# ── Render helpers ───────────────────────────────────────────────────────────
def gesture_html(snap):
    if not snap or not snap["label"]:
        raw = snap and snap["raw"]
        hint = (f"Seeing <b>{html.escape(raw)}</b> ({snap['raw_conf']:.0%}) — hold steady…" if raw
                else "Show your hand to the camera")
        return (f'<div class="panel"><div class="k"><span>Gesture</span></div>'
                f'<div class="gesture-text gesture-empty">—</div><div class="sub">{hint}</div>'
                f'<div class="k"><span>Confidence</span><span class="pct">—</span></div>'
                f'<div class="track"><div class="fill" style="width:0%"></div></div></div>')
    pct = int(snap["conf"] * 100)
    hold = int(snap["progress"] * 100)
    token = GESTURE_TEXT.get(snap["label"], snap["label"])
    adds = "space" if token == SPACE else f"“{html.escape(token)}”"
    return (f'<div class="panel"><div class="k"><span>Gesture · {html.escape((snap["hand"] or "").upper())} hand</span></div>'
            f'<div class="gesture-text">{html.escape(snap["label"])}</div>'
            f'<div class="sub">adds {adds} · {snap["fps"]:.0f} fps</div>'
            f'<div class="k"><span>Confidence</span><span class="pct">{pct}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct}%"></div></div>'
            f'<div class="k"><span>Hold to add</span><span class="pct" style="color:#d97706">{hold}%</span></div>'
            f'<div class="track"><div class="fill-hold" style="width:{hold}%"></div></div></div>')


def sentence_html(text):
    body = html.escape(text) if text else '<span class="sent-placeholder">Hold a sign steady to start typing…</span>'
    return f'<div class="sent-box">{body}<span class="cursor"></span></div>'


def play(text, ph):
    try:
        audio, mime = speak(text, tts_engine)
        ph.audio(audio, format=mime, autoplay=True)
    except TTSError as e:
        ph.warning(str(e))


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_upload, tab_model, tab_about = st.tabs(
    ["📷  Live", "🖼️  Upload", "📊  Model", "ℹ️  About"])

with tab_live:
    cam_col, side_col = st.columns([3, 2], gap="medium")
    with cam_col:
        ctx = webrtc_streamer(
            key="signlang-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=engine,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                                      "audio": False},
            async_processing=True,
        )
        st.caption("Click **START** and allow camera access. Hold a sign for the hold time to add it; "
                   "show **Fist** to insert a space. To repeat a sign, lower your hand briefly.")

    with side_col:
        gesture_ph = st.empty()
        st.markdown('<div class="k" style="margin-top:0.2rem"><span>💬 Sentence</span></div>',
                    unsafe_allow_html=True)
        sentence_ph = st.empty()
        b1, b2, b3 = st.columns(3)
        b4, b5, b6 = st.columns(3)
        if b1.button("➕ Add sign", width="stretch", help="Add the current sign right now"):
            engine.add_current()
        if b2.button("␣ Space", width="stretch"):
            engine.edit("space")
        if b3.button("⌫ Delete", width="stretch", help="Delete the last character"):
            engine.edit("backspace")
        if b4.button("↶ Undo", width="stretch", help="Remove the last added sign"):
            engine.edit("undo")
        if b5.button("🗑 Clear", width="stretch"):
            engine.edit("clear")
        speak_clicked = b6.button("🔊 Speak", width="stretch", type="primary")
        audio_ph = st.empty()

        snap = engine.get()
        gesture_ph.markdown(gesture_html(snap if ctx.state.playing else None), unsafe_allow_html=True)
        sentence_ph.markdown(sentence_html(snap["text"]), unsafe_allow_html=True)
        if speak_clicked:
            play(snap["text"], audio_ph)
        if snap["text"]:
            st.download_button("⬇ Save text", snap["text"], file_name="signlang.txt", width="stretch")

    # Live refresh: keep updating the side panel while the stream runs.
    # Any button click interrupts this loop with a rerun, which is what we want.
    last_chunks = snap["n_chunks"]
    while ctx.state.playing:
        snap = engine.get()
        gesture_ph.markdown(gesture_html(snap), unsafe_allow_html=True)
        sentence_ph.markdown(sentence_html(snap["text"]), unsafe_allow_html=True)
        if auto_speak and snap["n_chunks"] > last_chunks:
            new = snap["last_chunk"].strip()
            if len(new) > 1:  # speak words/phrases, not single letters
                play(new, audio_ph)
        last_chunks = snap["n_chunks"]
        time.sleep(0.12)

with tab_upload:
    up = st.file_uploader("Upload a photo of a hand sign (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if up:
        img = np.array(Image.open(up).convert("RGB"))
        with st.spinner("Detecting hands…"):
            hands = load_image_detector().detect(img)
        annotated = img.copy()
        results = []
        for h in hands:
            probs = classifier.predict_proba(h.features)[0]
            top = np.argsort(probs)[::-1][:3]
            results.append((h, [(classifier.labels[i], float(probs[i])) for i in top]))
            draw_hand(annotated, h, classifier.labels[top[0]])
        lcol, rcol = st.columns([3, 2], gap="medium")
        with lcol:
            t1, t2 = st.tabs(["With landmarks", "Original"])
            t1.image(annotated, width="stretch")
            t2.image(img, width="stretch")
        with rcol:
            if not results:
                st.info("No hand detected — try a clearer photo with good lighting and the whole hand visible.")
            for h, top3 in results:
                lbl, c = top3[0]
                rows = "".join(
                    f'<div class="k" style="margin-top:0.4rem"><span>{html.escape(l)}</span>'
                    f'<span class="pct">{p:.0%}</span></div>'
                    f'<div class="track"><div class="fill" style="width:{p*100:.0f}%"></div></div>'
                    for l, p in top3)
                st.markdown(f'<div class="panel"><div class="k"><span>{h.handedness.upper()} hand</span></div>'
                            f'<div class="gesture-text">{html.escape(lbl)}</div>'
                            f'<div class="sub">Top 3 predictions</div>{rows}</div>', unsafe_allow_html=True)

with tab_model:
    if not metrics:
        st.info("No metrics yet — run `python train.py` to generate them.")
    else:
        import pandas as pd

        st.markdown("Numbers below are on a **held-out test set** the model never saw during training "
                    "or model selection. *Left-hand accuracy* re-runs the same test set with every hand "
                    "mirrored.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']*100:.1f}%")
        m2.metric("Macro-F1", f"{metrics['test_macro_f1']*100:.1f}%",
                  help="Average F1 across classes, so every class counts equally regardless of size.")
        m3.metric("Left-hand accuracy", f"{metrics['test_accuracy_mirrored']*100:.1f}%")
        m4.metric("Train / val / test", "{train} / {val} / {test}".format(**metrics["split"]))

        counts = metrics["class_counts"]
        df = pd.DataFrame([
            {"Gesture": l, "Samples": counts[l],
             "Precision": metrics["per_class"][l]["precision"],
             "Recall": metrics["per_class"][l]["recall"],
             "F1": metrics["per_class"][l]["f1-score"],
             "Test samples": int(metrics["per_class"][l]["support"])}
            for l in metrics["labels"]]).sort_values("Samples")
        low = df[df["Samples"] < 50]["Gesture"].tolist()
        if low:
            st.warning(f"Low on data: **{', '.join(low)}**. Collect more with "
                       f"`python collect_data.py --fill 100` and retrain — results for these "
                       f"classes are based on very few test samples.")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("##### Samples per gesture")
            st.bar_chart(df.set_index("Gesture")["Samples"], color="#5b5bd6", horizontal=True, height=380)
        with c2:
            st.markdown("##### Per-class test results")
            st.dataframe(df.sort_values("F1"), hide_index=True, width="stretch", height=380,
                         column_config={k: st.column_config.ProgressColumn(k, format="%.2f", min_value=0, max_value=1)
                                        for k in ("Precision", "Recall", "F1")})
        c3, c4 = st.columns([1, 1], gap="large")
        if (STATIC_DIR / "confusion_matrix.png").exists():
            c3.image(str(STATIC_DIR / "confusion_matrix.png"), width="stretch")
        if (STATIC_DIR / "training_curves.png").exists():
            c4.image(str(STATIC_DIR / "training_curves.png"), width="stretch")
        st.caption(f"Trained with class-weighted loss: {metrics['class_weights']} · augmentation: "
                   f"{metrics['augment']} (strength {metrics.get('aug_strength', 1.0)}) · best epoch "
                   f"{metrics['best_epoch']}/{metrics['epochs']}")

with tab_about:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🏗️ Pipeline")
        st.code("""Browser camera ── WebRTC ──► server
→ MediaPipe Hands: 21 landmarks (x, y, z)
→ normalize: wrist origin, scale to [-1, 1]
→ GestureNet MLP (PyTorch)
     63 → 256 → 128 → 64 → classes
→ confidence threshold
→ majority vote over last N frames
→ hold-to-commit sentence builder
→ text-to-speech (gTTS / pyttsx3)""", language="text")
    with c2:
        st.markdown("#### 🧠 Training")
        st.markdown("""
- Stratified **70 / 15 / 15** train / val / test split
- **Class-weighted** cross-entropy for imbalanced classes
- **Augmentation**: left/right mirroring, rotation, stretch, landmark jitter
- Checkpoint chosen by validation **macro-F1**
- Per-class report + confusion matrix on the held-out test set
""")

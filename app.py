"""
SignLang AI — real-time sign recognition with a sentence builder and text-to-speech.

    streamlit run app.py

The webcam runs through streamlit-webrtc, so the camera is the *viewer's* camera
(works when deployed, not just on localhost).
"""

import base64
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

st.set_page_config(page_title="SignLang AI", page_icon="🤟", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {
  --paper: #F5F6F8; --surface: #FFFFFF; --ink: #16181D; --muted: #676C7A;
  --line: #DDE0E6; --accent: #3F44C8; --hold: #D9822B;
}
#MainMenu, footer, [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* header */
.brand { margin: 0; font-size: 1.9rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; line-height: 1.1; }
.tagline { margin: 0.25rem 0 0; color: var(--muted); font-size: 1rem; }

/* prediction: plain text, no card */
.pred { padding: 0.2rem 0 1.4rem; }
.pred-label { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.15rem; }
.pred-sign { font-size: 2.6rem; font-weight: 700; color: var(--ink); line-height: 1.05; letter-spacing: -0.02em; }
.pred-sign.idle { color: #B9BDC7; }
.pred-row { display: flex; align-items: center; gap: 0.75rem; margin-top: 0.55rem; }
.pred-pct { font-weight: 700; color: var(--ink); min-width: 3.2rem; font-variant-numeric: tabular-nums; }
.bar { flex: 1; height: 6px; background: #E8EAEE; border-radius: 3px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
.bar.hold > i { background: var(--hold); }
.pred-hint { color: var(--muted); font-size: 0.88rem; margin-top: 0.5rem; }

/* sentence: the one bordered surface, the boldest type */
.sentence { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
            padding: 1rem 1.15rem; min-height: 6.2rem; font-size: 1.7rem; line-height: 1.35;
            font-weight: 700; color: var(--ink); white-space: pre-wrap; word-break: break-word;
            margin-bottom: 0.9rem; }
.sentence .empty { color: #A4A9B4; font-weight: 400; font-size: 1.05rem; }
.cursor { display: inline-block; width: 2px; height: 1.15em; background: var(--accent);
          vertical-align: text-bottom; margin-left: 2px; animation: blink 1.1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .cursor { animation: none; } }

/* comfortable buttons */
.stButton button { min-height: 2.9rem; font-weight: 700; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }

/* camera is the focus */
.camera-off { aspect-ratio: 4 / 3; background: #E9EBEF; border-radius: 14px; display: flex;
              flex-direction: column; align-items: center; justify-content: center; gap: 0.35rem;
              color: var(--muted); text-align: center; padding: 1rem; }
.camera-off strong { color: var(--ink); font-size: 1.15rem; }
.camera-note { color: var(--muted); font-size: 0.92rem; margin-top: 0.6rem; }

/* sign vocabulary list */
.vocab { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.35rem 1.2rem; }
.vocab div { padding: 0.35rem 0; border-bottom: 1px solid #ECEEF2; color: var(--ink); }
.vocab span { color: var(--muted); }

@media (max-width: 640px) {
  .block-container { padding-top: 1.2rem; }
  .st-key-header [data-testid="stHorizontalBlock"],
  .st-key-controls [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: 0.5rem; }
  .st-key-header [data-testid="stColumn"],
  .st-key-controls [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
  .st-key-header [data-testid="stColumn"]:last-child { flex: 0 0 auto !important; }
  .brand { font-size: 1.6rem; }
  .pred-sign { font-size: 2.1rem; }
  .sentence { font-size: 1.35rem; min-height: 4.5rem; }
}
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


# ── Header + settings ────────────────────────────────────────────────────────
head_l, head_r = st.container(key="header").columns([5, 1], vertical_alignment="bottom")
head_l.markdown('<h1 class="brand">SignLang AI</h1><p class="tagline">Sign → Text → Speech</p>',
                unsafe_allow_html=True)
with head_r.popover("Settings", width="stretch"):
    st.markdown("**Recognition**")
    engine.settings["threshold"] = st.slider(
        "Minimum confidence", 0.30, 0.95, 0.60, 0.05,
        help="Predictions below this are ignored.")
    engine.settings["window"] = st.slider(
        "Smoothing window (frames)", 1, 20, 8,
        help="Majority vote over this many frames. Higher = steadier but slower to react.")
    engine.builder.hold_seconds = st.slider(
        "Hold time to add a sign (s)", 0.3, 3.0, 1.0, 0.1,
        help="How long a sign must be held steady before it's added to the sentence.")
    st.markdown("**Camera**")
    engine.settings["mirror"] = st.toggle("Mirror camera (selfie view)", True)
    engine.settings["show_skeleton"] = st.toggle("Draw hand skeleton", True)
    st.markdown("**Voice**")
    tts_engine = st.radio("Voice", ["auto", "gtts", "offline"], horizontal=True,
                          label_visibility="collapsed",
                          format_func={"auto": "Auto", "gtts": "Google", "offline": "Offline"}.get,
                          help="Google (gTTS) needs internet. Offline uses your OS voice via pyttsx3.")
    st.caption("Auto uses Google's voice when online and your computer's voice otherwise.")

st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)


# ── Render helpers ───────────────────────────────────────────────────────────
def prediction_html(snap, playing):
    """Current sign + confidence. Plain text, no model metrics."""
    if not playing:
        return ('<div class="pred"><div class="pred-label">Current sign</div>'
                '<div class="pred-sign idle">—</div>'
                '<div class="pred-hint">Start the camera to begin.</div></div>')
    if not snap["label"]:
        hint = (f"Seeing {html.escape(snap['raw'])}, hold it steady…" if snap["raw"]
                else "Show your hand to the camera.")
        return ('<div class="pred"><div class="pred-label">Current sign</div>'
                f'<div class="pred-sign idle">—</div><div class="pred-hint">{hint}</div></div>')
    pct = int(snap["conf"] * 100)
    hold = int(snap["progress"] * 100)
    token = GESTURE_TEXT.get(snap["label"], snap["label"])
    adds = "a space" if token == SPACE else f"“{html.escape(token)}”"
    return ('<div class="pred"><div class="pred-label">Current sign</div>'
            f'<div class="pred-sign">{html.escape(snap["label"])}</div>'
            f'<div class="pred-row"><span class="pred-pct">{pct}%</span>'
            f'<div class="bar" role="meter" aria-label="Confidence" aria-valuenow="{pct}">'
            f'<i style="width:{pct}%"></i></div></div>'
            f'<div class="pred-hint">Hold steady to add {adds}</div>'
            f'<div class="pred-row"><div class="bar hold" role="meter" aria-label="Hold to add" '
            f'aria-valuenow="{hold}"><i style="width:{hold}%"></i></div></div></div>')


def sentence_html(text):
    body = html.escape(text) if text else '<span class="empty">Your sentence will appear here.</span>'
    return f'<div class="sentence" aria-live="polite">{body}<span class="cursor"></span></div>'


def play(text, ph):
    try:
        audio, mime = speak(text, tts_engine)
    except TTSError as e:
        ph.warning(str(e))
        return
    # Not st.audio: it takes no `key` and derives its element ID from the audio bytes,
    # so speaking the same phrase twice in one script run (auto-speak during a live
    # stream, or Speak + auto-speak) raises StreamlitDuplicateElementId. A plain
    # <audio> tag has no element ID; the counter makes every call a fresh element,
    # so repeats replay instead of being treated as unchanged.
    st.session_state.tts_count = st.session_state.get("tts_count", 0) + 1
    b64 = base64.b64encode(audio).decode()
    ph.html(f'<audio data-tts="{st.session_state.tts_count}" controls autoplay '
            f'style="width:100%" src="data:{mime};base64,{b64}"></audio>')


# ── Main screen: camera, current sign, sentence ──────────────────────────────
cam_col, side_col = st.columns([3, 2], gap="large")
with cam_col:
    cam_off_ph = st.empty()
    ctx = webrtc_streamer(
        key="signlang-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=engine,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                                  "audio": False},
        async_processing=True,
        translations={"start": "Start camera", "stop": "Stop camera", "select_device": "Switch camera"},
    )
    if not ctx.state.playing:
        cam_off_ph.markdown('<div class="camera-off"><strong>Camera is off</strong>'
                            'Press Start camera below and allow access.</div>', unsafe_allow_html=True)
    st.markdown('<p class="camera-note">Hold a sign steady to add it. Make a fist for a space. '
                'To repeat a sign, lower your hand for a moment.</p>', unsafe_allow_html=True)

with side_col:
    pred_ph = st.empty()
    sentence_ph = st.empty()
    controls = st.container(key="controls")
    b1, b2, b3 = controls.columns(3)
    b4, b5, b6 = controls.columns(3)
    if b1.button("Add sign", width="stretch", help="Add the current sign right now"):
        engine.add_current()
    if b2.button("Space", width="stretch"):
        engine.edit("space")
    if b3.button("Delete", width="stretch", help="Delete the last character"):
        engine.edit("backspace")
    if b4.button("Undo", width="stretch", help="Remove the last added sign"):
        engine.edit("undo")
    if b5.button("Clear", width="stretch"):
        engine.edit("clear")
    speak_clicked = b6.button("🔊 Speak", width="stretch", type="primary")
    auto_speak = st.toggle("Speak each word as it's added", False,
                           help="Auto speak: reads each new word aloud as soon as it's added.")
    audio_ph = st.empty()
    save_ph = st.empty()

    snap = engine.get()
    pred_ph.markdown(prediction_html(snap, ctx.state.playing), unsafe_allow_html=True)
    sentence_ph.markdown(sentence_html(snap["text"]), unsafe_allow_html=True)
    if speak_clicked:
        play(snap["text"], audio_ph)
    if snap["text"]:
        save_ph.download_button("Save as text", snap["text"], file_name="signlang.txt",
                                type="tertiary", on_click="ignore")


# ── Everything else, tucked away ─────────────────────────────────────────────
st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)

with st.expander("Recognize a photo"):
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
        lcol, rcol = st.columns([3, 2], gap="large")
        with lcol:
            t1, t2 = st.tabs(["With landmarks", "Original"])
            t1.image(annotated, width="stretch")
            t2.image(img, width="stretch")
        with rcol:
            if not results:
                st.info("No hand found. Try a well-lit photo with the whole hand in view.")
            for h, top3 in results:
                lbl, _ = top3[0]
                rows = "".join(
                    f'<div class="pred-row"><span class="pred-pct">{p:.0%}</span>'
                    f'<div class="bar"><i style="width:{p*100:.0f}%"></i></div>'
                    f'<span style="min-width:5.5rem">{html.escape(l)}</span></div>'
                    for l, p in top3)
                st.markdown(f'<div class="pred"><div class="pred-label">{h.handedness} hand</div>'
                            f'<div class="pred-sign">{html.escape(lbl)}</div>'
                            f'<div class="pred-hint">Top 3 predictions</div>{rows}</div>',
                            unsafe_allow_html=True)

with st.expander("Accuracy and training data"):
    if not metrics:
        st.info("No metrics yet. Run `python train.py` to generate them.")
    else:
        import pandas as pd

        st.markdown("Measured on a **held-out test set** the model never saw during training or "
                    "model selection. *Left-hand accuracy* re-runs that test set with every hand "
                    "mirrored.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']*100:.1f}%")
        m2.metric("Macro-F1", f"{metrics['test_macro_f1']*100:.1f}%",
                  help="Average F1 across classes, so every class counts equally regardless of size.")
        m3.metric("Left-hand accuracy", f"{metrics['test_accuracy_mirrored']*100:.1f}%")
        m4, m5, m6 = st.columns(3)
        m4.metric("Training samples", f"{sum(metrics['class_counts'].values()):,}")
        m5.metric("Gestures", len(classifier.labels))
        m6.metric("Train / val / test", "{train} / {val} / {test}".format(**metrics["split"]))

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
                       f"`python collect_data.py --fill 100` and retrain. Results for these "
                       f"classes are based on very few test samples.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Samples per gesture**")
            st.bar_chart(df.set_index("Gesture")["Samples"], color="#3F44C8", horizontal=True, height=380)
        with c2:
            st.markdown("**Per-class test results**")
            st.dataframe(df.sort_values("F1"), hide_index=True, width="stretch", height=380,
                         column_config={k: st.column_config.ProgressColumn(k, format="%.2f", min_value=0, max_value=1)
                                        for k in ("Precision", "Recall", "F1")})
        c3, c4 = st.columns(2, gap="large")
        if (STATIC_DIR / "confusion_matrix.png").exists():
            c3.image(str(STATIC_DIR / "confusion_matrix.png"), width="stretch")
        if (STATIC_DIR / "training_curves.png").exists():
            c4.image(str(STATIC_DIR / "training_curves.png"), width="stretch")
        st.caption(f"Trained with class-weighted loss: {metrics['class_weights']}. Augmentation: "
                   f"{metrics['augment']} (strength {metrics.get('aug_strength', 1.0)}). Best epoch "
                   f"{metrics['best_epoch']} of {metrics['epochs']}.")

with st.expander("Sign vocabulary"):
    st.markdown("What each sign adds to your sentence.")
    st.markdown('<div class="vocab">' + "".join(
        f'<div>{html.escape(k)} <span>→ {"space" if v == SPACE else html.escape(v)}</span></div>'
        for k, v in GESTURE_TEXT.items() if k in classifier.labels) + "</div>", unsafe_allow_html=True)
    st.caption("To change these, edit `GESTURE_TEXT` in `signlang/config.py`.")

with st.expander("How it works and training your own model"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Pipeline**")
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
        st.markdown("**Training**")
        st.markdown("""
- Stratified 70 / 15 / 15 train / val / test split
- Class-weighted cross-entropy for imbalanced classes
- Augmentation: left/right mirroring, rotation, stretch, landmark jitter
- Checkpoint chosen by validation macro-F1
- Per-class report and confusion matrix on the held-out test set
""")
        st.markdown("**Add data and retrain** (run in a terminal, then restart the app)")
        st.code("# top up every gesture to 100 samples\npython collect_data.py --fill 100\n\n"
                "# retrain and refresh these metrics\npython train.py", language="bash")


# ── Live refresh ─────────────────────────────────────────────────────────────
# Kept last: this loop blocks the script while the camera runs, so everything above
# must already be rendered. Any button click interrupts it with a rerun.
last_chunks = snap["n_chunks"]
while ctx.state.playing:
    snap = engine.get()
    pred_ph.markdown(prediction_html(snap, True), unsafe_allow_html=True)
    sentence_ph.markdown(sentence_html(snap["text"]), unsafe_allow_html=True)
    if auto_speak and snap["n_chunks"] > last_chunks:
        new = snap["last_chunk"].strip()
        if len(new) > 1:  # speak words/phrases, not single letters
            play(new, audio_ph)
    last_chunks = snap["n_chunks"]
    time.sleep(0.12)

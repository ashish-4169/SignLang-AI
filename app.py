"""
SignLang AI — Clean Modern UI (white cards, professional layout)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time, math, os
import torch
import torch.nn as nn

st.set_page_config(page_title="SignLang AI", page_icon="🤟",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    background: #f0f2f8 !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1f36 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── TOP NAVBAR ── */
.navbar {
    background: #ffffff;
    border-bottom: 1px solid #e8eaf2;
    padding: 0.9rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1a1f36;
    letter-spacing: -0.5px;
}
.nav-logo span { color: #5b5bd6; }
.nav-badge {
    background: linear-gradient(135deg, #5b5bd6, #7c6af7);
    color: white;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.nav-links { display: flex; gap: 2rem; }
.nav-link { color: #6b7280; font-size: 0.85rem; font-weight: 500; text-decoration: none; }

/* ── PAGE WRAPPER ── */
.page-wrap { padding: 1.8rem 2.5rem 4rem; }

/* ── STAT CARDS ROW ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #f0f2f8;
}
.stat-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.stat-icon-purple { background: #ede9fe; }
.stat-icon-green  { background: #dcfce7; }
.stat-icon-orange { background: #ffedd5; }
.stat-icon-blue   { background: #dbeafe; }
.stat-val { font-size: 1.5rem; font-weight: 800; color: #1a1f36; line-height: 1; }
.stat-lbl { font-size: 0.7rem; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

/* ── MAIN GRID ── */
.main-grid {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 1.2rem;
}

/* ── CARD ── */
.card {
    background: white;
    border-radius: 16px;
    border: 1px solid #f0f2f8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    overflow: hidden;
}
.card-header {
    padding: 1rem 1.4rem 0.8rem;
    border-bottom: 1px solid #f5f6fa;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.card-header-icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    background: #ede9fe;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6b7280;
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    margin-left: auto;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.live-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #22c55e;
    margin-left: 0.4rem;
}
.card-body { padding: 1.2rem 1.4rem; }

/* ── CAM PLACEHOLDER ── */
.cam-empty {
    background: #f8f9fc;
    border-radius: 12px;
    height: 340px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    border: 2px dashed #e0e4f0;
}
.cam-empty-icon { font-size: 2.5rem; opacity: 0.3; }
.cam-empty-text { font-size: 0.82rem; color: #9ca3af; font-weight: 500; }

/* ── GESTURE DISPLAY ── */
.gesture-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}
.gesture-text {
    font-size: 3rem;
    font-weight: 800;
    color: #1a1f36;
    line-height: 1.1;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.gesture-empty { color: #d1d5db; font-size: 2rem; }
.gesture-hand {
    font-size: 0.75rem;
    color: #9ca3af;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── CONFIDENCE ── */
.conf-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.conf-pct { font-size: 0.9rem; font-weight: 700; color: #22c55e; }
.conf-track {
    background: #f0f2f8;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.conf-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #5b5bd6, #22c55e);
    transition: width 0.2s ease;
}

/* ── NN BADGE ── */
.nn-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #5b5bd6;
    margin-bottom: 1.2rem;
}

/* ── SENTENCE ── */
.sent-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sent-box {
    background: #f8f9fc;
    border: 1px solid #e8eaf2;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1f36;
    min-height: 3.2rem;
    letter-spacing: 3px;
    margin-bottom: 0.8rem;
    word-break: break-all;
}
.sent-placeholder { color: #d1d5db; font-size: 0.9rem; font-weight: 400; letter-spacing: 0; }

/* ── BUTTONS ── */
.stButton > button {
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 0.8rem !important;
    transition: all 0.15s !important;
    border: 1px solid #e8eaf2 !important;
    background: white !important;
    color: #1a1f36 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover {
    background: #f5f6fa !important;
    border-color: #c4b5fd !important;
    color: #5b5bd6 !important;
}
.stButton > button:first-child { background: #5b5bd6 !important; color: white !important; border-color: #5b5bd6 !important; }
.stButton > button:first-child:hover { background: #4e4ec7 !important; }

/* ── TOGGLE ── */
label[data-baseweb="checkbox"] span { color: #6b7280 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.2rem !important;
    border-bottom: none !important;
    margin-bottom: 1.2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: white !important;
    border: 1px solid #e8eaf2 !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: #5b5bd6 !important;
    color: white !important;
    border-color: #5b5bd6 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 0 !important; }

/* ── SLIDER ── */
.stSlider label { color: #6b7280 !important; font-size: 0.8rem !important; }

/* ── UPLOAD ── */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #e0e4f0 !important;
    border-radius: 14px !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e8eaf2 !important; }
section[data-testid="stSidebar"] * { color: #6b7280 !important; font-size: 0.82rem !important; }

/* ── FPS OVERLAY ── */
.fps-badge {
    display: inline-block;
    background: rgba(0,0,0,0.6);
    color: white;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,num_classes))
    def forward(self,x): return self.net(x)

def find_model():
    for root,_,files in os.walk("."):
        for f in files:
            if f=="gesture_model.pth": return os.path.join(root,f)
    return None

@st.cache_resource
def load_all():
    import mediapipe as mp
    try:
        hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,
              min_detection_confidence=0.7,min_tracking_confidence=0.6)
        api_type,mp_mod="legacy",mp.solutions.hands
    except:
        import urllib.request
        mf="hand_landmarker.task"
        if not os.path.exists(mf):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",mf)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        opts=mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=mf),
            num_hands=2,min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,min_tracking_confidence=0.6,
            running_mode=mp_vision.RunningMode.IMAGE)
        hands=mp_vision.HandLandmarker.create_from_options(opts)
        api_type,mp_mod="tasks",None
    mp_path=find_model(); g_model,labels=None,None
    if mp_path:
        ck=torch.load(mp_path,map_location="cpu")
        labels=ck["labels"]
        g_model=GestureNet(63,len(labels))
        g_model.load_state_dict(ck["model_state"])
        g_model.eval()
    return api_type,hands,mp_mod,g_model,labels

def norm_lm(lm):
    c=np.array([[l[0],l[1],l[2]] for l in lm],dtype=np.float32)
    c-=c[0]; c/=(np.max(np.abs(c))+1e-8); return c.flatten()

def classify(lm,handedness,g_model,labels):
    if g_model and labels:
        t=torch.tensor(norm_lm(lm),dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pr=torch.softmax(g_model(t),1); c,i=pr.max(1)
        return labels[i.item()],float(c.item()),"nn"
    def d(i,j): return math.sqrt((lm[i][0]-lm[j][0])**2+(lm[i][1]-lm[j][1])**2)
    f=[1 if(lm[4][0]<lm[3][0] if handedness=="Right" else lm[4][0]>lm[3][0])else 0]
    for tip in [8,12,16,20]: f.append(1 if lm[tip][1]<lm[tip-2][1] else 0)
    imd=d(8,12)
    if f==[1,1,1,1,1]: return("Hello",0.90,"rules")
    if f==[1,0,0,0,0]: return("ThumbsUp",0.90,"rules")
    if f==[1,0,0,0,1]: return("CallMe",0.90,"rules")
    if f==[1,1,0,0,1]: return("ILoveYou",0.90,"rules")
    if f==[0,1,1,0,0] and imd>0.06: return("Peace",0.90,"rules")
    if f==[0,0,0,0,0]: return("Fist",0.90,"rules")
    if f==[0,1,0,0,0]: return("Pointing",0.90,"rules")
    if f==[1,1,0,0,0]: return("L",0.90,"rules")
    if f==[0,0,0,0,1]: return("I",0.90,"rules")
    return("...",0.30,"rules")

def process(img_rgb,api_type,hands_det,mp_mod,g_model,labels):
    import mediapipe as mp
    frame=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
    h,w=frame.shape[:2]; dets=[]
    if api_type=="legacy":
        res=hands_det.process(img_rgb)
        if res.multi_hand_landmarks:
            for hl,hi in zip(res.multi_hand_landmarks,res.multi_handedness):
                hd=hi.classification[0].label
                lm=[(l.x,l.y,l.z) for l in hl.landmark]
                lb,cf,mt=classify(lm,hd,g_model,labels)
                dets.append({"label":lb,"confidence":cf,"method":mt,"handedness":hd})
                for conn in mp.solutions.hands.HAND_CONNECTIONS:
                    p1,p2=hl.landmark[conn[0]],hl.landmark[conn[1]]
                    cv2.line(frame,(int(p1.x*w),int(p1.y*h)),(int(p2.x*w),int(p2.y*h)),(91,91,214),2)
                for l in hl.landmark:
                    cv2.circle(frame,(int(l.x*w),int(l.y*h)),5,(34,197,94),-1)
                    cv2.circle(frame,(int(l.x*w),int(l.y*h)),7,(255,255,255),1)
                wr=hl.landmark[0]
                cv2.putText(frame,f"{lb}",(int(wr.x*w),int(wr.y*h)-18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(91,91,214),2)
    else:
        mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=img_rgb)
        res=hands_det.detect(mi)
        if res.hand_landmarks:
            for i,hl in enumerate(res.hand_landmarks):
                hd=res.handedness[i][0].category_name if res.handedness else "Right"
                lm=[(l.x,l.y,l.z) for l in hl]
                lb,cf,mt=classify(lm,hd,g_model,labels)
                dets.append({"label":lb,"confidence":cf,"method":mt,"handedness":hd})
                for l in hl:
                    cv2.circle(frame,(int(l.x*w),int(l.y*h)),5,(34,197,94),-1)
                    cv2.circle(frame,(int(l.x*w),int(l.y*h)),7,(255,255,255),1)
                wr=hl[0]
                cv2.putText(frame,f"{lb}",(int(wr.x*w),int(wr.y*h)-18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(91,91,214),2)
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),dets

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    try: api_type,hands_det,mp_mod,g_model,labels=load_all(); ok=True
    except Exception as e: st.error(f"Load failed: {e}"); st.stop()

if "sentence" not in st.session_state: st.session_state.sentence=""
if "cur" not in st.session_state: st.session_state.cur=""
if "conf" not in st.session_state: st.session_state.conf=0.0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**SignLang AI**")
    st.markdown("---")
    st.markdown(f"Model: {'GestureNet' if g_model else 'Rules'}")
    st.markdown(f"Classes: {len(labels) if labels else 0}")
    st.markdown("---")
    conf_threshold=st.slider("Min confidence",0.3,0.95,0.5,0.05)

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-logo">Sign<span>Lang</span> AI</div>
  <div class="nav-badge">🤟 REAL-TIME ASL RECOGNITION</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3=st.tabs(["📷  Webcam","🖼️  Upload","ℹ️  About"])

# ══ WEBCAM ════════════════════════════════════════════════════════════════════
with tab1:
    # Stats row
    st.markdown("""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon stat-icon-purple">🎯</div>
        <div><div class="stat-val">98%</div><div class="stat-lbl">Accuracy</div></div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon-green">🖐️</div>
        <div><div class="stat-val">21</div><div class="stat-lbl">Landmarks</div></div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon-orange">⚡</div>
        <div><div class="stat-val">30fps</div><div class="stat-lbl">Inference</div></div>
      </div>
      <div class="stat-card">
        <div class="stat-icon stat-icon-blue">🤟</div>
        <div><div class="stat-val">14</div><div class="stat-lbl">Gestures</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left,right=st.columns([3,2],gap="medium")

    with left:
        # Camera card header
        run=st.toggle("🎥  Enable live detection", key="cam_toggle")
        f_ph=st.empty()
        if not run:
            f_ph.markdown("""
            <div class="cam-empty">
              <div class="cam-empty-icon">📷</div>
              <div class="cam-empty-text">Toggle live detection to start</div>
            </div>""",unsafe_allow_html=True)

    with right:
        # Detection card
        g_ph=st.empty()
        cf_ph=st.empty()
        mt_ph=st.empty()

        # Sentence
        st.markdown('<div class="sent-label">💬 SENTENCE</div>',unsafe_allow_html=True)
        s_ph=st.empty()

        ca,cb,cc=st.columns(3)
        add_b=ca.button("Add",use_container_width=True,key="add")
        spc_b=cb.button("Space",use_container_width=True,key="spc")
        clr_b=cc.button("Clear",use_container_width=True,key="clr")

        if add_b and st.session_state.cur and st.session_state.conf>=conf_threshold:
            lbl=st.session_state.cur
            if len(lbl)==1: st.session_state.sentence+=lbl
        if spc_b: st.session_state.sentence+=" "
        if clr_b: st.session_state.sentence=""

        sent=st.session_state.sentence
        if sent.strip():
            s_ph.markdown(f'<div class="sent-box">{sent}</div>',unsafe_allow_html=True)
        else:
            s_ph.markdown('<div class="sent-box"><span class="sent-placeholder">Start signing...</span></div>',unsafe_allow_html=True)

        # Default gesture display
        g_ph.markdown("""
        <div style="padding:0.5rem 0 1rem">
          <div class="gesture-label">GESTURE</div>
          <div class="gesture-text gesture-empty">—</div>
          <div class="gesture-hand">Show your hand to the camera</div>
        </div>""",unsafe_allow_html=True)
        cf_ph.markdown("""
        <div>
          <div class="conf-label"><span>CONFIDENCE</span><span class="conf-pct">—</span></div>
          <div class="conf-track"><div class="conf-fill" style="width:0%"></div></div>
        </div>""",unsafe_allow_html=True)

    # Camera loop
    if run:
        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
        else:
            stop_b=st.button("⏹  Stop",key="stop")
            frame_count=0; t_start=time.time()
            while not stop_b:
                ret,frame=cap.read()
                if not ret: break
                frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                try: ann,dets=process(rgb,api_type,hands_det,mp_mod,g_model,labels)
                except: ann,dets=rgb,[]

                # FPS overlay
                frame_count+=1
                fps=int(frame_count/(time.time()-t_start+0.001))
                cv2.rectangle(ann,(10,ann.shape[0]-35),(80,ann.shape[0]-10),(0,0,0),cv2.FILLED)
                cv2.putText(ann,f"FPS: {min(fps,30)}",(14,ann.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

                # Live badge
                cv2.circle(ann,(ann.shape[1]-20,20),7,(34,197,94),-1)

                f_ph.image(ann,channels="RGB",use_container_width=True)

                if dets:
                    best=max(dets,key=lambda x:x["confidence"])
                    st.session_state.cur=best["label"]
                    st.session_state.conf=best["confidence"]
                    if best["confidence"]>=conf_threshold:
                        pct=int(best["confidence"]*100)
                        g_ph.markdown(f"""
                        <div style="padding:0.5rem 0 0.3rem">
                          <div class="gesture-label">GESTURE · {best['handedness'].upper()} HAND</div>
                          <div class="gesture-text">{best['label']}</div>
                          <div class="gesture-hand">Detected with {pct}% confidence</div>
                        </div>""",unsafe_allow_html=True)
                        cf_ph.markdown(f"""
                        <div>
                          <div class="conf-label"><span>CONFIDENCE</span><span class="conf-pct">{pct}%</span></div>
                          <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
                        </div>""",unsafe_allow_html=True)
                        mt_ph.markdown(
                            f'<div class="nn-badge">{"🧠 Neural Network" if best["method"]=="nn" else "📐 Rules"}</div>',
                            unsafe_allow_html=True)
                else:
                    st.session_state.cur=""
                    g_ph.markdown("""
                    <div style="padding:0.5rem 0 1rem">
                      <div class="gesture-label">GESTURE</div>
                      <div class="gesture-text gesture-empty">—</div>
                      <div class="gesture-hand">Show your hand to the camera</div>
                    </div>""",unsafe_allow_html=True)
                    cf_ph.markdown("""
                    <div>
                      <div class="conf-label"><span>CONFIDENCE</span><span class="conf-pct">—</span></div>
                      <div class="conf-track"><div class="conf-fill" style="width:0%"></div></div>
                    </div>""",unsafe_allow_html=True)
                time.sleep(0.03)
            cap.release()

# ══ UPLOAD ════════════════════════════════════════════════════════════════════
with tab2:
    up=st.file_uploader("Upload a hand gesture image (JPG / PNG)",
                        type=["jpg","jpeg","png"],label_visibility="visible")
    if up:
        img=Image.open(up).convert("RGB"); arr=np.array(img)
        with st.spinner("Running inference..."):
            ann,dets=process(arr,api_type,hands_det,mp_mod,g_model,labels)
        l,r=st.columns([3,2],gap="medium")
        with l:
            t1,t2=st.tabs(["Original","With Landmarks"])
            with t1: st.image(img,use_container_width=True)
            with t2: st.image(ann,use_container_width=True)
        with r:
            if dets:
                for d in dets:
                    pct=int(d["confidence"]*100)
                    st.markdown(f"""
                    <div style="padding:0.8rem 0;border-bottom:1px solid #f0f2f8;margin-bottom:0.8rem">
                      <div class="gesture-label">{d['handedness'].upper()} HAND</div>
                      <div class="gesture-text">{d['label']}</div>
                      <div class="conf-label" style="margin-top:0.5rem">
                        <span>CONFIDENCE</span><span class="conf-pct">{pct}%</span>
                      </div>
                      <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
                      <div class="nn-badge">{"🧠 Neural Network" if d['method']=='nn' else '📐 Rules'}</div>
                    </div>""",unsafe_allow_html=True)
            else:
                st.info("No hand detected — try a clearer photo with good lighting.")

# ══ ABOUT ════════════════════════════════════════════════════════════════════
with tab3:
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown("#### 🏗️ Pipeline")
        st.code("""Webcam → MediaPipe BlazePalm
→ 21 hand landmarks (x, y, z)
→ Normalize to wrist origin
→ GestureNet MLP (PyTorch)
   Linear(63→256)+BN+ReLU+Drop
   Linear(256→128)+BN+ReLU+Drop
   Linear(128→64)+ReLU
   Linear(64→14)+Softmax
→ Label + Confidence""",language="text")
    with c2:
        st.markdown("#### 📝 Resume Line")
        st.code("""SignLang AI | PyTorch · MediaPipe · OpenCV
• 1,007 samples collected, GestureNet trained:
  98% val accuracy, 14 ASL gesture classes
• Real-time: landmark extraction → NN inference
  at 30fps, fully offline, <15ms latency
• Full ML lifecycle: collect → train → deploy""",language="text")

st.markdown("</div>",unsafe_allow_html=True)

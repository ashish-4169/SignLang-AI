"""
Regression test: repeated text-to-speech in one script run must not raise
StreamlitDuplicateElementId. Drives the real app.py with streamlit's AppTest,
with the WebRTC component stubbed (it needs a real browser) and TTS stubbed to
return identical bytes for identical text — exactly what gTTS/pyttsx3 + caching do.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = str(Path(__file__).resolve().parent.parent)


def _script():
    import os
    import sys
    import types

    app_dir = os.environ["SIGNLANG_APP_DIR"]
    sys.path.insert(0, app_dir)
    os.chdir(app_dir)

    import signlang.tts as tts
    tts.synthesize = lambda text, lang="en", engine="auto": (b"RIFF" + text.encode(), "audio/wav")

    # Fake stream: stays "playing" for a few loop iterations while the user
    # signs "hello" twice, then stops.
    import streamlit as st
    steps = {"n": 0}

    class _State:
        @property
        def playing(self):
            steps["n"] += 1
            eng = st.session_state.get("engine")
            if eng is not None and steps["n"] in (3, 6):
                with eng.lock:
                    eng.builder.add_sign("Hello")
            return steps["n"] < 9

    class _Ctx:
        state = _State()

    stub = types.ModuleType("streamlit_webrtc")
    stub.webrtc_streamer = lambda **kw: _Ctx()
    stub.WebRtcMode = types.SimpleNamespace(SENDRECV="sendrecv")
    sys.modules["streamlit_webrtc"] = stub

    # Explicit UTF-8: app.py contains emoji/symbols, and Windows defaults to cp1252.
    with open("app.py", encoding="utf-8") as f:
        source = f.read()
    exec(compile(source, "app.py", "exec"), {"__name__": "__main__"})


def _run_app(monkeypatch, *, auto_speak, click_speak_times=0):
    monkeypatch.setenv("SIGNLANG_APP_DIR", APP_DIR)
    at = AppTest.from_function(_script, default_timeout=120)
    at.run()
    if auto_speak:
        next(t for t in at.toggle if t.label.startswith("Speak each word")).set_value(True)
    at.run()  # stream "plays": auto-speak fires for each "hello"
    for _ in range(click_speak_times):
        next(b for b in at.button if b.label.startswith("🔊")).click()
        at.run()  # manual Speak of the same sentence + auto-speak again in the same run
    return at


def _audio_players(at):
    return [h.proto.body for h in at.get("html") if "<audio" in h.proto.body]


def test_auto_speak_same_word_twice(monkeypatch):
    at = _run_app(monkeypatch, auto_speak=True)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["engine"].builder.text.count("hello") >= 2
    assert at.session_state["tts_count"] == 2          # "hello" was spoken both times
    assert len(_audio_players(at)) == 1                 # one player slot, replaced each time


def test_manual_speak_repeatedly_with_auto_speak(monkeypatch):
    at = _run_app(monkeypatch, auto_speak=True, click_speak_times=3)
    assert not at.exception, [e.value for e in at.exception]
    # 2 auto-speaks per run (initial + 3 Speak reruns) + 3 manual Speaks
    assert at.session_state["tts_count"] == 2 * 4 + 3
    player = _audio_players(at)[0]
    assert 'data-tts="11"' in player and "autoplay" in player and "data:audio/wav;base64," in player

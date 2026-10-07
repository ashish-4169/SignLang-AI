"""
Text-to-speech.

Tries gTTS first (natural Google voice, needs internet — works on Streamlit Cloud /
Hugging Face), then pyttsx3 (fully offline, uses the OS voice: SAPI on Windows,
NSSpeechSynthesizer on macOS, eSpeak on Linux).
"""

import io
import os
import tempfile


class TTSError(RuntimeError):
    pass


def _gtts(text: str, lang: str) -> tuple[bytes, str]:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue(), "audio/mp3"


def _pyttsx3(text: str) -> tuple[bytes, str]:
    import pyttsx3
    engine = pyttsx3.init()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
        os.remove(path)
    if not data:
        raise TTSError("pyttsx3 produced no audio")
    return data, "audio/wav"


def synthesize(text: str, lang: str = "en", engine: str = "auto") -> tuple[bytes, str]:
    """Return (audio_bytes, mime_type). engine: 'auto' | 'gtts' | 'offline'."""
    text = text.strip()
    if not text:
        raise TTSError("Nothing to speak yet")
    order = {"auto": [lambda: _gtts(text, lang), lambda: _pyttsx3(text)],
             "gtts": [lambda: _gtts(text, lang)],
             "offline": [lambda: _pyttsx3(text)]}[engine]
    errors = []
    for attempt in order:
        try:
            return attempt()
        except Exception as e:  # missing package, no internet, no OS voice...
            errors.append(f"{type(e).__name__}: {e}")
    raise TTSError("Text-to-speech failed — " + " | ".join(errors))

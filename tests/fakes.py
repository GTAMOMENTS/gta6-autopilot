"""Plain helpers for tests (import with `from fakes import ...`; tests/ is on the pytest pythonpath).

Fixtures that use them live in conftest.py (make_clip, fake_tts).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="FFmpeg is not installed")

WORDS_PER_SEC = 2.6  # the product's own speaking-speed estimate (video_processor._expected_sec)


def ffmpeg(*args: str) -> None:
    """Run ffmpeg quietly; raises if it fails."""
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True, timeout=300)


def stream_info(path: Path) -> dict:
    """{"width", "height", "v_dur", "a_dur", "duration"} of a rendered file (per-stream durations)."""
    res = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries",
                          "format=duration:stream=codec_type,width,height,duration", str(path)],
                         capture_output=True, text=True, timeout=60, check=True)
    data = json.loads(res.stdout)
    out = {"width": 0, "height": 0, "v_dur": 0.0, "a_dur": 0.0,
           "duration": float(data.get("format", {}).get("duration") or 0)}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not out["width"]:
            out.update(width=int(s["width"]), height=int(s["height"]), v_dur=float(s.get("duration") or 0))
        elif s.get("codec_type") == "audio" and not out["a_dur"]:
            out["a_dur"] = float(s.get("duration") or 0)
    return out


def fake_tts_wav(prompt: str, model: str, voice: str, out_wav: Path, wps: float = WORDS_PER_SEC) -> Path:
    """Stand-in for video_processor._gemini_tts_wav: one tone per script paragraph (as long as the words would
    take to say), 0.7 s pause between paragraphs - so the one-take narration splits at the pauses."""
    paras = [q for q in prompt.split("\n\n") if q.strip()]
    filt = "".join(f"sine=f={500 + 40 * i}:d={len(q.split()) / wps:.2f}:sample_rate=48000,apad=pad_dur=0.7[a{i}];"
                   for i, q in enumerate(paras))
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-filter_complex", filt + "".join(f"[a{i}]" for i in range(len(paras)))
           + f"concat=n={len(paras)}:v=0:a=1[o]", "-map", "[o]", "-ar", "48000", str(out_wav))
    return out_wav


def heard(*spans: tuple[float, str]) -> list[dict]:
    """Fake Whisper output: heard((0.5, "Number one."), (3.0, "next line")) -> one word dict per word,
    0.3 s per word starting at each span's time."""
    out = []
    for start, text in spans:
        t = start
        for w in text.split():
            out.append({"text": w, "start": round(t, 3), "end": round(t + 0.25, 3), "prob": 0.9})
            t += 0.3
    return out

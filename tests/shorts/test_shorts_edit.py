"""Shorts editing (video_processor captions / fx, highlight cuts): word timing, caption chunks, safe zones,
emphasis words, dialogue-aware line placement, zoom expressions, native-resolution cuts, impact detection and
one small end-to-end render. Self-contained: synthetic audio / video made with FFmpeg, no network, no AI.
Run: python -m pytest tests/shorts -q"""
from __future__ import annotations

import math
import re
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import highlight  # noqa: E402
import transcribe  # noqa: E402
import video_processor as vp  # noqa: E402
from common import probe  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
SAFE_TOP, SAFE_BOTTOM, SAFE_LEFT, SAFE_RIGHT = 192, 1536, 60, 930  # 1080x1920: top 10 %, bottom 20 %, buttons


@pytest.fixture(autouse=True)
def _no_whisper(monkeypatch):
    """Whisper is never downloaded in tests (each test that wants it fakes it)."""
    monkeypatch.setattr(transcribe, "_get_model", lambda: None)


def _vowel_groups(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", re.sub(r"[^a-z]", "", w.lower()))))


def _fake_voice(path: Path, text: str, sr: int = 48000) -> list[tuple[str, float, float]]:
    """A 'voice' with real speech structure: one tone per word (length by syllables), tiny gaps between words,
    pauses after commas / full stops. Returns the true (word, start, end)."""
    samples, truth, t = [0] * int(sr * 0.08), [], 0.08
    for i, w in enumerate(text.split()):
        d = 0.1 + 0.15 * _vowel_groups(w) + (0.25 if re.search(r"\d", w) else 0)
        n = int(sr * d)
        f = 280 + 50 * (i % 5)
        samples += [int(11000 * min(1, k / 300, (n - k) / 300) * math.sin(2 * math.pi * f * k / sr)) for k in range(n)]
        truth.append((w, round(t, 3), round(t + d, 3)))
        t += d
        gap = 0.45 if re.search(r"[.!?]$", w) else 0.24 if re.search(r"[,;:]$", w) else 0.035
        samples += [0] * int(sr * gap)
        t += gap
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return truth


LINES = ["Five stars, and he still goes for the jump!",
         "The landing is actually brutal, look at that. Somehow he walks away.",
         "Number four, you can rob a store with a mask on.",
         "Would you have tried that jump or taken the bridge?"]


# --------------------------------------------------------------------------- word timing

@needs_ffmpeg
@pytest.mark.parametrize("text", LINES)
def test_estimated_word_times_follow_the_voice(tmp_path, text):
    wav = tmp_path / "line.wav"
    truth = _fake_voice(wav, text)
    dur = probe(wav)["duration"]
    est = vp._estimate_word_times(text, wav, dur)
    assert [w["text"] for w in est] == text.split(), "captions = exactly the script's words"
    err = [abs(w["start"] - s) for w, (_, s, _) in zip(est, truth)]
    even = [abs(w["start"] - s) for w, (_, s, _) in zip(vp._even_word_timings(text, dur, lead=0.05), truth)]
    assert sum(err) / len(err) < 0.12, err
    assert max(err) < 0.3, err
    assert sum(err) <= sum(even) + 1e-6, "must beat the old even spread"
    assert all(b["start"] >= a["start"] for a, b in zip(est, est[1:]))


def test_whisper_word_times_keep_the_script_text(monkeypatch):
    text = "Somehow he walks away with $500 like nothing happened."
    heard = [("Somehow", 0.10, 0.55), ("he", 0.58, 0.70), ("walks", 0.72, 1.00), ("away", 1.02, 1.30),
             ("with", 1.32, 1.45), ("$500", 1.48, 2.20),  # "like" not heard
             ("nothing", 2.62, 2.95), ("happened.", 2.97, 3.40)]
    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "words", lambda path, start=0.0, end=None: (
        [{"text": t, "start": a, "end": b, "prob": 0.9} for t, a, b in heard], {"language": "en"}))
    out = vp._whisper_word_times(Path("piece.wav"), text)
    assert [w["text"] for w in out] == text.split()
    assert out[0]["start"] == pytest.approx(0.10) and out[5]["start"] == pytest.approx(1.48)
    like = out[6]
    assert 2.2 <= like["start"] < like["end"] <= 2.62, "unheard word is placed between its neighbours"
    assert all(b["start"] >= a["start"] for a, b in zip(out, out[1:]))
    # too little of the line heard -> None (the caller falls back to the pause-based estimate)
    monkeypatch.setattr(transcribe, "words", lambda *a, **k: ([{"text": "uh", "start": 0, "end": 0.2}], {}))
    assert vp._whisper_word_times(Path("piece.wav"), text) is None


# --------------------------------------------------------------------------- caption chunks / emphasis

@pytest.mark.parametrize("text", LINES + ["GTA 6 finally lets you buy a $2,000,000 penthouse, and it's INSANE.",
                                          "Unbelievably, the streets of Vice City are packed tonight."])
def test_caption_chunks(text):
    words = vp._spread(text.split(), 0.0, len(text.split()) / 2.6)
    groups = vp._caption_groups(words, 104, 800)
    assert [w["text"] for g in groups for w in g] == text.split(), "nothing dropped, nothing reordered"
    for g in groups[:-1]:
        assert len(g) <= 4
        assert vp._bare(g[-1]["text"]).lower() not in {"the", "a", "an", "to", "of"}, groups
        if len(g) > 1:
            assert vp._text_w(" ".join(w["text"] for w in g), 104) <= 800
    assert not any(len(g) == 1 and g[0]["text"] in ("Vice", "City") for g in groups), "names stay together"


def test_emphasis_words():
    texts = "GTA 6 gave him $500 and number four is INSANE, this one is cool".split()
    got = {t for t, f in zip(texts, vp._emphasis_flags(texts)) if f}
    assert {"$500", "four", "INSANE,"} <= got
    assert not {"GTA", "6", "one", "cool"} & got


def test_balanced_banner_wrap():
    lines, size = vp._wrap("HE TOOK THE JUMP AT 200 MPH", 84, 520, 3, 52)
    assert len(lines) >= 2 and all(len(ln.split()) >= 2 for ln in lines), lines  # no lonely last word


# --------------------------------------------------------------------------- ASS: safe zones, karaoke, end card

def _events(ass: str) -> list[dict]:
    out = []
    for line in ass.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        layer, start, end, style, *_rest = line[10:].split(",", 9)
        text = line.split(",", 9)[9]
        out.append({"layer": int(layer), "start": start, "end": end, "style": style, "text": text})
    return out


def test_build_ass_safe_zones_karaoke_and_end_card(tmp_path):
    words, t = [], 1.3
    for line in LINES:
        for w in line.split():
            words.append({"text": w, "start": t, "end": t + 0.3})
            t += 0.33
        t += 0.6
    total = t + 1.5
    q_start = words[-len(LINES[-1].split())]["start"]
    ass_path = tmp_path / "t.ass"
    marks = vp.build_ass(ass_path, "HE TOOK THE JUMP AT 200 MPH", words, total, "FOLLOW FOR DAILY GTA 6 MOMENTS",
                         flashes=[(1.2, 0.9), (5.0, 0.4, 0.15)], question={"text": LINES[-1], "start": q_start})
    ev = _events(ass_path.read_text(encoding="utf-8"))
    # the highlighted (popped) word walks through the script in order = captions match the voice word for word
    overlay = [re.sub(r"\{[^}]*\}", "", e["text"]) for e in ev if e["style"] == "Caption" and e["layer"] == 2]
    assert overlay == [w["text"].upper() for w in words]
    # every caption line sits inside the safe zone (not under the buttons, not in the bottom/top UI)
    for e in ev:
        if e["style"] != "Caption" or e["layer"] != 1:
            continue
        x, y = map(float, re.search(r"\\pos\(([\d.]+),([\d.]+)\)", e["text"]).groups())
        size = float(re.search(r"\\fs([\d.]+)", e["text"]).group(1))
        plain = re.sub(r"\{[^}]*\}", "", e["text"])
        half = vp._text_w(plain, size) * 1.08 / 2  # + the wider word gaps
        assert SAFE_TOP < y - size / 2 and y + size / 2 < SAFE_BOTTOM
        assert SAFE_LEFT <= x - half and x + half <= SAFE_RIGHT, (plain, x, half)
    banner = [e for e in ev if e["style"] == "Banner"]
    assert len(banner) == 1 and banner[0]["start"] == "0:00:00.00" and "\\fscx85" in banner[0]["text"]
    assert int(re.search(r"\\pos\(540,(\d+)\)", banner[0]["text"]).group(1)) >= SAFE_TOP
    card = [e for e in ev if e["style"] == "Card"]
    assert len(card) == 1 and card[0]["start"] == vp._ass_time(q_start) == banner[0]["end"]
    assert "BRIDGE?" in card[0]["text"]
    cta = [e for e in ev if e["style"] == "CTA"]
    assert len(cta) == 1 and cta[0]["end"] == vp._ass_time(total)
    assert len([e for e in ev if e["style"] == "Flash"]) == 2
    assert marks["pops"] == [0.0, q_start]
    assert any(abs(a - words[0]["start"]) < 1e-6 for a, _ in marks["emphasis"])  # "Five" (stars)


# --------------------------------------------------------------------------- placement around game dialogue

def test_place_lines_slides_off_game_dialogue():
    voiced = [{"t": 1.3, "text": "Hook line here.", "dur": 2.0, "path": "a"},
              {"t": 6.0, "text": "Middle line here.", "dur": 2.0, "path": "b"},
              {"t": 12.0, "text": "Would you do it?", "dur": 1.5, "path": "c"}]
    plain = vp.place_lines(voiced, 20.0)
    moved = vp.place_lines(voiced, 20.0, avoid=[(5.5, 7.0), (1.0, 1.8)])
    assert [p["start"] for p in plain] == [1.3, 6.0, 12.0]
    assert 1.3 <= moved[0]["start"] <= 1.3 + 0.4 + 1e-6, "the hook never moves into the cold open, never waits long"
    assert moved[1]["start"] >= 6.9 and moved[1]["start"] <= 7.5 + 1e-6, "slides into the gap after the dialogue"
    assert moved[2]["start"] == 12.0


# --------------------------------------------------------------------------- FFmpeg parts

@needs_ffmpeg
def test_zoom_filter_runs_in_ffmpeg(tmp_path):
    events = [{"a": -1.0, "b": 0.0, "amt": 0.1, "in": 0.01, "out": 0.5}, {"a": 0.6, "b": 0.9, "amt": 0.22}]
    zf = vp._zoom_filter(events, [(0.7, 0.4)], 14, size="540x960")
    out = tmp_path / "z.mp4"
    res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=540x960:r=30:d=1.5",
                          "-vf", f"fps=30{zf}", "-c:v", "libx264", "-preset", "ultrafast", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-500:]
    assert probe(out)["duration"] == pytest.approx(1.5, abs=0.1)
    assert vp._zoom_filter([]) == ""


def _clip(path: Path, dur: float, size: str, bursts=()) -> Path:
    expr = "0.06*sin(2*PI*180*t)+0.03*(random(0)-0.5)" + "".join(
        f"+0.9*(random(1)-0.5)*between(t,{b},{b + 0.5})" for b in bursts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={dur}",
                    "-f", "lavfi", "-i", f"aevalsrc='{expr}|{expr}':s=48000:d={dur}", "-shortest",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)], check=True)
    return path


@needs_ffmpeg
def test_cut_segments_keeps_native_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(highlight, "WORK_DIR", tmp_path)
    src = _clip(tmp_path / "src.mp4", 12, "1280x720")
    out = highlight.cut_segments(src, tmp_path / "cut.mp4", [(6.0, 7.2), (1.0, 5.0)])
    info = probe(out)
    assert (info["width"], info["height"]) == (1280, 720), "720p is not blown up to 1080p any more"
    assert info["duration"] == pytest.approx(5.2, abs=0.1) and info["has_audio"]
    vert = _clip(tmp_path / "v.mp4", 3, "1080x1920")
    vi = probe(highlight.cut_segments(vert, tmp_path / "vcut.mp4", [(0.5, 2.5)]))
    assert (vi["width"], vi["height"]) == (1080, 1920), "vertical sources keep their full width"


@needs_ffmpeg
def test_impacts_find_sudden_loud_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(highlight, "WORK_DIR", tmp_path)
    src = _clip(tmp_path / "imp.mp4", 16, "320x180", bursts=(3.3, 9.0, 13.2))
    assert highlight.impacts(src, count=3, min_gap=3.0) == pytest.approx([3.3, 9.0, 13.2], abs=0.15)
    assert highlight.impacts(src, count=3, min_gap=3.0, skip=[(8.5, 9.5)]) == pytest.approx([3.3, 13.2], abs=0.15)


@needs_ffmpeg
def test_render_small_short(tmp_path, monkeypatch):
    """6 s clip, two voice lines: vertical 1080x1920, sound = picture length, captions = the voice's words,
    no high technical QA issue."""
    import music
    import qa
    monkeypatch.setattr(vp, "VO_DIR", tmp_path / "voice")
    monkeypatch.setattr(vp, "READY_DIR", tmp_path / "ready")
    monkeypatch.setattr(music, "pick", lambda: None)
    (tmp_path / "voice").mkdir()
    src = _clip(tmp_path / "clip.mp4", 6, "1280x720", bursts=(3.0,))
    voiced = []
    for i, (t, text) in enumerate([(0.3, "He really went for it, look!"), (3.4, "Would you try that jump?")]):
        wav = tmp_path / f"l{i}.wav"
        _fake_voice(wav, text)
        voiced.append({"t": t, "text": text, "path": wav, "dur": probe(wav)["duration"], "words": []})
    r = vp.render(src, "tiny", "HE WENT FOR IT", [], voiced=voiced,
                  fx={"zoom": [(2.9, 3.6)], "shake": [(3.0, 0.4)], "flash": [(3.0, 0.5)],
                      "sfx": [{"t": 2.0, "kind": "riser"}, {"t": 3.0, "kind": "hit"}], "action": [(2.5, 4.5)]})
    assert r, "render failed"
    out = Path(r["path"])
    info = probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    v, a = qa._sdur(out, "v:0"), qa._sdur(out, "a:0")
    assert abs(v - a) <= 0.12, (v, a)
    assert [c["text"] for c in r["captions"]] == " ".join(t for _, t in
                                                          [(0, voiced[0]["text"]), (1, voiced[1]["text"])]).split()
    high = [i for i in qa.technical(out) if i["severity"] == "high"]
    assert high == [], high

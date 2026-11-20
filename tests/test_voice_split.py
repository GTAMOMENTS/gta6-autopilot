"""Narration voice (video_processor.py): Whisper split of the one-take Gemini narration into lines, token
normalisation, and the owner's voice rules (TTS gets only the script; never a robotic fallback voice)."""
from __future__ import annotations

from pathlib import Path

import pytest

import transcribe
import video_processor as vp
from fakes import WORDS_PER_SEC, heard, needs_ffmpeg

FULL = Path("/nonexistent/narration_all.wav")


@pytest.mark.parametrize("text,want", [
    ("Number 14.", ["number", "fourteen"]),
    ("NUMBER FOURTEEN!", ["number", "fourteen"]),
    ("0 1 7 20 21 100", ["zero", "one", "seven", "twenty", "21", "100"]),
    ("It's 25 bucks, don't panic", ["it's", "25", "bucks", "don't", "panic"]),
    ("'quoted' words", ["quoted", "words"]),
    ("Mr. Smith - 3rd place", ["mr", "smith", "3rd", "place"]),
    ("", []),
])
def test_norm_tokens(text, want):
    assert vp._norm_tokens(text) == want


def test_norm_tokens_amounts_are_consistent():
    # script and Whisper may format money differently - both normalise to the same tokens when they agree
    assert vp._norm_tokens("$4,500") == vp._norm_tokens("4,500 dollars")[:2] == ["four", "500"]
    assert vp._norm_tokens("4.5") == ["45"], "dots are dropped before tokenising"


@pytest.fixture
def whisper(monkeypatch):
    def install(words, dur=30.0):
        monkeypatch.setattr(transcribe, "_get_model", lambda: object())
        monkeypatch.setattr(transcribe, "words", lambda path, *a, **k: (list(words), {}))
        monkeypatch.setattr(vp, "probe", lambda path: {"duration": dur, "width": 0, "height": 0, "has_audio": True})
    return install


LINES = ["He really thought that ramp was a good idea.",
         "Five stars and he still goes for the jump.",
         "Would you have tried that jump?"]


def test_whisper_not_available_returns_none():
    assert vp._whisper_cuts(FULL, LINES) is None  # conftest: Whisper "not installed"


def test_split_at_the_pauses(whisper):
    words = heard((0.4, LINES[0]), (4.0, LINES[1]), (7.6, LINES[2]))
    whisper(words, dur=10.0)
    cuts = vp._whisper_cuts(FULL, LINES)
    assert len(cuts) == 3
    first_end = words[8]["end"]          # "idea." ends at 2.80
    second_start = words[9]["start"]     # "Five" at 4.0
    gap_mid = (first_end + second_start) / 2
    assert cuts[0][0] == pytest.approx(0.4 - 0.08)
    assert cuts[0][1] == pytest.approx(min(gap_mid, first_end + 0.15)), "tight: right after the last word"
    assert cuts[1][0] == pytest.approx(max(gap_mid, second_start - 0.08)), "tight: right before the first word"
    assert cuts[2][1] == pytest.approx(min(10.0, words[-1]["end"] + 0.18))
    for (a, b), (c, d) in zip(cuts, cuts[1:]):
        assert a < b <= c < d, "pieces in order, never overlapping"


def test_split_last_piece_capped_at_audio_end(whisper):
    whisper(heard((0.4, LINES[0]), (4.0, LINES[1]), (7.6, LINES[2])), dur=8.9)
    assert vp._whisper_cuts(FULL, LINES)[-1][1] == pytest.approx(8.9)


def test_split_digits_vs_number_words(whisper):
    texts = ["Number fourteen is the fastest car.", "And it only costs 15 grand."]
    whisper(heard((0.3, "Number 14 is the fastest car."), (3.5, "And it only costs fifteen grand.")), dur=6.0)
    cuts = vp._whisper_cuts(FULL, texts)
    assert len(cuts) == 2 and cuts[0][1] <= cuts[1][0]


def test_split_ignores_extra_filler_between_lines(whisper):
    whisper(heard((0.4, LINES[0]), (3.2, "uh"), (4.0, LINES[1]), (7.6, LINES[2])), dur=10.0)
    cuts = vp._whisper_cuts(FULL, LINES)
    # boundary = the LONGEST pause between the two lines' matched words (before or after the "uh")
    assert cuts[0][1] <= 3.2 or cuts[1][0] >= 3.45


def test_split_tolerates_a_misheard_word(whisper):
    misheard = LINES[1].replace("stars", "cars")
    whisper(heard((0.4, LINES[0]), (4.0, misheard), (7.6, LINES[2])), dur=10.0)
    assert len(vp._whisper_cuts(FULL, LINES)) == 3


def test_no_speech_heard_raises(whisper):
    whisper([])
    with pytest.raises(vp._SplitError, match="no speech"):
        vp._whisper_cuts(FULL, LINES)


def test_missing_line_raises(whisper):
    whisper(heard((0.4, LINES[0]), (7.6, LINES[2])))
    with pytest.raises(vp._SplitError, match="could not find the line 'Five stars"):
        vp._whisper_cuts(FULL, LINES)


def test_lines_in_wrong_order_raise(whisper):
    texts = ["Grab the car first.", "Then drive to the docks."]
    whisper(heard((0.4, "Then drive to the docks."), (3.0, "Grab the car first.")))
    with pytest.raises(vp._SplitError):
        vp._whisper_cuts(FULL, texts)


# --------------------------------------------------------------------------- the whole voice step (FFmpeg)

@needs_ffmpeg
def test_one_take_narration_split_into_line_files(fake_tts, monkeypatch, sandbox):
    """Gemini one-take audio (fake tones) + Whisper word times -> one file per line, each as long as its line."""
    texts = LINES
    # the fake TTS speaks paragraph i for words/2.6 s, then a 0.7 s pause: build what Whisper would hear
    spans, t = [], 0.0
    for text in texts:
        d = len(text.split()) / WORDS_PER_SEC
        n = len(text.split())
        spans += [{"text": w, "start": t + d * k / n, "end": t + d * (k + 0.8) / n, "prob": 0.9}
                  for k, w in enumerate(text.split())]
        t += d + 0.7
    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "words", lambda path, *a, **k: (spans, {}))
    out = vp.synthesize_lines([{"t": 0.0, "text": x} for x in texts], "split_test")
    assert fake_tts.prompts == ["\n\n".join(texts)], "ONE request, and it contains only the script"
    assert [o["text"] for o in out] == texts
    for o in out:
        expect = len(o["text"].split()) / WORDS_PER_SEC
        assert Path(o["path"]).exists() and Path(o["path"]).parent == sandbox.voice
        assert expect * 0.8 < o["dur"] < expect + 0.6, (o["text"], o["dur"], expect)
        assert o["words"], "word timings for the captions"


@needs_ffmpeg
def test_split_without_whisper_uses_pauses(fake_tts, sandbox):
    out = vp.synthesize_lines([{"t": 0.0, "text": x} for x in LINES], "pause_split")
    assert len(out) == 3 and all(o["dur"] > 1.0 for o in out)


def test_tts_prompt_is_only_the_script():
    assert vp._tts_prompt(["  Line one. ", "", "Line two?"]) == "Line one.\n\nLine two?"


def test_no_robot_voice_without_gemini(monkeypatch, cfg):
    """Owner's rule: human voice (Gemini Fenrir) only - no key / Gemini down = no video, never edge-tts/gTTS."""
    assert cfg["voice"]["engine"] == "gemini" and cfg["voice"]["robot_fallback"] is False
    monkeypatch.setattr(vp, "_edge_line", lambda *a, **k: pytest.fail("robotic edge-tts voice was used"))
    with pytest.raises(vp.HumanVoiceUnavailable, match="GEMINI_API_KEY is not set"):
        vp.synthesize_lines([{"t": 0, "text": "Hello there friends."}], "x")


def test_no_robot_voice_when_gemini_tts_fails(monkeypatch, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(vp, "_gemini_tts_models", lambda: ["tts-a", "tts-b"])
    tried = []

    def broken(prompt, model, voice, out):
        tried.append((model, voice))
        raise RuntimeError("500 internal")
    monkeypatch.setattr(vp, "_gemini_tts_wav", broken)
    monkeypatch.setattr(vp, "_edge_line", lambda *a, **k: pytest.fail("robotic edge-tts voice was used"))
    with pytest.raises(vp.HumanVoiceUnavailable):
        vp.synthesize_lines([{"t": 0, "text": "Hello there friends."}], "x")
    assert [m for m, _ in tried] == ["tts-a", "tts-b"], "every Gemini TTS model is tried once"
    assert all(v == "Fenrir" for _, v in tried), "always the owner's voice"


def test_place_lines_keeps_hook_and_closing_question():
    voiced = [{"t": 0.0, "text": "Hook", "dur": 2.0}, {"t": 3.0, "text": "Middle long", "dur": 5.0},
              {"t": 6.0, "text": "Middle", "dur": 2.0}, {"t": 9.0, "text": "Question?", "dur": 2.0}]
    placed = vp.place_lines(voiced, total=10.0)
    texts = [p["text"] for p in placed]
    assert texts[0] == "Hook" and texts[-1] == "Question?" and "Middle long" not in texts
    for a, b in zip(placed, placed[1:]):
        assert a["start"] + a["dur"] <= b["start"], "no overlapping lines"
    assert placed[-1]["start"] + placed[-1]["dur"] <= 10.0

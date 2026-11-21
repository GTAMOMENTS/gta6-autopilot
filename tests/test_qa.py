"""Strict QA (qa.py): Whisper voice check, intentional-still ranges, the verdict report and technical checks."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import qa
import transcribe
from fakes import fake_tts_wav, ffmpeg, heard, needs_ffmpeg

VIDEO = Path("/nonexistent/final.mp4")  # voice_check only hands it to (faked) Whisper


@pytest.fixture
def whisper(monkeypatch):
    """whisper(words) -> Whisper "hears" exactly these words; records the files it was asked about."""
    asked = []

    def install(words, model=object()):
        def fake_words(path, *a, **k):
            asked.append(Path(path))
            return list(words), {"language": "en", "language_probability": 0.99}
        monkeypatch.setattr(transcribe, "words", fake_words)
        monkeypatch.setattr(transcribe, "_get_model", lambda: model)
        return asked
    return install


def line(text, start, dur=2.5, **kw):
    return {"text": text, "start": start, "dur": dur, **kw}


# --------------------------------------------------------------------------- voice check

def test_voice_in_sync_passes(whisper):
    whisper(heard((1.0, "He really thought that ramp was a good idea."),
                  (5.0, "Five stars and he still goes for the jump.")))
    lines = [line("He really thought that ramp was a good idea.", 1.0),
             line("Five stars and he still goes for the jump.", 5.0)]
    assert qa.voice_check(VIDEO, lines) == []


def test_voice_slightly_late_is_tolerated(whisper):
    whisper(heard((1.45, "He really thought that ramp was a good idea.")))
    assert qa.voice_check(VIDEO, [line("He really thought that ramp was a good idea.", 1.0)]) == []


def offset_of(issue) -> float:
    return float(re.search(r"spoken ([+-][\d.]+)s", issue["detail"]).group(1))


def test_voice_late_is_out_of_sync(whisper):
    whisper(heard((2.3, "He really thought that ramp was a good idea.")))
    issues = qa.voice_check(VIDEO, [line("He really thought that ramp was a good idea.", 1.0)])
    assert [i["type"] for i in issues] == ["voice_out_of_sync"]
    assert issues[0]["severity"] == "high" and issues[0]["at"] == 1.0
    # anchor = "really" (the first non-filler word), expected 0.32 s per word after the line start
    assert offset_of(issues[0]) == pytest.approx(1.28, abs=0.02)


def test_voice_early_is_out_of_sync(whisper):
    whisper(heard((4.1, "Somehow he walks away like nothing happened.")))
    issues = qa.voice_check(VIDEO, [line("Somehow he walks away like nothing happened.", 5.0)])
    assert [i["type"] for i in issues] == ["voice_out_of_sync"]
    assert offset_of(issues[0]) == pytest.approx(-0.9, abs=0.02)


SYNC_LINE = "Somehow he walks away like nothing happened right there."


@pytest.mark.parametrize("early", [1.1, 1.6, 2.2])
def test_voice_far_too_early_is_out_of_sync(whisper, early):
    whisper(heard((5.0 - early, SYNC_LINE)))
    issues = qa.voice_check(VIDEO, [line(SYNC_LINE, 5.0, dur=3.0)])
    assert [i["type"] for i in issues] == ["voice_out_of_sync"]


@pytest.mark.parametrize("late", [2.6, 2.9])
def test_voice_far_too_late_is_out_of_sync(whisper, late):
    whisper(heard((5.0 + late, SYNC_LINE)))
    issues = qa.voice_check(VIDEO, [line(SYNC_LINE, 5.0, dur=3.0)])
    assert {i["type"] for i in issues} & {"voice_out_of_sync", "voice_mismatch"}


def test_wrong_words_are_a_mismatch(whisper):
    whisper(heard((1.0, "Completely different words were spoken here instead.")))
    issues = qa.voice_check(VIDEO, [line("He really thought that ramp was a good idea.", 1.0)])
    assert [i["type"] for i in issues] == ["voice_mismatch"]
    assert issues[0]["severity"] == "high" and "0%" in issues[0]["detail"]


def test_filler_word_of_previous_line_is_not_the_anchor(whisper):
    """The line's own "the" is swallowed; the "the" that ends the previous line must not decide the timing."""
    whisper(heard((1.0, "Grab the money and go for the"), (4.3, "cops are coming fast")))
    lines = [line("Grab the money and go for the", 1.0), line("The cops are coming fast", 4.0)]
    assert qa.voice_check(VIDEO, lines) == []


def test_number_words_match_digits_both_ways(whisper):
    whisper(heard((1.0, "Number 14. This car hits 15 stars"), (6.0, "and fourteen cops chase you")))
    lines = [line("Number fourteen. This car hits fifteen stars", 1.0), line("And 14 cops chase you", 6.0)]
    assert qa.voice_check(VIDEO, lines) == []


def test_caption_emphasis_stars_are_ignored(whisper):
    whisper(heard((1.0, "This is insane")))
    assert qa.voice_check(VIDEO, [line("This is *insane*", 1.0)]) == []


def test_partial_line_above_threshold_passes(whisper, cfg):
    # 4 of 5 words heard = 80% >= min_word_match (60%)
    whisper(heard((1.0, "really thought that ramp")))
    assert cfg["qa"]["min_word_match"] <= 0.8
    assert qa.voice_check(VIDEO, [line("He really thought that ramp", 1.0)]) == []


def test_no_lines_nothing_to_check(whisper):
    asked = whisper([])
    assert qa.voice_check(VIDEO, []) == []
    assert asked == [], "Whisper is not even started"


def test_whisper_missing_is_low_severity():
    # conftest: Whisper "not installed"
    issues = qa.voice_check(VIDEO, [line("Anything at all here", 1.0)])
    assert issues == [{"severity": "low", "type": "voice_unchecked", "detail": "Whisper not available"}]


def test_no_speech_heard_is_high(whisper):
    whisper([])
    issues = qa.voice_check(VIDEO, [line("Anything at all here", 1.0)])
    assert [(i["severity"], i["type"]) for i in issues] == [("high", "voice_missing")]


@needs_ffmpeg
def test_voice_check_listens_to_narration_stem(whisper, tmp_path):
    """With the voice files known, Whisper gets the narration alone (placed as in the mix), not the final video."""
    voices = [fake_tts_wav("one two three four", "m", "v", tmp_path / f"l{i}.wav") for i in range(2)]
    video = tmp_path / "final.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=6", "-f", "lavfi", "-i", "sine=f=200:d=6",
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video))
    asked = whisper(heard((0.5, "one two three four"), (3.0, "one two three four")))
    lines = [line("one two three four", 0.5, 1.6, path=str(voices[0])),
             line("one two three four", 3.0, 1.6, path=str(voices[1]))]
    assert qa.voice_check(video, lines) == []
    assert len(asked) == 1 and asked[0].name == "final_narration.wav"
    assert not asked[0].exists(), "the temporary narration stem is cleaned up"


# --------------------------------------------------------------------------- _outside / _ranges

@pytest.mark.parametrize("r,allowed,dur,want", [
    ((2.0, 5.0), [], 1e9, 3.0),
    ((2.0, 5.0), [(0.0, 10.0)], 1e9, 0.0),                # fully intentional
    ((2.0, 5.0), [(4.0, 10.0)], 1e9, 2.0),                # partly
    ((2.0, 8.0), [(3.0, 4.0), (6.0, 7.0)], 1e9, 4.0),     # two allowed ranges
    ((2.0, 1001.0), [], 6.0, 4.0),                        # no end timestamp: capped at the video's end
    ((2.0, 1001.0), [(5.0, 6.0)], 6.0, 3.0),              # still outro to the end is allowed
    ((7.0, 9.0), [], 6.0, 0.0),                           # starts after the end
])
def test_outside(r, allowed, dur, want):
    assert qa._outside(r, allowed, dur) == pytest.approx(want)


def test_ranges_parse_ffmpeg_output():
    err = ("[silencedetect @ 0x1] silence_start: 1.5\n[silencedetect @ 0x1] silence_end: 4.25 | silence_duration: 2.75\n"
           "[silencedetect @ 0x1] silence_start: 9\n")
    assert qa._ranges(err, "silence_start", "silence_end") == [(1.5, 4.25), (9.0, 1008.0)]
    assert qa._ranges("", "silence_start", "silence_end") == []


# --------------------------------------------------------------------------- verdict + report

@pytest.fixture
def checks(monkeypatch):
    """Fake the three QA parts; returns a dict the test fills (technical / voice issues, AI answer)."""
    state = {"technical": [], "voice": [], "ai": None}
    monkeypatch.setattr(qa, "technical", lambda path, still_ok=None: list(state["technical"]))
    monkeypatch.setattr(qa, "voice_check", lambda path, lines: list(state["voice"]))
    monkeypatch.setattr(qa, "ai_review", lambda path, context: state["ai"])
    return state


def test_report_pass_with_pro_level(checks, tmp_path):
    video = tmp_path / "short_1.mp4"
    checks["voice"] = [{"severity": "low", "type": "voice_unchecked", "detail": "Whisper not available"}]
    checks["ai"] = {"publish": True, "score": 8, "summary": "clean", "pro_level": 6,
                    "issues": [{"at": 3.2, "severity": "medium", "type": "pacing", "detail": "slow middle"}],
                    "improvements": ["Cut the first 3 s, start on the crash", "", "x" * 300] + [f"tip {i}" for i in range(9)]}
    v = qa.check(video, "short")
    assert v["pass"] is True and v["score"] == 8 and v["pro_level"] == 6
    lines = v["report"].splitlines()
    assert lines[0] == "QA PASSED (short) - AI score: 8"
    assert "- [low] voice_unchecked: Whisper not available" in lines
    assert "- [medium] ai:pacing at 3.2s: slow middle" in lines
    assert "PRO LEVEL: 6/10 - to look like a top YouTuber's edit:" in lines
    tips = [ln for ln in lines if ln.startswith("  * ")]
    assert tips[0] == "  * Cut the first 3 s, start on the crash"
    assert len(v["improvements"]) == 8, "at most 8 tips, empty ones dropped"
    assert max(len(t) for t in v["improvements"]) == 220
    assert (tmp_path / "short_1_QA.txt").read_text(encoding="utf-8") == v["report"] + "\n"


def test_no_ai_review_fails_in_strict_mode(checks, tmp_path):
    v = qa.check(tmp_path / "v.mp4", "long")
    assert v["pass"] is False and v["score"] is None
    assert [i["type"] for i in v["issues"]] == ["ai_review_unavailable"]
    assert v["report"].startswith("QA FAILED (long) - AI score: n/a")
    assert "PRO LEVEL" not in v["report"]


def test_no_ai_review_allowed_when_not_strict(checks, cfg, tmp_path):
    cfg["qa"]["require_ai_review"] = False
    assert qa.check(tmp_path / "v.mp4", "short")["pass"] is True


@pytest.mark.parametrize("ai,kind", [
    ({"publish": False, "score": 9, "summary": "narration talks over the punchline"}, "ai_verdict"),
    ({"publish": True, "score": 6, "summary": "meh"}, "ai_score"),
    ({"publish": True, "score": 9, "issues": [{"severity": "high", "type": "spoiler", "detail": "ending"}]}, "ai:spoiler"),
])
def test_ai_editor_can_block(checks, tmp_path, ai, kind):
    checks["ai"] = ai
    v = qa.check(tmp_path / "v.mp4", "stream")
    assert v["pass"] is False
    assert kind in [i["type"] for i in v["issues"] if i["severity"] == "high"]


def test_technical_high_issue_blocks(checks, tmp_path):
    checks["technical"] = [{"severity": "high", "type": "av_length", "detail": "audio 10.00s vs video 12.00s"}]
    checks["ai"] = {"publish": True, "score": 9}
    v = qa.check(tmp_path / "v.mp4", "short")
    assert v["pass"] is False and "- [high] av_length: audio 10.00s vs video 12.00s" in v["report"]


def test_medium_issues_do_not_block(checks, tmp_path):
    checks["technical"] = [{"severity": "medium", "type": "black", "at": 1.0, "detail": "black screen for 0.5s"}]
    checks["ai"] = {"publish": True, "score": 7}
    assert qa.check(tmp_path / "v.mp4", "short")["pass"] is True


def test_qa_disabled(checks, cfg, tmp_path):
    cfg["qa"]["enabled"] = False
    assert qa.check(tmp_path / "v.mp4", "short") == {"pass": True, "issues": [], "score": None, "report": "QA disabled"}


def test_ai_review_without_key_is_none(tmp_path):
    assert qa.ai_review(tmp_path / "v.mp4", "ctx") is None


def test_ai_review_parses_answer(monkeypatch, tmp_path):
    import gemini_writer
    from types import SimpleNamespace
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: SimpleNamespace(name="files/1"))
    monkeypatch.setattr(gemini_writer, "_client", lambda: SimpleNamespace(files=SimpleNamespace(delete=lambda name: None)))
    seen = {}

    def gen(contents, temperature=0.9, prefer_pro=False):
        seen["prompt"], seen["pro"] = contents[1], prefer_pro
        return {"publish": True, "score": 8}
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    assert qa.ai_review(tmp_path / "v.mp4", "Banner: HE TOOK THE JUMP") == {"publish": True, "score": 8}
    assert "Banner: HE TOOK THE JUMP" in seen["prompt"] and seen["pro"] is True
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: {"score": 8})  # no verdict key
    assert qa.ai_review(tmp_path / "v.mp4", "") is None


# --------------------------------------------------------------------------- technical checks (real FFmpeg)

def av(tmp_path, name, vsrc, asrc=None, vdur=6.0, adur=None, extra=()):
    out = tmp_path / name
    args = ["-f", "lavfi", "-i", f"{vsrc}:d={vdur}"]
    if asrc:
        args += ["-f", "lavfi", "-i", f"{asrc}"]
    args += [*extra, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if asrc:
        args += ["-c:a", "aac", "-t", str(max(vdur, adur or vdur))]
    ffmpeg(*args, str(out))
    return out


def kinds(issues, severity=None):
    return sorted(i["type"] for i in issues if severity is None or i["severity"] == severity)


@needs_ffmpeg
def test_technical_clean_video(tmp_path):
    v = av(tmp_path, "clean.mp4", "testsrc2=s=320x240:r=30", "sine=f=300:d=6")
    assert kinds(qa.technical(v)) == []


@needs_ffmpeg
def test_technical_no_audio(tmp_path):
    v = av(tmp_path, "mute.mp4", "testsrc2=s=320x240:r=30")
    assert "no_audio" in kinds(qa.technical(v), "high")


@needs_ffmpeg
def test_technical_av_length_mismatch(tmp_path):
    v = av(tmp_path, "short_audio.mp4", "testsrc2=s=320x240:r=30", "sine=f=300:d=4", vdur=6.0, adur=6.0)
    issues = qa.technical(v)
    assert "av_length" in kinds(issues, "high")


@needs_ffmpeg
def test_technical_silence_and_still_ok(tmp_path):
    # tone for 1.5 s, then 4.5 s of silence
    v = av(tmp_path, "quiet.mp4", "testsrc2=s=320x240:r=30", "sine=f=300:d=1.5,apad=whole_dur=6")
    issues = qa.technical(v)
    silence = [i for i in issues if i["type"] == "silence"]
    assert silence and silence[0]["severity"] == "high" and silence[0]["at"] == pytest.approx(1.5, abs=0.2)
    assert "silence" not in kinds(qa.technical(v, still_ok=[(1.0, 6.0)])), "an intentional quiet end card is OK"


@needs_ffmpeg
def test_technical_frozen_picture(tmp_path):
    v = av(tmp_path, "frozen.mp4", "color=c=0x336699:s=320x240:r=30", "sine=f=300:d=6")
    high = kinds(qa.technical(v), "high")
    assert "frozen" in high
    assert "frozen" not in kinds(qa.technical(v, still_ok=[(0.0, 6.0)]))


@needs_ffmpeg
def test_technical_black_and_clipping_are_medium(tmp_path):
    v = av(tmp_path, "black_loud.mp4", "testsrc2=s=320x240:r=30", "sine=f=300:d=6,volume=30",
           extra=["-vf", "drawbox=c=black:t=fill:enable='between(t,2,3.5)'"])
    issues = qa.technical(v)
    assert "black" in kinds(issues, "medium")
    assert "clipping" in kinds(issues, "medium")


def test_whisper_number_spelling_counts_as_heard(whisper):
    # Whisper writes "Number four:" as "No. 4" - that is the line, on time (a long video was blocked for this)
    whisper(heard((7.5, "No. 4: gravity always wins, especially when you jump.")))
    assert qa.voice_check(VIDEO, [line("Number four: gravity always wins, especially when you jump.", 7.5)]) == []


def test_next_real_word_is_tried_when_the_first_is_misheard(whisper):
    whisper(heard((3.0, "Numbr two escaping on a scooter is already funny.")))
    assert qa.voice_check(VIDEO, [line("Number two: escaping on a scooter is already funny.", 3.0)]) == []


def test_stretched_first_word_is_not_out_of_sync(whisper):
    # real run: Whisper heard "Look@6.6 at@8.4 this@8.4 ..." for a line placed at 8.2 - the speech is on time,
    # Whisper stretched "Look" back over the pause before it
    words = [{"text": "Look", "start": 6.6, "end": 8.3}] + [
        {"text": w, "start": 8.4 + 0.2 * i, "end": 8.55 + 0.2 * i}
        for i, w in enumerate("at this guy in GTA 6 stepping out of his SUV".split())]
    whisper(words)
    assert qa.voice_check(VIDEO, [line("Look at this guy in GTA 6 stepping out of his SUV", 8.2, dur=3.0)]) == []

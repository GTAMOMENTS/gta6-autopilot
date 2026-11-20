"""Long-video pro edit (long_video.py render side): brand overlay timeline (_build_ass), YouTube chapters, thumbnail
text layout, and - slow - the new pieces (push-in hook, moving end screen) being exactly as long in audio as in
video, plus the 3 thumbnail variants."""
from __future__ import annotations

import re

import pytest

import long_video as lv
from fakes import needs_ffmpeg, stream_info

TL = {"title": "Top 3 CRAZIEST GTA 6 Moments Today (IShowSpeed, Kai Cenat)", "brand": "GTA VI Top Moments",
      "handle": "@GTAVI6TopMoments", "total": 46.0, "outro_at": 34.0,
      "outro_text": "WHICH ONE WAS THE BEST?\\N\\NSUBSCRIBE FOR DAILY GTA 6 MOMENTS", "progress": True,
      "hook": (0.0, 4.0, "TODAY'S #1 MOMENT"), "coming": (4.0, 10.0), "flashes": [5.5, 7.0, 8.5],
      "intro": (4.3, 10.0),
      "cards": [{"start": 10.0, "end": 19.0, "n": 3, "label": "Kai {Cenat} Vs The Police Helicopter"},
                {"start": 19.0, "end": 27.0, "n": 2, "label": "Speed Jumps The Bridge"},
                {"start": 27.0, "end": 34.0, "n": 1, "label": "caseoh's Yacht Party"}],
      "credits": [{"start": 10.0, "end": 19.0, "name": "KaiCenat", "platform": "twitch"},
                  {"start": 19.0, "end": 27.0, "name": "IShowSpeed", "platform": "kick"},
                  {"start": 27.0, "end": 34.0, "name": "caseoh_", "platform": "x"}],
      "inserts": [{"start": 31.5, "end": 33.4, "label": "REPLAY"}],
      "cuts": [4.0, 10.0, 19.0, 27.0, 34.0]}


def _sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _events(path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("Dialogue:"):
            layer, a, b, style, _n, _l, _r, _v, _e, text = ln[len("Dialogue: "):].split(",", 9)
            out.append({"layer": int(layer), "a": _sec(a), "b": _sec(b), "style": style, "text": text})
    return out


def test_overlay_timeline(tmp_path):
    ass = tmp_path / "t.ass"
    lv._build_ass(ass, TL)
    ev = _events(ass)
    texts = " | ".join(e["text"] for e in ev)
    assert all(e["b"] <= TL["total"] + 0.01 and e["a"] >= 0 for e in ev)
    # number cards: the number the narrator says, slammed in right after each cut
    for cd in TL["cards"]:
        num = [e for e in ev if e["text"].endswith(f"}}#{cd['n']}") and e["style"] == "H"]
        assert num and num[0]["a"] == pytest.approx(cd["start"] + 0.08, abs=0.011), cd
        badge = [e for e in ev if e["text"].endswith(f"}}#{cd['n']}") and e["style"] == "HB"]
        assert badge and badge[0]["b"] <= cd["end"], "the corner badge belongs to its own moment"
    assert "KAI (CENAT) VS THE" in texts, "braces from the AI's label never become ASS override tags"
    # swipe transitions: two bands per cut, centred on the cut (whoosh-synced in render_long)
    for t in TL["cuts"]:
        bands = [e for e in ev if e["layer"] == 10 and e["a"] == pytest.approx(t - 0.2, abs=0.011)]
        assert len(bands) == 2 and all(e["b"] == pytest.approx(t + 0.2, abs=0.011) for e in bands)
    # lower thirds: platform tag + creator name, for every credited part
    for cr in TL["credits"]:
        assert any(cr["name"] in e["text"] and lv.PLATFORM_LOOK[cr["platform"]][0] in e["text"] for e in ev), cr
    # progress bar: one background segment per countdown item, only during the countdown
    bars = [e for e in ev if e["layer"] == 5]
    assert len(bars) == 3 and all(e["a"] == 10.0 and e["b"] == TL["outro_at"] for e in bars)
    # hook / preview / title card
    assert "TODAY'S #1 MOMENT" in texts and "COMING UP" in texts and "GTA VI TOP MOMENTS" in texts
    assert "TOP 3 CRAZIEST GTA 6\\NMOMENTS TODAY" in texts and "IShowSpeed, Kai" not in texts
    # end screen: everything there lasts to the very end; nothing but the end screen is drawn on it
    end = [e for e in ev if e["a"] >= TL["outro_at"] - 0.01 and e["layer"] != 10]
    assert any("WHICH ONE WAS THE BEST?" in e["text"] for e in end)
    assert any("SUBSCRIBE FOR DAILY GTA 6 MOMENTS" in e["text"] for e in end)
    assert len([e for e in end if re.search(r"\\p1", e["text"])]) == 3, "2 video boxes + 1 subscribe circle"
    assert all(e["b"] == pytest.approx(TL["total"], abs=0.011) for e in end)
    assert all(e["b"] <= TL["outro_at"] + 0.21 for e in ev if e["a"] < TL["outro_at"] - 0.25), \
        "countdown graphics end before the end screen"


def test_overlay_without_optional_parts(tmp_path):
    ass = tmp_path / "t.ass"
    lv._build_ass(ass, {"title": "GTA 6: Top 3 BEST Ways To Make Money", "total": 40.0, "outro_at": 39.0,
                        "cards": [{"start": 5.0, "end": 39.0, "n": 1, "label": "Street Races"}],
                        "progress": False, "outro_text": ""})
    ev = _events(ass)
    assert not [e for e in ev if e["layer"] in (5, 6, 10)], "no progress bar / swipes when not asked"
    assert not [e for e in ev if e["a"] >= 39.0], "a 1 s tail is too short for an end screen"


def test_chapters_follow_youtube_rules():
    cards = [{"start": 7.5, "n": 3, "label": "A", "m": {"streamer": "Kai"}},
             {"start": 31.0, "n": 2, "label": "B", "m": {"streamer": "Speed"}},
             {"start": 75.2, "n": 1, "label": "C", "m": {"streamer": "Jerma"}}]
    assert lv._chapters(cards, guide=False) == ["0:00 #3 A (Kai)", "0:31 #2 B (Speed)", "1:15 #1 C (Jerma)"], \
        "a cold open under 10 s joins the first chapter (YouTube ignores chapters shorter than 10 s)"
    cards[0]["start"] = 12.0
    assert lv._chapters(cards, guide=True)[:2] == ["0:00 Intro", "0:12 #3 A"]


def test_thumbnail_text_layout():
    for text in ("HE JUMPED?!", "EASY MONEY", "SPEED VS THE COPS", "WOW", "THE CRAZIEST GTA 6 STREAM EVER"):
        for n in (2, 3):
            lines = lv._split_lines(text, n)
            assert " ".join(lines) == text and 1 <= len(lines) <= n
    assert lv._split_lines("SPEED VS THE COPS", 2) == ["SPEED VS", "THE COPS"], "balanced lines"
    assert lv._em("WWW") > lv._em("III") > 0


@pytest.mark.slow
@needs_ffmpeg
def test_new_pieces_keep_audio_and_video_the_same_length(make_clip, tmp_path):
    src = make_clip("moment.mp4", dur=10, size="1280x720")
    hook = tmp_path / "hook.mp4"
    assert lv._segment(src, 2.0, 6.0, hook, push=0.12)
    end_pad, end_quiet = tmp_path / "end_pad.mp4", tmp_path / "end.mp4"
    assert lv._end_screen(src, 9.8, 12.0, end_pad, pad=True)  # best_t near the end: still 12 s (half speed, loop)
    assert lv._end_screen(src, 1.0, 7.0, end_quiet)
    for path, want in ((hook, 4.0), (end_pad, 12.0), (end_quiet, 7.0)):
        info = stream_info(path)
        assert (info["width"], info["height"]) == (1920, 1080)
        assert info["v_dur"] == pytest.approx(want, abs=0.04), path.name
        assert abs(info["v_dur"] - info["a_dur"]) <= 0.03, (path.name, info)


@pytest.mark.slow
@needs_ffmpeg
def test_three_thumbnail_variants(make_clip, tmp_path, sandbox):
    src = make_clip("moment.mp4", dur=6, size="1920x1080")
    cam = [0.02, 0.03, 0.25, 0.3]
    outs = [lv.thumbnail(src, 3.0, "HE JUMPED?!", tmp_path / "a.jpg", cam=cam, variant=0),
            lv.thumbnail(src, 9.0, "HE JUMPED?!", tmp_path / "b.jpg", cam=cam, variant=1),  # best_t past the end
            lv.thumbnail(src, 2.0, "CASEOH'S $1M YACHT", tmp_path / "c.jpg", variant=2, badge="#1")]
    for out in outs:
        assert out is not None and out.stat().st_size > 20_000
        info = stream_info(out)
        assert (info["width"], info["height"]) == (1280, 720)
    assert not list(lv.LONG_DIR.glob("thumb_*.txt")), "temporary text files are cleaned up"

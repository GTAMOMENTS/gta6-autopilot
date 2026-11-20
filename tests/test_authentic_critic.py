"""Authenticity gate (authentic.py, "is it really GTA 6?") and the critic's no-invention guard (critic.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

import authentic
import critic
import gemini_writer


# --------------------------------------------------------------------------- authentic.check

@pytest.fixture
def ai(monkeypatch):
    """ai(answer) -> the vision AI returns `answer`; records what it was sent."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(authentic, "frames", lambda path, n=4, width=1280: [b"\xff\xd8 fake jpeg"] * 2)
    sent = {}

    def install(answer):
        def gen(contents, temperature=0.9, prefer_pro=False):
            sent.update(contents=contents, temperature=temperature, prefer_pro=prefer_pro)
            return answer
        monkeypatch.setattr(gemini_writer, "_generate_json", gen)
        return sent
    return install


PATHS = [Path(f"/clips/c{i}.mp4") for i in range(1, 6)]


def _core(r: dict) -> dict:
    """The verdict fields (authentic.check also reports "cached" / "error")."""
    return {k: r[k] for k in ("game", "confidence", "evidence", "ok")}


def test_parses_verdicts_in_clip_order(ai, cfg):
    sent = ai({"clips": [
        {"clip": 2, "game": "GTA V", "confidence": 0.95, "evidence": "Vinewood sign"},
        {"clip": "1", "game": "GTA VI", "confidence": "0.9", "evidence": "Vice City beach"},
        {"clip": 3, "game": "GTA 6", "confidence": 0.7, "evidence": "Lucia"},
        {"clip": 4, "game": "GTA VI", "confidence": 0.5, "evidence": "maybe"},          # below min_confidence
        {"clip": "five", "game": "GTA VI", "confidence": 1},                            # unreadable number
    ]})
    out = authentic.check(PATHS)
    assert [o["ok"] for o in out] == [True, False, True, False, False]
    assert _core(out[0]) == {"game": "GTA VI", "confidence": 0.9, "evidence": "Vice City beach", "ok": True}
    assert out[1]["game"] == "GTA V"
    assert out[4]["game"] == "unsure" and out[4]["ok"] is False, "missing verdict = unsure, never a pass"
    assert cfg["authenticity"]["min_confidence"] == 0.65
    assert sent["prefer_pro"] is True and sent["temperature"] <= 0.2
    labels = [c for c in sent["contents"] if isinstance(c, str) and c.startswith("Clip ")]
    assert labels == ["Clip 1:", "Clip 2:", "Clip 3:", "Clip 4:", "Clip 5:"]
    assert "GAME FACTS" in sent["contents"][-1]


@pytest.mark.parametrize("game,ok", [("GTA VI", True), ("gta vi", True), ("GTA6", True), ("GTA 6", True),
                                     ("GTA Online", False), ("mod", False), ("other", False), ("unsure", False)])
def test_game_names(ai, game, ok):
    ai({"clips": [{"clip": 1, "game": game, "confidence": 0.99}]})
    assert authentic.check(PATHS[:1])[0]["ok"] is ok


@pytest.mark.parametrize("answer", [None, {}, {"clips": "GTA VI"}, ["not", "a", "dict"]])
def test_unusable_answer_is_none(ai, answer):
    ai(answer)
    assert authentic.check(PATHS[:2]) is None


def test_bad_confidence_value(ai):
    ai({"clips": [{"clip": 1, "game": "GTA VI", "confidence": "high"}]})
    assert _core(authentic.check(PATHS[:1])[0]) == {"game": "GTA VI", "confidence": 0.0, "evidence": "", "ok": False}


def test_no_key_or_no_paths_is_none(monkeypatch):
    assert authentic.check(PATHS) is None  # no GEMINI_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    assert authentic.check([]) is None


def test_no_frames_is_none(ai, monkeypatch):
    ai({"clips": [{"clip": 1, "game": "GTA VI", "confidence": 1}]})
    monkeypatch.setattr(authentic, "frames", lambda *a, **k: [])
    assert authentic.check(PATHS[:1]) is None


def test_keep_real_drops_fakes_and_trusts_stream_clips(monkeypatch, tmp_path):
    files = []
    for i in range(4):
        p = tmp_path / f"m{i}.mp4"
        p.write_bytes(b"x")
        files.append(p)
    clips = [{"key": "twitch:1", "path": files[0], "trusted": True},
             {"key": "yt:real", "path": files[1]}, {"key": "yt:gta5", "path": files[2]}, {"key": "yt:mod", "path": files[3]}]
    asked = []

    def fake_check(paths):
        asked.append([p.name for p in paths])
        return [{"game": "GTA VI" if p.name == "m1.mp4" else "GTA V", "confidence": 0.9, "evidence": "", "ok": p.name == "m1.mp4"}
                for p in paths]
    monkeypatch.setattr(authentic, "check", fake_check)
    kept = authentic.keep_real(clips)
    assert [m["key"] for m in kept] == ["twitch:1", "yt:real"]
    assert asked == [["m1.mp4", "m2.mp4", "m3.mp4"]], "trusted Twitch/Kick clips are not checked"
    assert not files[2].exists() and not files[3].exists(), "rejected footage is deleted"


def test_keep_real_is_strict_when_ai_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(authentic, "check", lambda paths: None)
    clips = [{"key": f"yt:{i}", "path": tmp_path / f"{i}.mp4"} for i in range(3)]
    assert authentic.keep_real(clips) == []


# --------------------------------------------------------------------------- critic guards

@pytest.mark.parametrize("field,new,known,invents", [
    ("youtube_title", "He Hit 200 MPH In GTA 6", "a car crash at 200 mph", False),
    ("youtube_title", "He Hit 300 MPH In GTA 6", "a car crash at 200 mph", True),       # new number
    ("youtube_title", "Top 5 GTA 6 Cars", "the five best cars", True),                  # digit not in source
    ("banner", "GTA 6 IS INSANE", "a chase", False),                                    # "6" is always fine
    ("caption", "Jason steals a boat at the docks.", "jason steals a boat", False),     # name is known
    ("caption", "He steals a boat from Lucia.", "he steals a boat", True),              # invented name
    ("caption", "Steals a boat. Lucia is mad.", "steals a boat", False),                # sentence start is fine
    ("youtube_title", "Lucia Steals A Boat", "a boat chase", False),                    # titles are Title Case
    ("intro_line", "Watch how GTA players escape.", "an escape", False),
    ("outro_line", "Subscribe for more Vice City chaos.", "chaos in vice city", False),
])
def test_invents(field, new, known, invents):
    assert critic._invents(field, new, known) is invents


@pytest.fixture
def critic_ai(monkeypatch):
    import ai_fallback

    def install(answer):
        monkeypatch.setattr(ai_fallback, "available", lambda: True)
        monkeypatch.setattr(ai_fallback, "chat_json", lambda prompt, temperature=0.9, frames=None: answer)
    return install


def test_polish_keeps_originals_that_break_rules(critic_ai):
    fields = {"youtube_title": "He jumped the bridge in GTA 6", "banner": "HE JUMPED", "caption": "He jumps the bridge.",
              "x_text": "Bridge jump", "intro_line": "Watch this jump."}
    critic_ai({"youtube_title": "He Jumped The Bridge At 140 MPH In GTA 6",   # invented number
               "banner": "HE JUMPED THE WHOLE BRIDGE AND LANDED ON A BOAT",  # over the limit
               "caption": "He jumps the bridge in GTA VI.",                  # says GTA VI
               "x_text": "",                                                 # empty
               "intro_line": "You will NOT believe this jump."})             # good
    out = critic.polish(fields, {"youtube_title": 70, "banner": 20, "caption": 220, "x_text": 100, "intro_line": 100},
                        "a player jumps a bridge")
    assert out["intro_line"] == "You will NOT believe this jump."
    for k in ("youtube_title", "banner", "caption", "x_text"):
        assert out[k] == fields[k], k


def test_polish_without_backup_ai_is_noop(monkeypatch):
    import ai_fallback
    monkeypatch.setattr(ai_fallback, "available", lambda: False)
    fields = {"banner": "HE JUMPED"}
    assert critic.polish(fields, {"banner": 40}, "ctx") == fields


def test_polish_disabled(cfg, critic_ai):
    cfg["critic"]["enabled"] = False
    critic_ai({"banner": "BETTER"})
    assert critic.polish({"banner": "OLD"}, {"banner": 40}, "ctx") == {"banner": "OLD"}

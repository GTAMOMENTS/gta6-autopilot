"""Surviving AI outages: an approved clip waits for a busy Gemini instead of being dropped, and the free backup
AIs stop asking models that never give a usable answer (runs of 2026-11-23 / 11-24 posted nothing)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import ai_fallback
import autopilot
import common


# --------------------------------------------------------------------------- a busy Gemini: wait, ask again

@pytest.fixture
def naps(monkeypatch):
    slept = []
    monkeypatch.setattr(autopilot.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    return slept


def answers(*seq):
    it = iter(seq)
    calls = []

    def write():
        calls.append(1)
        return next(it)
    return write, calls


def test_approved_clip_waits_for_a_busy_gemini(naps):
    write, calls = answers(None, None, {"lines": ["ok"]})
    out = autopilot.patient(write, {"ai_busy_retries": 2, "ai_busy_wait_sec": 60}, time.time() + 3600)
    assert out == {"lines": ["ok"]} and len(calls) == 3 and naps == [60.0, 60.0]


def test_an_answer_needs_no_wait(naps):
    write, calls = answers({"lines": ["ok"]})
    assert autopilot.patient(write, {}, time.time() + 3600) == {"lines": ["ok"]} and naps == [] and len(calls) == 1


def test_gives_up_after_the_retries(naps):
    write, calls = answers(None, None, None, {"never": "asked"})
    assert autopilot.patient(write, {"ai_busy_retries": 2, "ai_busy_wait_sec": 5}, time.time() + 3600) is None
    assert len(calls) == 3


def test_no_wait_without_a_key_or_time(naps, monkeypatch):
    write, calls = answers(None, {"lines": ["late"]})
    assert autopilot.patient(write, {"ai_busy_wait_sec": 60}, time.time() + 30) is None, "the run's budget is up"
    assert naps == [] and len(calls) == 1
    monkeypatch.delenv("GEMINI_API_KEY")
    write, calls = answers(None, {"lines": ["x"]})
    assert autopilot.patient(write, {}, time.time() + 3600) is None and naps == [], "no key: waiting can't help"


# --------------------------------------------------------------------------- backup AIs: two strikes

@pytest.fixture
def providers(monkeypatch):
    """One fake provider with a prose-writing model, a broken one and a good one; records every request."""
    monkeypatch.setattr(ai_fallback, "PROVIDERS", {"fake": {"key": "FAKE_KEY", "base": "https://fake.invalid/v1",
                                                            "max_images": 4}})
    monkeypatch.setattr(ai_fallback, "_model_cache", {"fake": [("prose", True), ("broken", True), ("good", True)]})
    monkeypatch.setattr(ai_fallback, "_dead", set())
    monkeypatch.setattr(ai_fallback, "_strikes", {})
    monkeypatch.setattr(ai_fallback, "_down_until", {"vision": 0.0, "text": 0.0})
    monkeypatch.setattr(ai_fallback, "_dead_vision", set())
    monkeypatch.setenv("FAKE_KEY", "x")
    common.load_config()["ai_fallback"].update(text_providers=["fake"], providers=["fake"])
    asked = []

    def post(url, headers=None, timeout=None, json=None):
        model = json["model"]
        asked.append(model)
        if model == "broken":  # what GitHub Models' retired endpoint does: 200, text/plain "OK"
            return SimpleNamespace(status_code=200, text="OK", headers={"content-type": "text/plain"},
                                   json=lambda: (_ for _ in ()).throw(ValueError("not JSON")))
        content = "The user wants me to look at the frames. Let me think..." if model == "prose" else '{"score": 8}'
        return SimpleNamespace(status_code=200, text=content, headers={"content-type": "application/json"},
                               json=lambda: {"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(ai_fallback.requests, "post", post)
    return asked


def test_models_without_a_usable_answer_are_skipped_after_two_strikes(providers):
    for _ in range(3):
        assert ai_fallback.chat_json("rate it")["score"] == 8
    assert providers == ["prose", "broken", "good", "prose", "broken", "good", "good"]


@pytest.fixture
def backups(monkeypatch):
    """Two fake providers: "slow" (its model times out) and "fast" (a text-only model that refuses images and a
    vision model). behave[model] = "timeout" / "no_images" / "ok" / "busy"; every request is recorded."""
    monkeypatch.setattr(ai_fallback, "PROVIDERS", {
        "slow": {"key": "SLOW_KEY", "base": "https://slow.invalid/v1", "max_images": 4},
        "fast": {"key": "FAST_KEY", "base": "https://fast.invalid/v1", "max_images": 8}})
    monkeypatch.setattr(ai_fallback, "_model_cache", {"slow": [("hangs", True)],
                                                      "fast": [("text-only", True), ("sees", True)]})
    for name, val in (("_dead", set()), ("_dead_vision", set()), ("_strikes", {}),
                      ("_down_until", {"vision": 0.0, "text": 0.0})):
        monkeypatch.setattr(ai_fallback, name, val)
    monkeypatch.setenv("SLOW_KEY", "x")
    monkeypatch.setenv("FAST_KEY", "x")
    common.load_config()["ai_fallback"].update(text_providers=["slow", "fast"], providers=["slow", "fast"])
    behave = {"hangs": "timeout", "text-only": "no_images", "sees": "ok"}
    asked = []

    def post(url, headers=None, timeout=None, json=None):
        model, content = json["model"], json["messages"][0]["content"]
        with_images = isinstance(content, list)
        asked.append((model, with_images, timeout))
        mode = behave[model]
        if mode == "timeout":
            raise ai_fallback.requests.Timeout("Read timed out. (read timeout=60)")
        if mode == "busy" or (mode == "no_images" and with_images):
            code = 429 if mode == "busy" else 400
            return SimpleNamespace(status_code=code, text="nope", headers={}, json=lambda: {})
        return SimpleNamespace(status_code=200, text="{}", headers={"content-type": "application/json"},
                               json=lambda: {"choices": [{"message": {"content": '{"score": 8}'}}]})
    monkeypatch.setattr(ai_fallback.requests, "post", post)
    return SimpleNamespace(asked=asked, behave=behave)


def test_a_model_that_times_out_twice_is_skipped(backups):
    """nvidia's gemma-4 timed out after 90 s seven times and the chain ran out of time before models that answer."""
    for _ in range(3):
        assert ai_fallback.chat_json("rate it")["score"] == 8
    assert [m for m, _, _ in backups.asked].count("hangs") == 2
    assert all(t <= common.load_config()["ai_fallback"]["call_timeout_sec"] for _, _, t in backups.asked)


def test_a_text_model_that_refuses_images_still_answers_text(backups):
    """cerebras/gpt-oss-120b (pinned = assumed to see images) got frames, answered 400 and was out for the run -
    including for the script judge, its main job."""
    assert ai_fallback.chat_json("look", frames=["aW1n"])["_ai_model"] == "fast/sees"
    assert ai_fallback.chat_json("judge it")["_ai_model"] == "fast/text-only"
    assert ai_fallback.chat_json("look again", frames=["aW1n"])["_ai_model"] == "fast/sees"
    assert ("text-only", True) not in [(m, i) for m, i, _ in backups.asked[3:]], "not sent images again"


def test_a_failed_frames_chain_does_not_switch_off_text_jobs(backups):
    backups.behave.update(sees="busy")
    assert ai_fallback.chat_json("look", frames=["aW1n"]) is None
    assert ai_fallback.chat_json("look", frames=["aW1n"]) is None and len(backups.asked) == 3, "frames jobs rest"
    assert ai_fallback.chat_json("judge it")["_ai_model"] == "fast/text-only", "the judge still has its AIs"


def test_all_frames_skips_providers_that_take_fewer_images(backups):
    """The script backup numbers every frame - a provider that keeps only the first 4 would see the wrong ones."""
    backups.behave.update(hangs="ok")
    out = ai_fallback.chat_json("write", frames=["aW1n"] * 6, all_frames=True)
    assert out["_ai_model"] == "fast/sees" and "hangs" not in [m for m, _, _ in backups.asked]


def test_pinned_models_see_images_only_if_the_name_says_so(monkeypatch):
    monkeypatch.setattr(ai_fallback, "_model_cache", {})
    monkeypatch.setitem(common.load_config()["ai_fallback"]["models"], "sambanova", ["gpt-oss-120b", "gemma-4-31B-it"])
    monkeypatch.setattr(ai_fallback.requests, "get", lambda *a, **k: (_ for _ in ()).throw(ValueError("offline")))
    monkeypatch.setenv("SAMBANOVA_API_KEY", "x")
    assert dict(ai_fallback._models("sambanova")) == {"gpt-oss-120b": False, "gemma-4-31B-it": True}


# --------------------------------------------------------------------------- the script writer without Gemini

def test_gemini_busy_a_backup_writes_the_script_from_frames(monkeypatch, make_clip):
    """Every narrated Short goes through write_final_script, which had no backup: with Gemini out of quota the
    clip was dropped even when the backup AIs were up."""
    import gemini_writer
    clip = make_clip("tour.mp4", dur=40)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: object())
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: None)  # busy / out of quota
    monkeypatch.setattr(ai_fallback, "available", lambda: True)
    seen = {}
    slots = [{"t": 1.0 + 3.5 * i, "text": f"old line {i}"} for i in range(10)]

    def backup(prompt, frames=None, temperature=0.9, all_frames=False):
        seen.update(prompt=prompt, frames=frames, all_frames=all_frames)
        return {"story": "a chase", "lines": [{"t": s["t"], "text": f"New line number {i} about the chase."}
                                              for i, s in enumerate(slots)], "_ai_model": "fake/vision"}
    monkeypatch.setattr(ai_fallback, "chat_json", backup)
    out = gemini_writer.write_final_script(clip, {"banner": "CHASE"}, slots=slots)
    assert out and out["model"] == "fake/vision" and len(out["lines"]) == 10
    assert [ln["t"] for ln in out["lines"]] == [s["t"] for s in slots], "a fixed edit keeps its times"
    assert seen["prompt"].startswith("YOU CANNOT WATCH OR HEAR THIS VIDEO") and seen["all_frames"]
    assert len(seen["frames"]) == 8 and "image 1 = slot 1 (2.0s)" in seen["prompt"]
    assert "image 8 = slot 10" in seen["prompt"], "10 slots -> 8 frames spread over them, first and last kept"


def test_gemini_answers_no_backup_is_asked(monkeypatch, make_clip):
    import gemini_writer
    clip = make_clip("cut.mp4", dur=20)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: object())
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: {"story": "s", "lines": [
        {"t": 0.5 + 4 * i, "text": f"A line about the chase number {i}."} for i in range(4)]})
    monkeypatch.setattr(ai_fallback, "chat_json", lambda *a, **k: pytest.fail("Gemini answered"))
    out = gemini_writer.write_final_script(clip, {"banner": "CHASE"})
    assert out and out["model"] == "gemini-video"


def test_github_models_is_gone():
    """GitHub retired GitHub Models on 2026-07-30 - its endpoint answers every request with a plain "OK"."""
    cfg = common.load_config()["ai_fallback"]
    assert "github" not in ai_fallback.PROVIDERS
    assert "github" not in cfg["providers"] and "github" not in cfg["text_providers"]


# --------------------------------------------------------------------------- tours: no shots where nothing happens

def test_calm_shots_are_dropped_from_a_tour():
    """Owner: "you can't do narration on a walking clip". A posted Short opened on a stroll past tourists."""
    energy = [0.1] * 10 + [0.9] * 10 + [0.15] * 10 + [0.8] * 30  # walking, chase, walking, crash...
    shots = [{"start": s, "text": f"shot {s}"} for s in (1, 12, 22, 31, 40, 50)]
    kept = autopilot.drop_calm_shots(shots, energy, calm=0.25, keep_min=4)
    assert [sh["start"] for sh in kept] == [12, 31, 40, 50], "both walking shots go, the order stays"
    kept = autopilot.drop_calm_shots(shots, energy, calm=0.25, keep_min=5)
    assert [sh["start"] for sh in kept] == [12, 22, 31, 40, 50], "only as many as keep the minimum: calmest first"
    assert autopilot.drop_calm_shots(shots, [], calm=0.25, keep_min=4) == shots, "no energy data: nothing dropped"


def test_the_editor_is_told_no_walking_shots():
    import gemini_writer
    assert "NEVER a shot of someone just walking" in gemini_writer.TOUR_PROMPT

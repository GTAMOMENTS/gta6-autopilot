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
    monkeypatch.setattr(ai_fallback, "_down_until", 0.0)
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


def test_github_models_is_gone():
    """GitHub retired GitHub Models on 2026-07-30 - its endpoint answers every request with a plain "OK"."""
    cfg = common.load_config()["ai_fallback"]
    assert "github" not in ai_fallback.PROVIDERS
    assert "github" not in cfg["providers"] and "github" not in cfg["text_providers"]

"""Schedule: every GitHub cron in main.yml must land on the right slot of config.json (TRT = UTC+3)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import autopilot

REPO = Path(__file__).resolve().parents[1]
MAIN_YML = REPO / ".github" / "workflows" / "main.yml"

# The owner's plan (HANDOFF.md section 2 / 8): UTC cron -> slot type
OWNER_PLAN = [
    ("10 12 * * *", "long_video"),            # 15:10 TRT
    ("50 13 * * *", "x_text_morning"),        # 16:50 TRT
    ("40 16 * * *", "vertical_clip"),         # 19:40 TRT  Video A
    ("10 19 * * *", "x_question_nostalgia"),  # 22:10 TRT
    ("40 22 * * *", "video_b"),               # 01:40 TRT  Video B (next day in Turkey)
    ("20 1 * * *", "x_daily_recap"),          # 04:20 TRT
]


def workflow_crons() -> list[str]:
    return re.findall(r"^\s*-\s*cron:\s*['\"]([^'\"]+)['\"]", MAIN_YML.read_text(encoding="utf-8"), re.M)


def utc_cron_for(local_hhmm: str, tz: str) -> str:
    hh, mm = map(int, local_hhmm.split(":"))
    local = datetime(2026, 11, 20, hh, mm, tzinfo=ZoneInfo(tz))
    utc = local.astimezone(ZoneInfo("UTC"))
    return f"{utc.minute} {utc.hour} * * *"


@pytest.mark.parametrize("cron,slot_type", OWNER_PLAN)
def test_owner_plan_cron_maps_to_slot(cfg, cron, slot_type):
    slot = autopilot._slot_from_cron(cron, cfg["schedule"])
    assert slot is not None, f"cron {cron!r} matches no slot in config.json"
    assert slot["type"] == slot_type


def test_every_workflow_cron_maps_to_a_slot(cfg):
    crons = workflow_crons()
    assert crons, "no cron lines found in main.yml"
    for cron in crons:
        slot = autopilot._slot_from_cron(cron, cfg["schedule"])
        assert slot is not None, f"main.yml cron {cron!r} matches no slot in config.json 'schedule'"


def test_every_enabled_slot_has_a_workflow_cron(cfg):
    crons = set(workflow_crons())
    for slot in cfg["schedule"]:
        if not slot.get("enabled", True):
            continue
        want = utc_cron_for(slot["time"], cfg["timezone"])
        assert want in crons, f"slot {slot['type']} at {slot['time']} TRT needs cron '{want}' in main.yml"


def test_workflow_crons_match_owner_plan():
    assert sorted(workflow_crons()) == sorted(c for c, _ in OWNER_PLAN)


def test_slot_types_are_unique(cfg):
    types = [s["type"] for s in cfg["schedule"]]
    assert len(types) == len(set(types)), types


@pytest.mark.parametrize("bad", ["", "garbage", "61", "* * * * *"])
def test_bad_cron_matches_nothing(cfg, bad):
    assert autopilot._slot_from_cron(bad, cfg["schedule"]) is None


def test_unknown_cron_time_matches_nothing(cfg):
    assert autopilot._slot_from_cron("7 3 * * *", cfg["schedule"]) is None


def test_resolve_slot_uses_schedule_cron(cfg, monkeypatch):
    monkeypatch.setenv("SCHEDULE_CRON", "40 22 * * *")
    slot = autopilot.resolve_slot(cfg, "auto")
    assert slot["type"] == "video_b" and not slot.get("format"), "Video B = normal rotation (money dropped)"
    assert not slot.get("manual")


def test_resolve_slot_manual_request(cfg):
    slot = autopilot.resolve_slot(cfg, "long_video")
    assert slot["type"] == "long_video" and slot["manual"] is True
    assert autopilot.resolve_slot(cfg, "x_something_new")["platforms"] == ["x"]
    assert autopilot.resolve_slot(cfg, "no_such_slot") is None


def test_video_slots_routes(cfg):
    slots = {s["type"]: s for s in cfg["schedule"]}
    assert slots["long_video"]["platforms"] == ["youtube"]
    assert slots["long_video"].get("every_days") == 3
    assert "format" not in slots["video_b"], "money dropped 2026-11-21: Video B uses the rotation"
    for t in ("x_text_morning", "x_question_nostalgia", "x_daily_recap"):
        assert slots[t]["platforms"] == ["x"]


def test_nearest_due_slot_window(cfg):
    slots = cfg["schedule"]
    tz = ZoneInfo(cfg["timezone"])
    late = datetime(2026, 11, 21, 20, 30, tzinfo=tz)  # 50 min after Video A (19:40)
    assert autopilot._nearest_due_slot(slots, late)["type"] == "vertical_clip"
    lonely = datetime(2026, 11, 21, 10, 0, tzinfo=tz)  # > 150 min after 04:20, before 15:10
    assert autopilot._nearest_due_slot(slots, lonely) is None

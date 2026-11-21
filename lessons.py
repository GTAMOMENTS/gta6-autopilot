"""Mistakes memory: the system learns from its own QA results (the AI models don't learn from our runs by
themselves - so the learning lives here).

  record(...)        every QA result (pass / fail, issue types, concrete examples, format, source)
  prompt_hints(...)  recent recurring mistakes + good examples -> added to the script / rewrite prompts, so the
                     writer stops repeating them
  should_rest(...)   formats that keep failing QA are rested for a while
  stats_line()       first-pass rate and attempts per published video (logged every run)

Stored in queue/qa_lessons.json (saved by the Actions state cache after real runs).
"""
from __future__ import annotations

import logging
import time

from common import STATE_DIR, load_config, load_json, save_json

log = logging.getLogger("autopilot")
LESSONS_FILE = STATE_DIR / "qa_lessons.json"
SCRIPT_TYPES = {"line_not_shown", "voice_mismatch"}  # mistakes the writer can avoid (not footage / tech problems)


def _cfg() -> dict:
    return load_config().get("lessons", {})


def _load() -> dict:
    return load_json(LESSONS_FILE, {"attempts": [], "good": {}})


def record(kind: str, fmt_id: str, passed: bool, issues: list[dict], stage: str = "final",
           source: dict | None = None, lines: list[str] | None = None, repaired: bool = False) -> None:
    """One QA result. stage: final (after render) | precheck (before voice / render) | repair."""
    if not _cfg().get("enabled", True):
        return
    d = _load()
    high = [i for i in issues if str(i.get("severity")) == "high"]
    examples = [str(i.get("detail", ""))[:220] for i in high if i.get("type") in SCRIPT_TYPES][:3]
    d["attempts"] = (d.get("attempts", []) + [{
        "at": time.time(), "kind": kind, "fmt": fmt_id or "", "pass": bool(passed), "stage": stage,
        "types": sorted({str(i.get("type")) for i in high}), "examples": examples, "repaired": repaired,
        "source": {k: (source or {}).get(k, "") for k in ("provider", "channel")}}])[-400:]
    if passed and lines and stage in ("final", "repair"):
        good = d.setdefault("good", {}).setdefault(fmt_id or "_", [])
        good.append([str(x)[:140] for x in lines[:8]])
        d["good"][fmt_id or "_"] = good[-6:]
    save_json(LESSONS_FILE, d)


def _recent(days: float) -> list[dict]:
    cut = time.time() - days * 86400
    return [a for a in _load().get("attempts", []) if a.get("at", 0) >= cut]


def prompt_hints(fmt_id: str | None = None, kind: str = "short") -> str:
    """Short block for the writer's prompt: recent recurring script mistakes (with real examples) + lines that
    passed QA for this format. Empty when there is nothing to say yet."""
    c = _cfg()
    if not c.get("enabled", True) or not c.get("prompt_hints", True):
        return ""
    recent = [a for a in _recent(float(c.get("days", 14))) if a.get("kind") == kind]
    counts: dict[str, int] = {}
    examples: list[str] = []
    for a in reversed(recent):
        for t in a.get("types", []):
            counts[t] = counts.get(t, 0) + 1
        if len(examples) < 3:
            examples += [e for e in a.get("examples", []) if e not in examples][: 3 - len(examples)]
    parts = []
    if counts.get("line_not_shown") or examples:
        parts.append("RECENT MISTAKES TO AVOID (from our own QA): narration that describes something the picture "
                     "doesn't show at that moment. Every line must describe what is visible WHILE it is spoken.")
        parts += [f"- {e}" for e in examples]
    if any(t.startswith("countdown") for t in counts):
        parts.append("- a countdown must announce every number from N down to 1, one per item")
    good = (_load().get("good", {}) or {}).get(fmt_id or "_", [])[-2:]
    if good:
        parts.append("Lines that passed QA for this format (match the picture this closely, don't copy them):")
        parts += ["- " + " / ".join(g[:4]) for g in good]
    return "\n".join(parts)


def pass_rates(days: float = 10, kind: str = "short") -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for a in _recent(days):
        if a.get("kind") != kind or a.get("stage") == "precheck":
            continue
        p, n = out.get(a.get("fmt", ""), (0, 0))
        out[a.get("fmt", "")] = (p + (1 if a.get("pass") else 0), n + 1)
    return out


def should_rest(fmt_id: str) -> bool:
    """A format that keeps failing QA (>= lessons.rest_min_attempts finished videos in the last days, pass rate below
    lessons.rest_below) is rested - another format is made instead."""
    c = _cfg()
    p, n = pass_rates(float(c.get("rest_days", 7))).get(fmt_id, (0, 0))
    return n >= int(c.get("rest_min_attempts", 6)) and p / n < float(c.get("rest_below", 0.2))


def stats_line(days: float = 7) -> str:
    rec = [a for a in _recent(days) if a.get("kind") == "short" and a.get("stage") in ("final", "repair")]
    if not rec:
        return "no QA history yet"
    first = [a for a in rec if a.get("stage") == "final"]
    passed_first = sum(1 for a in first if a.get("pass"))
    published = sum(1 for a in rec if a.get("pass"))
    repaired = sum(1 for a in rec if a.get("pass") and a.get("repaired"))
    return (f"first-pass rate {passed_first}/{len(first)} ({100 * passed_first / max(1, len(first)):.0f}%), "
            f"{published} passed ({repaired} after repair), {len(rec) / max(1, published):.1f} finished videos per "
            f"published one - last {days:.0f} days")

"""Script judge: reads the WHOLE narration like a viewer and decides if it is good enough to post.

The owner (2026-11-22) on two posted Shorts that QA scored 9/10: "the script is so inconsistent, doesn't make sense,
starts with how to steal a car yet it never steals it, unfunny, not compelling - 0/10". The QA editor judged the
picture and the edit, never the story. This judge asks only about the script:
  - does it make sense from start to finish (one story, each line follows the last)?
  - does the video deliver what the hook / banner / title promise?
  - is it entertaining (funny / hyped / useful) - or a dry list of what is on screen?
  - plus certain checks with no AI: broken numbering, fake first person ("I tried..."), the same word in every line,
    no closing question.
A different AI from the writer judges (text models first: no Gemini quota, and no writer judging its own work).
"""
from __future__ import annotations

import logging
import re

from common import load_config

log = logging.getLogger("autopilot")

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10}
STOP = set("""this that with from they them their there then than what when where which while your about into over
just like have been were will would could should these those here only also very really still even back down
going gonna watch look gets okay right after before every some more most much make makes made guy guys dude""".split())
NAMES = {"gta", "jason", "lucia", "leonida"}

PROMPT = """You are the strict head of content of a GTA 6 YouTube Shorts / TikTok channel with millions of views.
Judge this narration script AS A VIEWER WHO HEARS IT. GAME FACTS: {facts}

Format: {fmt}
Banner on screen: "{banner}" | Title: "{title}"
What really happens in the video (from the AI that watched it): {story}
Script (seconds: line):
{script}
Problems found automatically: {static}

CHANNEL RULES - correct by design, never count them as problems:
- the last line is a short question to the viewer (required on every video)
- {structure}
- the format name (e.g. "Top 5 funniest moments") is the THEME, not a demand for numbers or an exact item count
- naming places / characters / game features is optional

Score it 1-10 (7 = good enough for a channel with millions of views; real top creators are 8-9):
- does it make sense from start to finish? does each line follow from the line before (a run of jokes about one
  character's chaos counts as one story)?
- does the video deliver what the first line / banner / title promise? (promising "how to steal a car" when no car
  gets stolen = 2 at most)
- is it entertaining - funny, surprising, hyped, or genuinely useful - or a dry list of what is on screen / filler?
  (a list of descriptions = 3 at most)
- natural English, no repeated words in every line, no fake first person ("I tried"), numbering correct (if any)

CALIBRATION (score like this):
- 2/10 (dry, repetitive, fake "I", no story): "I tried stealing this dirt bike right from the NPC and it worked in GTA
  6! / This dirt bike handles the track jumps surprisingly well right off the line. / The suspension on this dirt bike
  handles the dirt track bumps so smoothly. / Would you try this dirt track trial or stick to sports cars?"
- 3/10 (hook never delivered, broken numbering, captions of frames): "You will not believe how easy it is to steal a
  cop car in GTA 6. / Number two: Opening the car trunk to check stored gear. / Number seven: Running across the sunlit
  road toward a parked pickup truck. / Would you risk your cash to save a stolen ride?"
- 8/10 (real creator: every line a joke or a reaction, about what happens, builds to a payoff): "This man walked up
  to the bike like he was the main character. / He was not. Ten out of ten style, zero out of ten landing. / Then he
  picks a fight with an SUV. The SUV did not lose. / Supercar versus palm tree... the tree remains undefeated. / Could
  you survive five minutes in Vice City without becoming a cartoon?"
Return ONLY JSON: {{"score": number, "makes_sense": bool, "promise_kept": bool, "entertaining": bool,
"problems": [str], "best_fix": str}}"""


def _cfg() -> dict:
    return load_config().get("script_judge", {})


def _num(tok: str) -> int | None:
    tok = tok.lower().strip(":.,")
    return int(tok) if tok.isdigit() else NUM_WORDS.get(tok)


def dead_air(lines: list[dict], length: float, lead: float = 0.0, spans=None, wps: float = 2.7) -> list[str]:
    """Long stretches where nobody talks (no narration and no game dialogue) - real creators talk almost all the
    time. A dry run's script left 12 s and 14 s of silence."""
    c = _cfg()
    max_gap = float(c.get("max_silence_sec", 6.0))
    busy = sorted([(float(ln["t"]), float(ln["t"]) + len(str(ln["text"]).split()) / wps) for ln in lines] +
                  [(float(a), float(b)) for a, b in (spans or [])])
    out, cursor = [], lead
    for a, b in busy + [(length - 0.5, length)]:
        if a - cursor > max_gap:
            out.append(f"{a - cursor:.0f} s of silence ({cursor:.0f}-{a:.0f}s) - fill it with a line about what "
                       "happens there")
        cursor = max(cursor, b)
    talk = sum(len(str(ln["text"]).split()) for ln in lines) / wps
    room = max(1.0, length - lead - sum(max(0.0, float(b) - float(a)) for a, b in (spans or [])))
    if talk < float(c.get("min_coverage", 0.6)) * room:
        out.append(f"the narrator talks only {talk:.0f} of {room:.0f} quiet seconds - write more (and longer) lines")
    return out[:3]


def quiet_tail(lines: list[dict], length: float, lead: float = 0.0, spans=None, wps: float = 2.7) -> float | None:
    """When the script's only long silence comes right before its closing line: the second the talking stops. The
    writer watched the video and had nothing to say there because nothing happens (a 34 s crash clip: the rider
    walks around from 15 s on - 9 scripts in a row left 10-17 s of silence there). The cut can end there instead
    (autopilot.produce). None = no such tail (a gap in the middle is a real script problem)."""
    if len(lines) < 3:
        return None
    max_gap = float(_cfg().get("max_silence_sec", 6.0))
    busy = sorted([(float(ln["t"]), float(ln["t"]) + len(str(ln["text"]).split()) / wps) for ln in lines[:-1]] +
                  [(float(a), float(b)) for a, b in (spans or [])])
    cursor = lead
    for a, b in busy:
        if a - cursor > max_gap:
            return None
        cursor = max(cursor, b)
    closing = min(float(lines[-1]["t"]), length)
    return cursor if closing - cursor > max_gap else None


def static_problems(lines: list[str], numbered_ok: bool = False, countdown: int | None = None) -> list[str]:
    """Certain problems, no AI needed."""
    out = []
    texts = [str(x).strip() for x in lines if str(x).strip()]
    if not texts:
        return ["empty script"]
    nums = []
    for t in texts:
        m = re.match(r"\s*(?:number|no\.?|#)\s*(\w+)", t, re.I)
        if m and _num(m.group(1)) is not None:
            nums.append(_num(m.group(1)))
    if nums:
        if not numbered_ok:
            out.append("numbered items over ONE continuous scene (it is not a list) - tell it as a story")
        else:
            n = countdown or max(nums)
            if sorted(nums) != list(range(1, n + 1)):
                out.append(f"broken numbering {nums}: a countdown must say every number from {n} down to 1 once")
            elif nums != sorted(nums, reverse=True):
                out.append(f"numbering out of order {nums}: count DOWN to number one")
    fp = [t for t in texts if re.search(r"\b(I|I'm|I've|I'll|I'd|my)\b", re.sub(r"(?i)\boh my\b", "", t))]
    if fp:
        out.append(f'fake first person ("{fp[0][:50]}"): the narrator talks ABOUT Jason / Lucia / the player')
    if len(texts) >= 4:
        counts: dict[str, int] = {}
        for t in texts:
            for w in set(re.findall(r"[a-z']{4,}", t.lower())):
                if w not in STOP and w not in NAMES and w != "number" and w not in NUM_WORDS:
                    counts[w] = counts.get(w, 0) + 1
        rep = sorted((n, w) for w, n in counts.items() if n >= max(3, round(len(texts) * 0.65)))
        if rep:
            out.append(f'"{rep[-1][1]}" in {rep[-1][0]} of {len(texts)} lines - vary the words')
    last = texts[-1].rstrip()
    max_close = int(_cfg().get("max_closing_words", 14))
    if not last.endswith("?"):
        out.append("no closing question to the viewer")
    elif not numbered_ok and len(last.split()) > max_close:
        # a posted Short ended on "Bro swaps for a delivery van and tears straight across the tarmac while dodging
        # cruisers, but would you survive this airport chase?" - the question tacked onto a 22-word line
        out.append(f"the closing question is buried in a {len(last.split())}-word line - end with ONE short question "
                   f"to the viewer (max {max_close} words), nothing else in that line")
    return out


def judge(lines: list[dict], ctx: dict, numbered_ok: bool = False, countdown: int | None = None,
          timing: dict | None = None) -> dict:
    """{"ok", "score", "problems", "by", "static"}. ctx: fmt, banner, title, story. ok=False when the script is not good
    enough - or when no AI could judge it and script_judge.required is on (strict)."""
    import ai_fallback
    import gemini_writer
    c = _cfg()
    texts = [ln["text"] for ln in lines]
    static = static_problems(texts, numbered_ok, countdown)
    if timing and timing.get("length"):  # timing = {"length", "lead", "spans"} (not for fixed tour slots)
        static += dead_air(lines, float(timing["length"]), float(timing.get("lead", 0)), timing.get("spans"))
    prompt = PROMPT.format(facts=gemini_writer._game_context(), fmt=ctx.get("fmt", "GTA 6 Short"),
                           banner=ctx.get("banner", ""), title=ctx.get("title", ""),
                           story=ctx.get("story") or "(unknown)",
                           script="\n".join(f'{float(ln.get("t", 0)):.1f}s: {ln["text"]}' for ln in lines),
                           static="; ".join(static) or "none",
                           structure=("this is a countdown: the numbered items must be complete and in order" if numbered_ok
                                      else "this is ONE edit told as a story or a run of jokes - it is correct that "
                                           "nothing is numbered"))
    data, by = None, ""
    if ai_fallback.available():
        data = ai_fallback.chat_json(prompt, temperature=0.2)
        by = (data or {}).get("_ai_model", "")
    if not isinstance(data, dict) or "score" not in data:
        data = gemini_writer._generate_json(prompt, temperature=0.2) if not data else data
        by = by or "gemini"
    need = float(c.get("min_score", 7))
    if not isinstance(data, dict) or "score" not in data:
        log.warning("[judge] no AI could judge the script")
        return {"ok": not c.get("required", True) and not static, "score": None, "problems": static,
                "by": "", "static": static}
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        score = 0.0
    problems = [str(p)[:200] for p in (data.get("problems") or []) if str(p).strip()][:6]
    if data.get("best_fix"):
        problems.append("fix: " + str(data["best_fix"])[:200])
    hard = data.get("makes_sense") is False or data.get("promise_kept") is False
    ok = score >= need and not static and not hard
    log.info("[judge] script %.0f/10 by %s%s%s", score, by or "?", " ✅" if ok else " ✋",
             (" | " + "; ".join(static + problems))[:400] if not ok else "")
    return {"ok": ok, "score": score, "problems": static + problems, "by": by, "static": static}


def feedback(verdict: dict) -> str:
    """Tell the writer why its script was rejected."""
    return ("YOUR LAST SCRIPT WAS REJECTED by the head of content"
            + (f" ({verdict['score']:.0f}/10)" if verdict.get("score") is not None else "") + ":\n"
            + "\n".join(f"- {p}" for p in verdict.get("problems", [])[:8])
            + "\nWrite a NEW, much better script for the whole video - don't patch the old one.")

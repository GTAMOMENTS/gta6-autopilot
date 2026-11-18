"""All AI writing: clip verification + voiceover script + captions + X text posts.

Gemini WATCHES the actual clip (uploaded as a small 480p proxy) so the script
describes what is really on screen, and it also acts as a quality gate:
it rejects GTA V / mods / fan concepts / facecam streams / boring footage.

Robustness:
  * hard request timeout + only 2 retries per model (the old code hung 8 min on a 503)
  * falls through a list of models (config.json -> gemini.models)
  * if every model fails, a large randomised fallback pool is used so videos
    never repeat the exact same script
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import date
from pathlib import Path

import ai_fallback
from common import WORK_DIR, env, load_config, now_local, parse_json_loose, run

log = logging.getLogger("autopilot")

_client_obj = None
_unavailable_until = 0.0  # skip Gemini for the rest of the run after repeated overloads


# --------------------------------------------------------------------------- client

def _client():
    global _client_obj
    if _client_obj is None:
        from google import genai
        from google.genai import types
        timeout_ms = int(load_config().get("gemini", {}).get("timeout_sec", 50)) * 1000
        _client_obj = genai.Client(
            api_key=env("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                # attempts=1 -> never retry/wait on the same model; we move to the next model instead
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _client_obj


_dead_models: set[str] = set()
_discovered: list[str] | None = None


def _version_key(name: str) -> tuple:
    nums = re.findall(r"\d+(?:\.\d+)?", name)
    v = float(nums[0]) if nums else 0.0
    lite = "lite" in name
    preview = "preview" in name or "exp" in name
    return (v, not lite, not preview)


def _discover_models() -> list[str]:
    """Ask the API which Flash models this key can actually call (names change often)."""
    global _discovered
    if _discovered is None:
        _discovered = []
        try:
            names = []
            for m in _client().models.list():
                name = (m.name or "").removeprefix("models/")
                actions = [a.lower() for a in (getattr(m, "supported_actions", None) or [])]
                if actions and not any("generatecontent" in a for a in actions):
                    continue
                if "flash" not in name or any(x in name for x in
                                              ("image", "tts", "audio", "live", "embedding", "native", "robotics")):
                    continue
                names.append(name)
            _discovered = sorted(set(names), key=_version_key, reverse=True)[:3]
            log.info("[gemini] models available to this key: %s", _discovered)
        except Exception as exc:  # noqa: BLE001
            log.warning("[gemini] could not list models: %.200s", exc)
    return _discovered


_client_beta_obj = None


def _client_beta():
    """Same key, but the v1beta API - newer features like text-to-speech models live there."""
    global _client_beta_obj
    if _client_beta_obj is None:
        from google import genai
        from google.genai import types
        _client_beta_obj = genai.Client(
            api_key=env("GEMINI_API_KEY"),
            http_options=types.HttpOptions(api_version="v1beta", timeout=60_000,
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )
    return _client_beta_obj


def _models() -> list[str]:
    cfg = load_config().get("gemini", {})
    models = [env("GEMINI_MODEL")] + cfg.get("models", []) + _discover_models()
    return [m for m in dict.fromkeys(models) if m and m not in _dead_models]


def _parse_json(text: str) -> dict | None:
    return parse_json_loose(text)


def _generate_json(contents, temperature: float = 0.9) -> dict | None:
    """Try each Gemini model ONCE. Busy / out of quota / error -> next model immediately.
    If every model fails, Gemini is skipped for the rest of this run (backup AIs take over)."""
    global _unavailable_until
    if not env("GEMINI_API_KEY"):
        log.warning("[gemini] GEMINI_API_KEY not set - using backup AI")
        return None
    if time.time() < _unavailable_until:
        return None
    from google.genai import types
    for model in _models():
        try:
            resp = _client().models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            data = _parse_json(resp.text or "")
            if data:
                log.info("[gemini] response from %s", model)
                return data
            log.warning("[gemini] %s returned unparseable output - next model", model)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            log.warning("[gemini] %s failed (%.120s) - next model", model, msg)
            if any(x in msg for x in ("404", "NOT_FOUND", "not supported", "exceeded your current quota",
                                      "RESOURCE_EXHAUSTED")):
                _dead_models.add(model)  # gone or out of quota for today - don't try it again this run
    log.warning("[gemini] all models busy/out of quota - skipping Gemini for the rest of this run")
    _unavailable_until = time.time() + 3600
    return None


def _upload_video(clip: Path):
    """Upload a light 480p proxy of the clip to the Gemini Files API."""
    proxy = WORK_DIR / "gemini_proxy.mp4"
    res = run(["ffmpeg", "-y", "-i", str(clip), "-vf", "scale=-2:480,fps=12",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
               "-c:a", "aac", "-b:a", "48k", "-ac", "1", str(proxy)], timeout=240)
    if res.returncode != 0:
        return None
    client = _client()
    f = client.files.upload(file=str(proxy), config={"mime_type": "video/mp4"})
    deadline = time.time() + 120
    while getattr(f.state, "name", str(f.state)) == "PROCESSING" and time.time() < deadline:
        time.sleep(4)
        f = client.files.get(name=f.name)
    state = getattr(f.state, "name", str(f.state))
    if state != "ACTIVE":
        log.warning("[gemini] uploaded video not ready (state=%s)", state)
        return None
    return f


# --------------------------------------------------------------------------- context

def _game_context() -> str:
    cfg = load_config()
    g = cfg.get("game", {})
    today = now_local().date()
    try:
        launch = date.fromisoformat(g.get("launch_date", "2026-11-19"))
    except ValueError:
        launch = today
    days = (today - launch).days
    if days < 0:
        status = f"GTA VI launches in {-days} day(s), on {launch:%B %d, %Y}."
    elif days == 0:
        status = "TODAY is GTA VI launch day!"
    else:
        status = f"GTA VI has been out for {days} day(s) (launched {launch:%B %d, %Y})."
    return f"Today is {today:%A, %B %d, %Y}. {status} {g.get('facts', '')}".strip()


# --------------------------------------------------------------------------- clip copy

CLIP_PROMPT = """You are the editor AND writer of "{brand}", a viral faceless channel posting the most gripping
Grand Theft Auto 6 moments as vertical Shorts/Reels/TikToks. Viewers decide in 2 seconds whether to keep watching.

CONTEXT: {context}
Source footage title: "{title}" (credit: {credit}). Source type: {provider}.
{video_note}
{kind_note}

TASKS
1. QUALITY GATE - judge honestly:
   - is_gta6_footage: true ONLY for real Grand Theft Auto VI footage (Leonida / Vice City, Jason & Lucia,
     GTA VI HUD). false for GTA V / GTA Online, mods, other games, fan "concept" trailers, AI fakes,
     real-life videos, news anchors, slideshows or menus.
   - has_facecam: true if a streamer webcam / face overlay or a big channel watermark covers the video.
   - has_narrator: true if a YouTuber / commentator talks over the footage in their own voice
     (NOT game characters, NOT music). Only judge this if you can hear the audio; else false.
   - hype_score: 1-10 for the BEST part you pick in step 2, judged like a strict viral-shorts editor:
     9-10 = jaw-dropping (insane crash, wild chase, shootout, huge stunt, shocking story twist, hilarious chaos)
     7-8 = clearly exciting or intriguing; 5-6 = okay but would be scrolled past;
     1-4 = dialogue, walking, menus, loading, calm driving. Be strict - most footage is 4-6.
   - scene_summary: one sentence on exactly what happens in the part you pick.
2. EDIT - pick the single best continuous part, {min_len} to {max_len} seconds long:
   - clip_start / clip_end in seconds of THIS footage ({duration} s long).
   - Start 1-2 seconds BEFORE the action kicks in, so the first seconds already hook. End right after the
     payoff (crash, escape, punchline, reveal). Skip dialogue-only, walking and loading parts.
   - Motion/loudness analysis says the most intense parts are around: {hints}. Use it as a hint only.
3. WRITE for the part you picked - an excited, fun GTA fan talking to his friends, NOT a documentary narrator.
   content_type (one of):
     "explainer" = a secret, easter egg, hidden detail, new mechanic or story moment that needs context
     "action"    = chase, shootout, crash, stunt, heist, chaos
     "scenic"    = views, cruising, city life, mostly calm
   How much the voice talks depends on content_type:
     * explainer: 4 to 7 lines that explain what the viewer is looking at and why it's cool, right when it
       appears (max 20 words per line, under 80 words total).
     * action: 3 or 4 short reactions at the craziest moments (max 8 words each).
     * scenic: ONLY 2 lines - the hook and a question at the end.
   - banner: max 6 words, ALL CAPS, makes people NEED to see what happens. No emojis.
   - commentary: list of {{"t": second of THIS footage, "line": "words"}}, every t between clip_start and
     clip_end. Line 1 at t=clip_start is the HOOK: max 8 words, creates curiosity or tension about what is
     about to happen, e.g. "He had no idea what was coming...", "This chase went SO wrong", "Watch Lucia's
     face right here". The last line, about 4 seconds before clip_end, is a short question or call to action.
     At least 3 seconds between lines.
   - Every line must be about something SPECIFIC in this clip. BANNED filler: "Rockstar cooked", "this game
     is insane", "the detail is unreal", "look at that", "wait for it", "next level" - unless tied to a
     concrete thing on screen. Never describe the obvious ("He walks into the store"). Everyday words,
     contractions, exclamation marks. No emojis, hashtags or stage directions. Say "GTA 6", never "GTA VI".
     Never invent facts - if unsure what something means, react to it instead of explaining it.
   - youtube_title: max 70 characters, curiosity-driven, no hashtags.
   - caption: 1-2 short sentences for Instagram/TikTok ending with a question, no hashtags.
   - x_text: max 200 characters punchy post for X to go with the clip, no hashtags, no links.
   - hashtags: 4 relevant hashtags (strings starting with #), always include #GTA6.

Return ONLY JSON with exactly these keys:
{{"is_gta6_footage": bool, "has_facecam": bool, "has_narrator": bool, "hype_score": int, "scene_summary": str,
 "clip_start": number, "clip_end": number, "content_type": "explainer" | "action" | "scenic",
 "banner": str, "commentary": [{{"t": number, "line": str}}], "youtube_title": str, "caption": str,
 "x_text": str, "hashtags": [str]}}"""


STREAM_NOTE = """THIS IS A STREAM CLIP: a streamer playing GTA 6 live, usually with a facecam and loud reactions.
- The facecam is expected and fine: set has_facecam=false unless the webcam covers most of the game.
- is_gta6_footage is about the GAME being shown (must be real GTA 6).
- hype_score: rate the gameplay AND how funny/wild the streamer's reaction is.
- Pick the part with the best gameplay moment AND the streamer's biggest reaction (include the build-up).
- Set content_type to "stream". The streamer is already talking, so the commentary must be ONLY 2 lines:
  a hook at clip_start (max 7 words, e.g. "This streamer was NOT ready!") and a short question near the end.
  Never talk over the middle."""


def write_clip_copy(clip: Path, title: str, credit: str, provider: str, kind: str = "gameplay",
                    hints: list | None = None) -> dict:
    """The AI watches the whole downloaded section, judges it, picks the best part (clip_start/clip_end)
    and writes for it. Commentary times are in SECTION time; finalize_commentary() converts them."""
    cfg = load_config()
    gcfg = cfg.get("gemini", {})
    vcfg = cfg.get("video", {})
    from common import probe
    clip_dur = probe(clip)["duration"] or 60
    min_len = int(min(vcfg.get("min_clip_sec", 25), clip_dur))
    max_len = int(min(vcfg.get("duration_sec", 50), clip_dur))
    prompt_kwargs = dict(
        brand=cfg.get("brand", {}).get("display_name", "GTA VI Top Moments"),
        context=_game_context(), title=title, credit=credit, provider=provider,
        duration=int(clip_dur), min_len=min_len, max_len=max_len,
        hints=", ".join(f"{a}-{b}s" for a, b in (hints or [])) or "unknown",
        kind_note=STREAM_NOTE if kind == "stream" else "",
    )

    data = None
    ai_source = ""
    # 1) Gemini WATCHES the full clip (best: sees motion + hears audio)
    uploaded = None
    if gcfg.get("watch_video", True) and env("GEMINI_API_KEY") and time.time() >= _unavailable_until:
        try:
            uploaded = _upload_video(clip)
        except Exception as exc:  # noqa: BLE001
            log.warning("[gemini] video upload failed: %s", exc)
    if uploaded is not None:
        prompt = CLIP_PROMPT.format(video_note="The attached video is the raw footage. Watch ALL of it carefully "
                                               "before choosing the best part.", **prompt_kwargs)
        data = _generate_json([uploaded, prompt])
        ai_source = "gemini-video"
        try:
            _client().files.delete(name=uploaded.name)
        except Exception:  # noqa: BLE001
            pass

    # 2) Gemini busy -> free vision model looks at frames sampled from the clip
    if data is None and ai_fallback.available():
        frames = ai_fallback.extract_frames(clip, int(gcfg.get("fallback_frames", 8)))
        if frames:
            full = probe(clip)["duration"] or clip_dur
            times = ", ".join(f"{full * (i + 0.5) / len(frames):.0f}s" for i in range(len(frames)))
            prompt = CLIP_PROMPT.format(
                video_note=f"The attached {len(frames)} images are frames from the raw footage, taken in order "
                           f"at these seconds: {times}. Study every frame carefully; use those times to choose "
                           "clip_start/clip_end and the commentary timestamps.",
                **prompt_kwargs)
            data = ai_fallback.chat_json(prompt, frames=frames)
            ai_source = (data or {}).get("_ai_model", "")

    if data is not None:
        data["verified_by_video"] = True
    else:
        # 3) nobody could look at the clip -> text-only copy from the title
        prompt = CLIP_PROMPT.format(video_note="(You cannot see the video this time - base everything on the title, "
                                               "keep descriptions general, and set hype_score to 6.)", **prompt_kwargs)
        data = _generate_json(prompt)
        if data is None and ai_fallback.available():
            data = ai_fallback.chat_json(prompt)
        ai_source = "text-only"
        if data is not None:
            data["verified_by_video"] = False

    if data and not _raw_lines(data):
        log.warning("[ai] %s gave no usable voice lines - keeping its verdict, lines will be rescued "
                    "(keys=%s, commentary=%.200r)", ai_source, sorted(data.keys()), data.get("commentary"))

    # 4) everything failed -> built-in randomised lines
    if not data:
        log.warning("[ai] using built-in fallback copy")
        data = fallback_clip_copy(clip_dur)
        data["verified_by_video"] = False
        ai_source = "built-in"
    out = _clean_copy(data, clip_dur)
    out["ai_source"] = ai_source
    if ai_source == "built-in":
        out["commentary_raw"] = []
    log.info("[ai] copy written by: %s", ai_source)
    return out


def _raw_lines(d: dict) -> list[dict]:
    """AI lines with their (section-time) timestamps, before any trimming."""
    return _commentary({**d, "content_type": "explainer"}, 10**6, keep_times=True)


def finalize_commentary(copy: dict, start: float, end: float) -> list[dict]:
    """Convert the AI's section-time lines to times inside the chosen clip [start, end].
    If nothing usable is left, a second AI writes lines from the scene description."""
    length = end - start
    shifted = []
    for ln in copy.get("commentary_raw", []):
        if ln.get("t") is None:  # no timestamp -> spread evenly later
            shifted.append({"t": None, "line": ln["text"]})
            continue
        rel = ln["t"] - start
        if -2.0 <= rel <= length - 1.0:
            shifted.append({"t": max(0.0, rel), "line": ln["text"]})
    lines = _commentary({"commentary": shifted, "content_type": copy.get("content_type", "action")}, length)
    if len(lines) >= 2 or (lines and copy.get("content_type") in ("scenic", "stream")):
        return lines
    return rescue_commentary(copy, length)


RESCUE_PROMPT = """Write the voiceover for a {length:.0f}-second GTA 6 short.
What happens: {summary}
Banner on screen: {banner}. Type: {ctype}.
You are an excited, fun GTA fan talking to friends. {amount}
Line 1 at t=0 is a hook (max 8 words) that creates curiosity about what's about to happen.
The last line, about 4 seconds before the end, is a short question or call to action.
Every line must be about something specific in this clip - no generic filler like "Rockstar cooked".
Don't repeat the game's name in every line (at most once, as "GTA 6"). No emojis or hashtags.
Return ONLY JSON: {{"commentary": [{{"t": seconds, "line": "words"}}]}}"""


def rescue_commentary(copy: dict, length: float) -> list[dict]:
    ctype = copy.get("content_type", "action")
    amount = {"explainer": "Write 4-6 lines that explain what's interesting.",
              "scenic": "Write ONLY 2 lines.", "stream": "Write ONLY 2 lines."}.get(
        ctype, "Write 3-4 short hype reactions (max 8 words each).")
    prompt = RESCUE_PROMPT.format(length=length, summary=copy.get("scene_summary") or "an intense GTA 6 moment",
                                  banner=copy.get("banner", ""), ctype=ctype, amount=amount)
    data = _generate_json(prompt)
    if data is None and ai_fallback.available():
        data = ai_fallback.chat_json(prompt)
    lines = _commentary({**(data or {}), "content_type": ctype}, length) if data else []
    if lines:
        log.info("[ai] voice lines rescued from the scene description")
        return lines
    # last resort: hook built from the AI's own (specific) banner + a question
    hook = (copy.get("banner") or "Watch this").capitalize().rstrip(".!?") + "!"
    log.warning("[ai] using minimal hook + question lines")
    return [{"t": 0.0, "text": hook}, {"t": max(3.0, length - 4), "text": random.choice(_OUTROS)}]


LIMITS = {  # content_type -> (max lines, max words per line)
    "explainer": (7, 22), "action": (4, 10), "scenic": (2, 10), "stream": (2, 10),
}


def _commentary(d: dict, clip_dur: float, keep_times: bool = False) -> list[dict]:
    """Normalise the AI's commentary into [{"t": float, "text": str}], in order, sized by content type."""
    raw = d.get("commentary") or d.get("lines") or d.get("voiceover") or d.get("voice_lines") or []
    if not raw and d.get("script"):  # old-style single script -> split into sentences
        raw = [x.strip() for x in re.findall(r"[^.!?]+[.!?]*", str(d["script"])) if x.strip()]
    if isinstance(raw, str):
        raw = [x.strip() for x in re.findall(r"[^.!?]+[.!?]*", raw) if x.strip()]
    ctype = str(d.get("content_type", "action")).lower()
    max_lines, max_words = LIMITS.get(ctype, LIMITS["action"])
    if len(raw) > max_lines:  # keep the hook, the outro and the first middle lines
        raw = raw[:max_lines - 1] + raw[-1:]
    lines = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            text, t = item.get("line") or item.get("text") or "", item.get("t", item.get("time", 0))
        else:
            text, t = str(item), None
        text = re.sub(r"[*_#\[\]()]|\s+", " ", str(text)).strip()
        if not text:
            continue
        words = text.split()
        if len(words) > max_words:  # keep whole sentences when possible
            kept, count = [], 0
            for sent in re.findall(r"[^.!?]+[.!?]*", text):
                n = len(sent.split())
                if count + n > max_words:
                    break
                kept.append(sent.strip())
                count += n
            text = " ".join(kept) if kept else " ".join(words[:max_words]).rstrip(",;:") + "!"
        try:
            ts = str(t).strip().rstrip("s")
            if ":" in ts:  # "0:05" / "00:12"
                mm, ss = ts.split(":")[-2:]
                t = int(mm) * 60 + float(ss)
            else:
                t = float(ts)
        except (TypeError, ValueError):
            t = None
        lines.append({"t": t, "text": text})
    if not lines:
        return []
    if keep_times:
        return lines
    n = len(lines)
    for i, ln in enumerate(lines):  # fill missing / silly timestamps evenly
        if ln["t"] is None or not 0 <= ln["t"] < clip_dur:
            ln["t"] = 0.0 if i == 0 else round((clip_dur - 5) * i / max(1, n - 1), 1)
    lines.sort(key=lambda x: x["t"])
    lines[0]["t"] = min(lines[0]["t"], 1.0)
    return lines


def _clean_copy(d: dict, clip_dur: float = 48) -> dict:
    def s(key, default=""):
        return re.sub(r"\s+", " ", str(d.get(key) or default)).strip()
    tags = d.get("hashtags") or ["#GTA6", "#GTAVI", "#GrandTheftAutoVI", "#Gaming"]
    tags = ["#" + re.sub(r"[^A-Za-z0-9_]", "", t.lstrip("#")) for t in tags if str(t).strip("# ")]
    if "#GTA6" not in tags:
        tags.insert(0, "#GTA6")
    raw = _raw_lines(d)
    try:
        clip_start = float(d.get("clip_start")) if d.get("clip_start") is not None else None
        clip_end = float(d.get("clip_end")) if d.get("clip_end") is not None else None
    except (TypeError, ValueError):
        clip_start = clip_end = None
    return {
        "is_gta6_footage": bool(d.get("is_gta6_footage", True)),
        "has_facecam": bool(d.get("has_facecam", False)),
        "has_narrator": bool(d.get("has_narrator", False)),
        "content_type": str(d.get("content_type", "action")).lower(),
        "hype_score": int(d.get("hype_score") or 6),
        "scene_summary": s("scene_summary"),
        "verified_by_video": bool(d.get("verified_by_video", False)),
        "banner": re.sub(r"[^A-Z0-9 !?'.,-]", "", s("banner", "INSANE GTA 6 MOMENT").upper())[:48].strip() or "INSANE GTA 6 MOMENT",
        "commentary_raw": raw,
        "clip_start": clip_start, "clip_end": clip_end,
        "script": " ".join(ln["text"] for ln in raw),
        "youtube_title": s("youtube_title", "This GTA 6 moment is unreal")[:80],
        "caption": s("caption", "This GTA 6 moment goes crazy. Would you try this?"),
        "x_text": s("x_text", "This GTA 6 moment goes crazy")[:240],
        "hashtags": list(dict.fromkeys(tags))[:5],
    }


# --------------------------------------------------------------------------- fallback copy

_HOOKS = [
    "Bro, GTA 6 is actually insane!", "Okay, this is NOT normal!", "Wait for it...",
    "Rockstar did NOT hold back!", "Nobody was ready for this!", "This is GTA 6?! No way!",
    "Watch this. Just watch.", "Vice City is wild, man!",
]
_REACTIONS = [
    "NO WAY!", "Look at that!", "Oh my god, the physics!", "He's not stopping!", "Rockstar cooked!",
    "That's crazy!", "Bro, the detail on this!", "This looks like a movie!", "Did you see that?!",
    "Okay, that was smooth!", "This game is unreal!", "Chaos. Pure chaos!",
]
_OUTROS = [
    "Would you do this? Tell me!", "Rate this from one to ten!", "Jason or Lucia? Comment!",
    "Follow for more GTA 6!", "Best game ever? Be honest!", "Would you survive this?",
]
_BANNERS = [
    "THIS GTA 6 MOMENT IS UNREAL", "ROCKSTAR WENT CRAZY HERE", "WAIT FOR THE ENDING",
    "GTA 6 IS NEXT LEVEL", "LEONIDA IS PURE CHAOS", "NOBODY EXPECTED THIS",
    "VICE CITY HITS DIFFERENT", "THE DETAIL IS INSANE", "WATCH THIS TWICE",
]


def fallback_clip_copy(clip_dur: float = 48) -> dict:
    reactions = random.sample(_REACTIONS, 2)
    commentary = [
        {"t": 0.0, "text": random.choice(_HOOKS)},
        {"t": round(clip_dur * 0.35, 1), "text": reactions[0]},
        {"t": round(clip_dur * 0.65, 1), "text": reactions[1]},
        {"t": round(max(10, clip_dur - 5), 1), "text": random.choice(_OUTROS)},
    ]
    banner = random.choice(_BANNERS)
    return {
        "is_gta6_footage": True, "has_facecam": False, "hype_score": 6, "scene_summary": "",
        "banner": banner, "commentary": commentary,
        "youtube_title": banner.title(),
        "caption": f"{banner.capitalize()}. {random.choice(_OUTROS)}",
        "x_text": f"{banner.capitalize()} 🔥",
        "hashtags": random.choice(load_config().get("hashtags_pool", [["#GTA6", "#GTAVI"]])),
    }


# --------------------------------------------------------------------------- X text posts

TEXT_KINDS = {
    "x_text_morning": "a good-morning engagement post: a fun question or hot take about GTA 6 gameplay, "
                      "missions, Leonida, cars or characters that people will want to reply to",
    "x_question_nostalgia": "a nostalgia post comparing a GTA memory (GTA III, Vice City, San Andreas, GTA IV, GTA V) "
                            "with GTA 6, ending with a question, OR a 'this or that' poll-style question",
    "x_daily_recap": "an end-of-day recap post for our followers summarising today's GTA 6 hype and the clips we posted, "
                     "ending with a teaser for tomorrow",
}

_TEXT_FALLBACK = {
    "x_text_morning": [
        "Good morning Leonida ☀️ What's the first thing you're doing in GTA 6 today?",
        "Be honest: are you a 'follow the story' player or a '6 hours of chaos before mission 1' player in GTA 6?",
        "First car you're stealing in Vice City? Wrong answers only 🚗",
        "Jason or Lucia - who are you trusting with the getaway car? 👀",
        "GTA 6 hot take: the side activities are going to be better than most full games. Agree?",
        "What's your GTA 6 playstyle: stealth, loud, or 'I'll just drive around and look at the sunset'?",
        "Morning check: how many hours of GTA 6 are we putting in today? ⏱️",
        "Rate your GTA 6 wanted-level survival skills from 1 to 5 ⭐",
    ],
    "x_question_nostalgia": [
        "From Tommy Vercetti to Jason & Lucia. Vice City has come a long way 🌴 Which era was your favourite?",
        "GTA San Andreas or GTA 6 Leonida - which map are you exploring forever?",
        "Your first ever GTA game was ___. Go 👇",
        "GTA IV physics or GTA 6 physics? Be honest 🚗💥",
        "Remember typing cheat codes in Vice City? What's the first thing you'd want in GTA 6?",
        "Best GTA soundtrack of all time? Vice City set the bar 📻",
        "Michael, Franklin, Trevor... or Jason & Lucia? Who's the best GTA crew?",
        "13 years after GTA 5. Was the wait worth it? 👀",
    ],
    "x_daily_recap": [
        "That's a wrap on today's GTA 6 moments 🎬 Which clip was your favourite? More tomorrow.",
        "Another day in Leonida, another set of insane GTA 6 moments. Tomorrow we go even bigger 🔥",
        "Today's GTA 6 highlights are done. Drop the moment you want us to cover next 👇",
        "Daily recap: chaos, chases and Vice City views. Same time tomorrow 🌴",
        "GTA 6 never sleeps, but we do. Back tomorrow with more top moments 🌙",
    ],
}


def write_text_post(kind: str, recent_texts: list[str], todays_clips: list[str]) -> str:
    kind = kind if kind in TEXT_KINDS else "x_text_morning"
    prompt = f"""You run the X account of "{load_config().get('brand', {}).get('display_name', 'GTA VI Top Moments')}".
CONTEXT: {_game_context()}
Write {TEXT_KINDS[kind]}.
Clips we posted today: {todays_clips or 'none yet'}.
Rules: max 230 characters, conversational, at most 1-2 emojis, at most 1 hashtag, NO links,
never invent news, leaks, prices or release details. Must be clearly different from these recent posts:
{recent_texts[-12:]}
Return ONLY JSON: {{"text": "..."}}"""
    data = _generate_json(prompt, temperature=1.0)
    if data is None and ai_fallback.available():
        data = ai_fallback.chat_json(prompt, temperature=1.0)
    text = re.sub(r"\s+", " ", str((data or {}).get("text", ""))).strip()
    text = re.sub(r"https?://\S+", "", text).strip()
    if 15 <= len(text) <= 280 and text not in recent_texts:
        return text
    pool = [t for t in _TEXT_FALLBACK[kind] if t not in recent_texts] or _TEXT_FALLBACK[kind]
    return random.choice(pool)

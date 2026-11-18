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
        timeout_ms = int(load_config().get("gemini", {}).get("timeout_sec", 75)) * 1000
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

CLIP_PROMPT = """You are the head writer for "{brand}", a viral faceless channel posting the most hype
Grand Theft Auto VI moments as vertical Shorts/Reels/TikToks.

CONTEXT: {context}
Source footage title: "{title}" (credit: {credit}). Source type: {provider}.
{video_note}

TASKS
1. QUALITY GATE - judge the footage honestly:
   - is_gta6_footage: true ONLY if this is real Grand Theft Auto VI footage (Leonida / Vice City,
     Jason & Lucia, GTA VI HUD, official trailers). false for GTA V / GTA Online, mods, other games,
     fan-made "concept" trailers, AI fakes, real-life videos, news anchors, slideshows or menus.
   - has_facecam: true if a streamer webcam / face overlay or a big channel watermark covers the video.
   - hype_score: 1-10, how exciting the visuals are for a GTA fan (chases, shootouts, stunts,
     cinematic story moments, stunning views = high; menus, walking, loading screens = low).
   - scene_summary: one sentence describing exactly what happens on screen.
2. WRITE THE VOICEOVER - fun, simple, hyped. Think: your funniest friend watching this clip next to you
   and losing his mind. NOT a narrator, NOT a documentary. The game audio carries the video; the voice only
   jumps in a few times with short reactions.
   - banner: max 6 words, ALL CAPS, curiosity hook for the top of the video, no emojis.
   - commentary: 3 to 5 lines. Each line = {{"t": second in the clip when it is said, "line": "words"}}.
     The clip is {duration} seconds long.
       * line 1 at t=0: scroll-stopping hook, max 7 words. Like: "Bro, GTA 6 is actually insane!"
       * 1-3 reactions placed EXACTLY when the craziest things happen (crash, jump, explosion, twist,
         stunning view), max 8 words each. Like: "NO WAY he made that jump!", "Look at those physics!",
         "Rockstar really cooked with this one."
       * last line around t={end_t}: quick fun question or call to action, max 8 words.
         Like: "Would you survive this? Tell me!"
       * at least 5 seconds between lines, under 40 words in total.
       * NEVER narrate what is on screen ("He walks into the store", "We see a red car", "The camera pans").
         React to it instead. Everyday words, contractions, exclamation marks. No emojis, hashtags or
         stage directions. Say "GTA 6", never "GTA VI". Don't invent facts.
   - youtube_title: max 70 characters, curiosity-driven, no hashtags.
   - caption: 1-2 short sentences for Instagram/TikTok ending with a question, no hashtags.
   - x_text: max 200 characters punchy post for X to go with the clip, no hashtags, no links.
   - hashtags: 4 relevant hashtags (strings starting with #), always include #GTA6.

Return ONLY JSON with exactly these keys:
{{"is_gta6_footage": bool, "has_facecam": bool, "hype_score": int, "scene_summary": str,
 "banner": str, "commentary": [{{"t": number, "line": str}}], "youtube_title": str, "caption": str,
 "x_text": str, "hashtags": [str]}}"""


def write_clip_copy(clip: Path, title: str, credit: str, provider: str) -> dict:
    """Return copy dict. Always returns something usable (fallback if AI fails)."""
    cfg = load_config()
    gcfg = cfg.get("gemini", {})
    from common import probe
    clip_dur = min(probe(clip)["duration"] or 50, float(cfg.get("video", {}).get("duration_sec", 48)))
    prompt_kwargs = dict(
        brand=cfg.get("brand", {}).get("display_name", "GTA VI Top Moments"),
        context=_game_context(), title=title, credit=credit, provider=provider,
        duration=int(clip_dur), end_t=max(10, int(clip_dur) - 5),
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
        prompt = CLIP_PROMPT.format(video_note="The attached video IS the clip that will be posted. Watch it carefully.",
                                    **prompt_kwargs)
        data = _generate_json([uploaded, prompt])
        ai_source = "gemini-video"
        try:
            _client().files.delete(name=uploaded.name)
        except Exception:  # noqa: BLE001
            pass

    # 2) Gemini busy -> free vision model looks at frames sampled from the clip
    if data is None and ai_fallback.available():
        frames = ai_fallback.extract_frames(clip, 6)
        if frames:
            full = probe(clip)["duration"] or clip_dur
            times = ", ".join(f"{full * (i + 0.5) / len(frames):.0f}s" for i in range(len(frames)))
            prompt = CLIP_PROMPT.format(
                video_note=f"The attached {len(frames)} images are frames from the clip that will be posted, "
                           f"taken in order at these seconds: {times}. Study every frame carefully and use "
                           "those times for the commentary timestamps.",
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

    # 4) everything failed -> built-in randomised lines
    if not data or not _commentary(data, clip_dur):
        log.warning("[ai] using built-in fallback copy")
        data = fallback_clip_copy(clip_dur)
        data["verified_by_video"] = False
        ai_source = "built-in"
    out = _clean_copy(data, clip_dur)
    out["ai_source"] = ai_source
    log.info("[ai] copy written by: %s", ai_source)
    return out


def _commentary(d: dict, clip_dur: float) -> list[dict]:
    """Normalise the AI's commentary into [{"t": float, "text": str}], short and in order."""
    raw = d.get("commentary") or []
    if isinstance(raw, str):
        raw = [raw]
    lines = []
    for i, item in enumerate(raw[:6]):
        if isinstance(item, dict):
            text, t = item.get("line") or item.get("text") or "", item.get("t", item.get("time", 0))
        else:
            text, t = str(item), None
        text = re.sub(r"[*_#\[\]()]|\s+", " ", str(text)).strip()
        if not text:
            continue
        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14]).rstrip(",;:") + "!"
        try:
            t = float(str(t).rstrip("s"))
        except (TypeError, ValueError):
            t = None
        lines.append({"t": t, "text": text})
    if not lines:
        return []
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
    commentary = _commentary(d, clip_dur) or fallback_clip_copy(clip_dur)["commentary"]
    return {
        "is_gta6_footage": bool(d.get("is_gta6_footage", True)),
        "has_facecam": bool(d.get("has_facecam", False)),
        "hype_score": int(d.get("hype_score") or 6),
        "scene_summary": s("scene_summary"),
        "verified_by_video": bool(d.get("verified_by_video", False)),
        "banner": re.sub(r"[^A-Z0-9 !?'.,-]", "", s("banner", "INSANE GTA 6 MOMENT").upper())[:48].strip() or "INSANE GTA 6 MOMENT",
        "commentary": commentary,
        "script": " ".join(ln["text"] for ln in commentary),
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

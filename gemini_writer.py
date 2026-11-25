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

import atexit
import logging
import random
import re
import threading
import time
from datetime import date
from pathlib import Path

import ai_fallback
import formats
from common import WORK_DIR, env, load_config, now_local, parse_json_loose, probe, run

log = logging.getLogger("autopilot")

_client_obj = None
_unavailable_until = 0.0  # skip Gemini for the rest of the run after repeated overloads
_busy_until: dict[str, float] = {}  # model -> time: a model that just said 503/504 is skipped for a few minutes


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
            # newest 3 Flash + newest 2 Flash-Lite: every model has its own free daily quota, and
            # Flash-Lite can watch video too
            full = sorted({n for n in names if "lite" not in n}, key=_version_key, reverse=True)[:3]
            lite = sorted({n for n in names if "lite" in n}, key=_version_key, reverse=True)[:2]
            _discovered = full + lite
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


_pro_cache: list[str] | None = None


def _pro_models() -> list[str]:
    """Gemini Pro models this key can call (free tier: small daily limit - used only where quality matters most)."""
    global _pro_cache
    if _pro_cache is None:
        _pro_cache = []
        if not load_config().get("gemini", {}).get("use_pro", True):
            return _pro_cache
        try:
            names = []
            for m in _client().models.list():
                name = (m.name or "").removeprefix("models/")
                actions = [a.lower() for a in (getattr(m, "supported_actions", None) or [])]
                if actions and not any("generatecontent" in a for a in actions):
                    continue
                if "pro" in name and name.startswith("gemini") and not any(
                        x in name for x in ("image", "tts", "audio", "live", "embedding", "vision", "robotics",
                                            "computer", "research")):
                    names.append(name)
            _pro_cache = sorted(set(names), key=_version_key, reverse=True)[:2]
            log.info("[gemini] Pro models for the important jobs: %s", _pro_cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("[gemini] could not list Pro models: %.150s", exc)
    return [m for m in _pro_cache if m not in _dead_models]


def _generate_json(contents, temperature: float = 0.9, prefer_pro: bool = False) -> dict | None:
    """Try each Gemini model ONCE. Busy / out of quota / error -> next model immediately.
    prefer_pro: try the (smarter, rate-limited) Pro models first, then Flash.
    If every model fails, Gemini is skipped for the rest of this run (backup AIs take over)."""
    global _unavailable_until
    if not env("GEMINI_API_KEY"):
        log.warning("[gemini] GEMINI_API_KEY not set - using backup AI")
        return None
    if time.time() < _unavailable_until:
        return None
    from google.genai import types
    for attempt in (1, 2):
        overloaded = False
        models = list(dict.fromkeys((_pro_models() if prefer_pro else []) + _models()))
        if attempt == 1:
            # a model that was overloaded a few minutes ago is skipped (each 503/504 costs up to ~50s)
            fresh = [m for m in models if _busy_until.get(m, 0) <= time.time()]
            if fresh and len(fresh) < len(models):
                overloaded = True  # if the fresh ones fail too, the 2nd round tries the busy ones again
                models = fresh
        for model in models:
            data, busy = _try_model(types, model, contents, temperature)
            if data:
                return data
            overloaded = overloaded or busy
        if attempt == 1 and overloaded:
            # 503 "high demand" / 504 timeouts are temporary - one more round after a short wait
            log.info("[gemini] models overloaded - waiting 40s and trying once more")
            time.sleep(40)
            continue
        break
    log.warning("[gemini] all models busy/out of quota - skipping Gemini for the rest of this run")
    _unavailable_until = time.time() + 3600
    return None


def _try_model(types, model: str, contents, temperature: float) -> tuple[dict | None, bool]:
    """One request. Returns (data, overloaded)."""
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
            return data, False
        log.warning("[gemini] %s returned unparseable output - next model", model)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        log.warning("[gemini] %s failed (%.120s) - next model", model, msg)
        if ("PerMinute" in msg or "per minute" in msg.lower()) and "PerDay" not in msg:
            # a per-MINUTE limit (two requests at once, or a quick retry) - not today's quota: pause this model
            # briefly instead of dropping it (and its remaining daily quota) for the rest of the run
            m = re.search(r"retry in ([\d.]+)\s*s|retryDelay'?\"?:\s*'?\"?(\d+)s", msg)
            wait = float(next((g for g in (m.groups() if m else ()) if g), 30))
            _busy_until[model] = time.time() + min(90.0, wait + 2)
            return None, True
        if any(x in msg for x in ("404", "NOT_FOUND", "not supported", "exceeded your current quota",
                                  "RESOURCE_EXHAUSTED")):
            _dead_models.add(model)  # gone or out of quota for today - don't try it again this run
            return None, False
        busy = any(x in msg for x in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "high demand"))
        if busy:
            _busy_until[model] = time.time() + float(load_config().get("gemini", {}).get("busy_cooldown_sec", 480))
        return None, busy
    return None, False


_uploads: dict[tuple, object] = {}  # (path, size, mtime) -> ACTIVE Gemini file of that exact file content
_uploads_lock = threading.Lock()
_cleanup_registered = False


def _file_id(clip: Path) -> tuple | None:
    try:
        st = Path(clip).stat()
    except OSError:
        return None
    return str(Path(clip).resolve()), st.st_size, st.st_mtime_ns


def _delete_upload(f) -> None:
    try:
        _client().files.delete(name=f.name)
    except Exception:  # noqa: BLE001 - the file expires on its own after 48 h
        pass


def release_uploads(clip: Path | None = None) -> None:
    """Delete this run's Gemini uploads of `clip` (every version of that file), or all of them."""
    path = str(Path(clip).resolve()) if clip else None
    with _uploads_lock:
        files = [_uploads.pop(k) for k in [k for k in _uploads if path is None or k[0] == path]]
    for f in files:
        _delete_upload(f)


def _upload_video(clip: Path):
    """The clip on the Gemini Files API, uploaded ONCE per file per run: the editor, every script attempt and a
    repair watch the same upload (a 5-min tour reel took ~50 s to encode + upload - three times for one posted
    Short). A changed file (a re-render) has a new size / modification time, so it is uploaded again. Uploads
    are deleted by release_uploads() - all of them when the run ends."""
    global _cleanup_registered
    fid = _file_id(clip)
    with _uploads_lock:
        if fid in _uploads:
            log.info("[gemini] reusing this run's upload of %s", Path(clip).name)
            return _uploads[fid]
    f = _upload_proxy(clip)
    if f is None or fid is None:
        return f
    with _uploads_lock:
        kept = _uploads.setdefault(fid, f)
        register = not _cleanup_registered
        _cleanup_registered = True
    if register:
        atexit.register(release_uploads)
    if kept is not f:  # another thread uploaded the same file meanwhile
        _delete_upload(f)
    return kept


def _upload_proxy(clip: Path):
    """Upload a light 480p proxy of the clip to the Gemini Files API."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    proxy = WORK_DIR / f"gemini_proxy_{threading.get_ident()}.mp4"  # own file per thread (parallel checks)
    src_dur = probe(clip)["duration"] or 0
    res = run(["ffmpeg", "-y", "-i", str(clip), "-vf", "scale=-2:480,fps=12",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
               "-c:a", "aac", "-b:a", "48k", "-ac", "1", str(proxy)], timeout=int(max(240, src_dur * 1.5 + 60)))
    if res.returncode != 0:
        proxy.unlink(missing_ok=True)
        return None
    got = probe(proxy)["duration"] or 0
    if src_dur and got < src_dur * 0.97 - 0.5:  # never let an AI judge a cut-off copy ("ends abruptly")
        log.warning("[gemini] the upload copy is %.1fs of %.1fs - not uploading a cut-off video", got, src_dur)
        proxy.unlink(missing_ok=True)
        return None
    client = _client()
    try:
        f = client.files.upload(file=str(proxy), config={"mime_type": "video/mp4"})
    finally:
        proxy.unlink(missing_ok=True)
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
{format_note}

TASKS
1. QUALITY GATE - judge honestly:
   - is_gta6_footage: true ONLY for real Grand Theft Auto VI footage (Leonida / Vice City, Jason & Lucia,
     GTA VI HUD). false for GTA V / GTA Online, mods, other games, fan "concept" trailers, AI fakes,
     real-life videos, news anchors, slideshows or menus.
   - has_facecam: true if a streamer webcam / face overlay or a big channel watermark covers the video.
   - has_narrator: true if a YouTuber / commentator talks over the footage in their own voice
     (NOT game characters, NOT music). Only judge this if you can hear the audio; else false.
   - unsafe: true for nudity or sexual content - IN-GAME TOO (strip clubs, lap dances, topless / naked characters,
     sex or shower scenes) - or slurs / hate speech, real-world violence. In-game violence and crime are fine.
   - is_spoiler: true if the footage (or the part you pick) shows STORY SPOILERS - story cutscenes,
     plot reveals, character deaths or betrayals, mission outcomes, endings, late-game locations/unlocks.
     Normal free-roam gameplay, chases, stunts, funny moments and the world itself are NOT spoilers.
   - fits_format: true if this footage can honestly make THIS VIDEO'S FORMAT (true if no format is given).
   - hype_score: 1-10 for the BEST part you pick in step 2, judged like a strict viral-shorts editor:
     9-10 = jaw-dropping (insane crash, wild chase, shootout, huge stunt, shocking story twist, hilarious chaos)
     7-8 = clearly exciting or intriguing; 5-6 = okay but would be scrolled past;
     1-4 = dialogue, walking, menus, loading, calm driving. Be strict - most footage is 4-6.
   - scene_summary: one sentence on exactly what happens in the part you pick.
2. EDIT - pick the single best continuous part, {min_len} to {max_len} seconds long:
   - clip_start / clip_end in seconds of THIS footage ({duration} s long).
   - NO STORY SPOILERS: never pick a story cutscene or plot moment; pick the part that shows the format's topic
     most clearly.
   - Start 1-2 seconds BEFORE the action kicks in, so the first seconds already hook. End right after the
     payoff (crash, escape, punchline, reveal). Skip dialogue-only, walking and loading parts.
   - Motion/loudness analysis says the most intense parts are around: {hints}. Use it as a hint only.
   - payoff_t: the second of THIS footage where the single biggest moment happens (the crash, the jump
     landing, the punchline). The video opens with a 1-second flash of it, and it gets a zoom + sound effect.
3. WRITE THE NARRATION SCRIPT for the part you picked. A voice actor reads it over the clip, so it must be a
   GREAT script like the top GTA tips & guides Shorts channels (millions of views): a clear, confident guide
   who knows the game, talking straight to the viewer - NOT an announcer, NOT a comedian, NOT a list of random
   reactions. It flows like one explanation:
     HOOK (first 2 s, the useful promise: "This is the fastest way to lose the cops in GTA 6.") ->
     CONTEXT (what you are seeing) -> HOW IT WORKS (what to do, step by step, at the moment it happens) ->
     PAYOFF (the result: what you get / what happens) -> a useful detail or a GTA 5 comparison you can SEE on
     screen -> ENDING question that makes people comment.
   content_type (one of):
     "explainer" = a secret, easter egg, hidden detail or new mechanic that needs context
     "action"    = chase, shootout, crash, stunt, heist, chaos
     "scenic"    = views, cruising, city life, mostly calm
   - commentary: list of {{"t": second of THIS footage, "line": "words"}}, every t between clip_start and
     clip_end, in order. 6 to 10 lines; each line is 1-2 spoken sentences (6 to 24 words). All lines together
     about 2.4 words per second of clip (a 50 s clip = about 120 words). The voice talks for most of the clip
     but leaves short gaps so the big moment's sound can hit.
   - Line 1 at t=clip_start is the HOOK (max 12 words). The last line, about 4 seconds before clip_end, is
     the question. Put each line's t where the thing it talks about happens on screen. Leave enough time for
     each line: the next t must be at least (words in this line / 2.7) seconds later.
   - VOICE: a clear, confident guide who knows the game - spoken English with contractions and short
     sentences, friendly, never slang like "bro" / "nah", never roasting or mocking the player. Each line tells
     the viewer something useful: what is happening, how it works, where it is, what you get, what to avoid.
   - banner: max 6 words, ALL CAPS, makes people NEED to see what happens. No emojis.
   - Every line must be about something SPECIFIC in this clip. BANNED filler: "Rockstar cooked", "this game
     is insane", "the detail is unreal", "look at that", "wait for it", "next level" - unless tied to a
     concrete thing on screen. Never just describe the obvious ("He drives the truck to the beach") - add
     the useful part: how, where or why ("Stolen cars get reported fast - change cars before the cops see
     you."). No emojis, hashtags or stage directions. Say "GTA 6", never "GTA VI", at most twice.
     Never invent facts - if unsure what something means, react to it instead of explaining it.
   PACKAGING - people find Shorts by SEARCHING: search words first, and promise only what this clip shows.
   {seo_note}
   - youtube_title: max 70 characters. START with "GTA 6" + the search phrase people type for this clip
     ("GTA 6 Police Chase ...", "GTA 6 Money Tip: ...", "GTA 6 Funny Moments: ..."), then the curiosity hook.
     Max 2 CAPITALISED power words, no hashtags, no emojis. Never words that sound like cheats or glitches
     (hack, glitch, exploit, cheat, mod, trick, unlimited / infinite money) - above all in money videos: say
     tip / way / method instead.
   - caption: TikTok/Instagram text, 1-2 short sentences: the first names the topic with the search words
     ("GTA 6" + topic), the last is a question that makes people comment. No hashtags.
   - description: 2 sentences for the YouTube description: the first starts with "GTA 6" + the main search
     phrase and says what the video shows; the second adds one specific detail from the clip. No hashtags/links.
   - x_text: one short punchy line for X (max 140 characters): a reaction or hot take about the clip, not the
     title again. No hashtags, no links.
   - hashtags: 4 (strings starting with #): #GTA6 + 3 specific ones people search (#GTA6Money, #GTA6Funny,
     #ViceCity...). Never company, console or car brands (no #RockstarGames, #PlayStation, #PS5, #Xbox,
     #Lamborghini, #ad) - no sponsored-looking content.

Return ONLY JSON with exactly these keys:
{{"is_gta6_footage": bool, "unsafe": bool, "has_facecam": bool, "has_narrator": bool, "is_spoiler": bool,
 "fits_format": bool, "shows_required": bool, "required_t": number | null, "required_evidence": str, "hype_score": int,
 "scene_summary": str, "clip_start": number, "clip_end": number, "payoff_t": number,
 "content_type": "explainer" | "action" | "scenic",
 "banner": str, "commentary": [{{"t": number, "line": str}}], "youtube_title": str, "caption": str,
 "description": str, "x_text": str, "hashtags": [str]}}"""


TOUR_PROMPT = """You are the editor AND narrator of "{brand}", a viral faceless channel posting Grand Theft Auto 6
gameplay shorts. You make "feature tour" shorts like the best GTA creators: a fast montage of different shots
with a warm, casual, human voice that keeps telling the viewer something new.

CONTEXT: {context}
Source footage title: "{title}" (credit: {credit}).
{video_note}
{format_note}

TASKS
1. QUALITY GATE - judge honestly:
   - is_gta6_footage: true ONLY for real Grand Theft Auto VI gameplay (Leonida / Vice City, Jason & Lucia,
     GTA VI HUD). false for GTA V / Online, mods, other games, fan concepts, AI fakes, real-life video, menus.
   - has_facecam: true if a webcam / face overlay or big channel watermark covers the video.
   - has_narrator: true if a YouTuber talks over the footage in their own voice (not game characters).
     Only judge this if you can hear the audio; else false.
   - unsafe: true for nudity or sexual content - IN-GAME TOO (strip clubs, lap dances, topless / naked characters,
     sex or shower scenes) - or slurs / hate speech, real-world violence. In-game violence and crime are fine.
   - is_spoiler: true if the footage (or the part you pick) shows STORY SPOILERS - story cutscenes,
     plot reveals, character deaths or betrayals, mission outcomes, endings, late-game locations/unlocks.
     Normal free-roam gameplay, chases, stunts, funny moments and the world itself are NOT spoilers.
   - fits_format: true if this footage can honestly make THIS VIDEO'S FORMAT (true if no format is given).
   - hype_score 1-10: how interesting and VARIED the footage is for a GTA fan (new mechanics, wildlife,
     chaos, combat, NPC reactions, funny moments = high; one long cutscene, menus, walking = low).
   - scene_summary: one sentence on what the footage shows.
2. CHOOSE THE ANGLE (like the top GTA shorts with millions of views). If THIS VIDEO'S FORMAT is given above,
   follow it; otherwise pick the angle below that fits this footage best.
   NO STORY SPOILERS: never pick story cutscenes or plot moments, never mention plot, characters' fates or
   mission outcomes. Prefer free-roam gameplay that shows the format's topic clearly.
     "top_moments" - "Top 5 craziest / funniest moments in GTA 6. Number five..." (3 to 5 numbered items)
     "listicle" - "Here's 5 crazy things you can do in GTA 6 that you could never do in GTA 5. Number one..."
                  (3 to 5 numbered items, each shown with 1-2 shots)
     "how_it_works" - one mechanic explained step by step in "you" form, like a guide:
                  "If you rob a store in GTA 6 the way you did in GTA 5, the police can be waiting outside..."
     "first_look" - a personal tour of the coolest things in the footage: "Let me show you what it's really like."
3. EDIT + NARRATE - pick 6 to 12 shots, in the order they should appear:
   - "start" = second in THIS footage where the shot begins (just before the interesting thing happens).
   - "line" = what the narrator says while that shot plays. The shot lasts exactly as long as its line,
     so each line is 1-3 short spoken sentences (8 to 30 words). All lines together: 160-220 words
     (about 60-80 seconds). The narrator talks the whole time - but EVERY sentence must tell the viewer
     something useful or new. Never filler, never jokes with no information.
   - Every shot shows something DIFFERENT (a mechanic, vehicle, animal, fight, shop, NPC reaction, funny
     detail). Never two shots of the same thing. The first shot is the most eye-catching one.
   - Every shot shows something HAPPENING (a crash, chase, fight, stunt, funny NPC reaction, a feature being
     used). NEVER a shot of someone just walking, standing, roaming around or driving calmly - the narrator
     would have nothing to talk about. If the footage has fewer than 5 such moments, say so with a low hype_score.
   - VOICE: a clear, confident guide who knows the game - friendly spoken English, one useful fact per line
     (how it works, where it is, what you get, what to avoid). Never slang like "bro", never roasting or
     joking at the player's expense. Compare with GTA 5 when the difference is obvious on screen. Read
     on-screen prompts, menus and HUD text to explain mechanics correctly. NEVER announcer lines like "Watch
     this X!" and never empty hype like "Rockstar cooked" without saying WHY.
   - Line 1 is the hook that states the useful promise this footage pays off (e.g. "This is the fastest way to
     lose a five-star wanted level in GTA 6.").
   - Around the middle, a retention line ("But wait till you see what happens when you carry a gun.").
   - The last line is a short question that makes people comment.
   - Only claim what you can actually see or hear. If you're not sure how something works, say what you see
     ("looks like...") instead of inventing facts. Say "GTA 6", never "GTA VI". No emojis, hashtags,
     stage directions or "[Number one]" brackets.
   Examples of the voice (top channels):
     "Here's five crazy things you can do in GTA 6 that you could never do in GTA 5." /
     "Number two: the wanted level goes all the way up to six stars, and the cops actually remember you." /
     "So put a mask on first - a mask can keep your description away from the police, even on camera." /
     "One of the craziest things: cars now need gas. See this? My tank isn't full." /
     "This is Jason's house. It's pretty small, but it's better than nothing."
4. angle: "top_moments" | "listicle" | "funny_moments" | "how_it_works" | "first_look".
   banner: max 6 words, ALL CAPS, states the angle (e.g. "5 THINGS GTA 5 COULDN'T DO", "ROBBING STORES IS DIFFERENT NOW").
   PACKAGING - people find Shorts by SEARCHING: search words first, and promise only what the video shows.
   {seo_note}
   youtube_title: max 70 characters, "GTA 6" + the search phrase in the first 40 characters, like the top
   channels: "GTA 6: 5 CRAZY Things You CAN'T Do In GTA 5", "GTA 6 Robbing Stores Works Differently To GTA 5",
   "GTA 6 Money Tip: ...". The number must match the items you narrate. Max 2 CAPITALISED words, no hashtags.
   Never words that sound like cheats or glitches (hack, glitch, exploit, cheat, mod, trick, unlimited money),
   above all in money videos - say tips / ways / methods.
   caption: 1-2 sentences for TikTok/Instagram: the first names the topic with the search words ("GTA 6" +
   topic), the last is a question that makes people comment. No hashtags.
   description: 2 sentences for YouTube: the first starts with "GTA 6" + the main search phrase and says what
   the video shows, the second adds one specific detail. No hashtags, no links.
   x_text: one short punchy line for X (max 140 characters), not the title again, no hashtags, no links.
   hashtags: 4, #GTA6 + 3 specific ones people search; never company, console or car brands (no #RockstarGames,
   #PlayStation, #PS5, #Xbox, #Lamborghini, #ad).

Return ONLY JSON with exactly these keys:
{{"is_gta6_footage": bool, "unsafe": bool, "has_facecam": bool, "has_narrator": bool, "is_spoiler": bool,
 "fits_format": bool, "shows_required": bool, "required_t": number | null, "required_evidence": str, "hype_score": int,
 "scene_summary": str, "angle": str, "shots": [{{"start": number, "line": str}}], "banner": str, "youtube_title": str, "caption": str,
 "description": str, "x_text": str, "hashtags": [str]}}"""


STREAM_NOTE = """THIS IS A STREAM CLIP: a streamer playing GTA 6 live, usually with a facecam and loud reactions.
- The facecam is expected and fine: set has_facecam=false unless the webcam covers most of the game.
- is_gta6_footage is about the GAME being shown (must be real GTA 6).
- hype_score: rate the gameplay AND how funny/wild the streamer's reaction is.
- NARRATION for streams: only 3 to 5 lines (about 40-70 words) - set the scene before the moment, then go
  quiet so the streamer's own reaction is heard, then one line + the question at the end.
- Pick the part with the best gameplay moment AND the streamer's biggest reaction (include the build-up).
- Set content_type to "stream". The streamer is already talking, so the commentary must be ONLY 2 lines:
  a hook at clip_start (max 7 words, e.g. "This streamer was NOT ready!") and a short question near the end.
  Never talk over the middle."""


def write_clip_copy(clip: Path, title: str, credit: str, provider: str, kind: str = "gameplay",
                    hints: list | None = None, tour: bool = False, fmt: dict | None = None) -> dict:
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
        format_note=formats.prompt_note(fmt, tour),
        seo_note=packaging_note(fmt),
    )
    template = TOUR_PROMPT if tour else CLIP_PROMPT

    def build(video_note: str) -> str:
        return template.format(video_note=video_note, **prompt_kwargs)

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
        prompt = build("The attached video is the raw footage" + (" (several slices of one gameplay video joined "
                       "together)" if tour else "") + ". Watch ALL of it carefully before choosing.")
        data = _generate_json([uploaded, prompt])
        ai_source = "gemini-video"  # the upload stays: a tour's script writer watches this same reel

    # 2) Gemini busy -> free vision model looks at frames sampled from the clip
    if data is None and ai_fallback.available():
        frames = ai_fallback.extract_frames(clip, int(gcfg.get("tour_frames" if tour else "fallback_frames", 8)))
        if frames:
            full = probe(clip)["duration"] or clip_dur
            times = ", ".join(f"{full * (i + 0.5) / len(frames):.0f}s" for i in range(len(frames)))
            prompt = build(f"The attached {len(frames)} images are frames from the raw footage, taken in order "
                           f"at these seconds: {times}. Study every frame carefully; use those times for every "
                           "timestamp you return.")
            data = ai_fallback.chat_json(prompt, frames=frames)
            ai_source = (data or {}).get("_ai_model", "")

    if data is not None:
        data["verified_by_video"] = True
    else:
        # 3) nobody could look at the clip -> text-only copy from the title
        prompt = build("(You cannot see the video this time - base everything on the title, keep descriptions "
                       "general, and set hype_score to 6.)")
        data = _generate_json(prompt)
        if data is None and ai_fallback.available():
            data = ai_fallback.chat_json(prompt)
        ai_source = "text-only"
        if data is not None:
            data["verified_by_video"] = False

    if data and not _raw_lines(data) and not _shots(data, clip_dur):
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
    if out["verified_by_video"] and ai_source not in ("built-in", "text-only"):
        out.update(polish_packaging(out, out.get("scene_summary", "") + " | " + out.get("script", ""), fmt, tour))
    if ai_source == "built-in":
        out["commentary_raw"] = []
    log.info("[ai] copy written by: %s", ai_source)
    return out


# Real-world companies / brands. GTA 6 uses its own parody brands, and naming real ones in a video
# can get a post flagged as undisclosed branded content (Instagram). Lines that say one are dropped.
REAL_BRANDS = (
    "ford", "chevrolet", "chevy", "dodge", "jeep", "ram trx", "cadillac", "gmc", "buick", "lincoln", "tesla",
    "toyota", "lexus", "honda", "acura", "nissan", "infiniti", "mazda", "subaru", "mitsubishi", "hyundai",
    "kia", "bmw", "mercedes", "benz", "audi", "volkswagen", "vw", "porsche", "ferrari", "lamborghini",
    "maserati", "bugatti", "mclaren", "bentley", "rolls royce", "rolls-royce", "range rover", "land rover",
    "jaguar", "aston martin", "volvo", "harley", "harley-davidson", "ducati", "yamaha", "kawasaki", "suzuki",
    "nike", "adidas", "gucci", "louis vuitton", "supreme", "coca-cola", "coca cola", "pepsi", "red bull",
    "mcdonald's", "mcdonalds", "burger king", "starbucks", "walmart", "apple", "iphone", "samsung", "rolex",
    "playstation", "ps5", "xbox", "sony", "microsoft", "nvidia", "amazon", "netflix", "uber",
)
_BRAND_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(b) for b in REAL_BRANDS) + r")(?![a-z0-9])", re.I)


def _yes(v) -> bool:
    """A yes/no from an AI: only a real true (or "true" / "yes") - any other text ("bool", "maybe") is no. A copied
    answer form said "is_gta6_footage": "bool", which bool() read as yes."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes")
    return bool(v)


def _int(v, default: int) -> int:
    """A number from an AI, or the default ("hype_score": "int" crashed a live run)."""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _has_brand(text: str) -> bool:
    return bool(_BRAND_RE.search(text or ""))


def _strip_brands(text: str, fallback: str = "") -> str:
    """Drop every sentence that names a real brand (cutting just the word leaves broken text)."""
    if not _block_brands() or not _has_brand(text):
        return text
    kept = [x.strip() for x in re.findall(r"[^.!?]+[.!?]*", text or "") if x.strip() and not _has_brand(x)]
    return " ".join(kept) or fallback


def _block_brands() -> bool:
    """Off by default: naming cars / brands in scripts is fine (config quality_gate.block_brand_names)."""
    return bool(load_config().get("quality_gate", {}).get("block_brand_names", False))


def _drop_brand_lines(lines: list[dict], key: str = "text") -> list[dict]:
    if not _block_brands():
        return lines
    kept = [ln for ln in lines if not _has_brand(ln.get(key, ""))]
    if len(kept) < len(lines):
        log.info("[ai] dropped %d line(s) naming real brands: %s", len(lines) - len(kept),
                 [ln.get(key, "")[:60] for ln in lines if _has_brand(ln.get(key, ""))])
    return kept


def _shots(d: dict, reel_dur: float) -> list[dict]:
    """Feature-tour shots [{"start": sec, "text": line}] in story order (bad/duplicate shots dropped)."""
    raw = d.get("shots") or d.get("segments") or d.get("scenes") or []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        text = re.sub(r"[*_#\[\]()]|\s+", " ", str(item.get("line") or item.get("text") or "")).strip()
        try:
            ts = str(item.get("start", item.get("t", ""))).strip().rstrip("s")
            start = int(ts.split(":")[-2]) * 60 + float(ts.split(":")[-1]) if ":" in ts else float(ts)
        except (TypeError, ValueError, IndexError):
            continue
        if not text or not 0 <= start < reel_dur - 1:
            continue
        if any(abs(start - o["start"]) < 2.5 for o in out):
            continue
        words = text.split()
        if len(words) > 40:
            text = " ".join(words[:40]).rstrip(",;:") + "."
        out.append({"start": start, "text": text})
    return _drop_brand_lines(out)[:14]


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
You are a clear, confident guide who knows the game, narrating this clip like a top GTA tips Shorts channel:
friendly spoken English, one useful fact per line, never slang like "bro". {amount} It flows like one explanation:
hook (the useful promise) -> what you are seeing -> how it works -> the result -> ending question.
Line 1 at t=0 is a hook (max 12 words) that creates curiosity about what's about to happen.
The last line, about 4 seconds before the end, is a short question or call to action.
Every line must be about something specific in this clip - no generic filler like "Rockstar cooked".
Don't repeat the game's name in every line (at most once, as "GTA 6"). No emojis or hashtags.
Return ONLY JSON: {{"commentary": [{{"t": seconds, "line": "words"}}]}}"""


def rescue_commentary(copy: dict, length: float) -> list[dict]:
    ctype = copy.get("content_type", "action")
    words = int(length * 2.4)
    amount = {"stream": "Write 3-4 lines (about 50 words) and leave room for the streamer's reaction."}.get(
        ctype, f"Write 6-9 lines of 1-2 spoken sentences each, about {words} words in total.")
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


GROUND_PROMPT = """You are the narrator-writer of a GTA 6 YouTube Shorts / TikTok channel. {facts}
The video is: {what}
Above, for each numbered narration line, are frames from the exact footage that will be on screen WHILE that line
is spoken. Rewrite every line so it talks ONLY about what its own frames show (the action, place, vehicle, character,
result you can SEE). Rules:
- keep the line's job: the first line stays a hook, a line starting "Number N" keeps exactly that start, a final
  question stays a question to the viewer
- casual, punchy YouTuber voice, 5-18 words, say "GTA 6", no story spoilers, no hashtags or emojis
- never mention anything that is not visible in that line's frames (no "highway" if the frames show a street, no
  "cruise ship" if they show a truck); never invent numbers, prices or names
- if a line already matches its frames, return it unchanged
Current lines:
{lines}
Return ONLY JSON: {{"lines": [{{"line": number, "text": str}}]}}"""


def ground_lines(clip: Path, items: list[dict], what: str, extra: str = "") -> list[str] | None:
    """Rewrite narration lines so each one describes the footage it plays over. items: [{"text", "start", "dur"}]
    with times inside `clip`. Returns the new texts (same order) or None (no AI / bad answer -> keep the old ones).
    Before this, the AI wrote all lines at once from a long low-res proxy and its timing drifted: "chaos on the
    highway" over a street fight, "running from cops on a cruise ship" over a truck (QA line check)."""
    import subprocess
    import ai_fallback
    if not items or not (env("GEMINI_API_KEY") or ai_fallback.available()):
        return None
    dur = probe(clip)["duration"] or 0
    groups, listing = [], []
    for i, it in enumerate(items, 1):
        a = max(0.0, float(it["start"]))
        d = max(1.0, float(it.get("dur") or 2.5))
        imgs = []
        for t in (a + min(0.4, d * 0.2), a + d * 0.5, a + d * 0.85):
            res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{min(t, max(0.0, dur - 0.2)):.2f}", "-i", str(clip),
                                  "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", "-f", "image2pipe",
                                  "-vcodec", "mjpeg", "-"], capture_output=True, timeout=60)
            if res.returncode == 0 and len(res.stdout) > 1500:
                imgs.append(res.stdout)
        if not imgs:
            return None
        groups.append((f"Line {i} frames", imgs))
        listing.append(f'{i}. "{it["text"]}"')
    prompt = GROUND_PROMPT.format(facts=_game_context(), what=what[:300], lines="\n".join(listing))
    if extra:  # lessons from our own QA, or why these exact lines were rejected
        prompt += "\n" + extra[:1500]
    import vision  # Gemini first, then the backup AIs that can see images
    data = vision.ask(groups, prompt, key="lines", num="line", temperature=0.5, prefer_pro=False)
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        return None
    by = {}
    for v in data["lines"]:
        try:
            by[int(v.get("line"))] = re.sub(r"\s+", " ", str(v.get("text") or "")).strip().strip('"')
        except (TypeError, ValueError, AttributeError):
            continue
    out = []
    for i, it in enumerate(items, 1):
        new, old = by.get(i) or "", it["text"]
        num_old = re.match(r"\s*(number\s+\w+|#\d+)", old, re.I)
        if not (3 <= len(new.split()) <= 24) or "GTA VI" in new:
            new = old
        elif num_old and not new.lower().startswith(num_old.group(1).lower()):
            body = re.sub(r"^(number\s+\w+|#\d+)[:,.]?\s*", "", new, flags=re.I)
            new = f"{num_old.group(1)}: {body}"
        elif old.rstrip().endswith("?") and not new.rstrip().endswith("?"):
            new = old  # the closing question stays a question
        elif set(re.findall(r"\d+", new)) - set(re.findall(r"\d+", old)) - {"6"}:
            new = old  # no invented numbers
        out.append(new)
    changed = sum(1 for a, b in zip(out, items) if a != b["text"])
    log.info("[script] lines grounded on their footage: %d of %d rewritten", changed, len(items))
    return out


def script_ok(lines: list[dict], kind: str, ai_source: str) -> tuple[bool, str]:
    """Only post videos with a real script written by an AI that actually saw the clip."""
    g = load_config().get("quality_gate", {})
    if ai_source in ("built-in", "text-only", ""):
        return False, f"no AI could watch the clip and write the script ({ai_source or 'none'})"
    n_words = sum(len(ln.get("text", "").split()) for ln in lines)
    if kind == "stream":
        need_lines, need_words = int(g.get("min_script_lines_stream", 2)), int(g.get("min_script_words_stream", 12))
    else:
        need_lines, need_words = int(g.get("min_script_lines", 4)), int(g.get("min_script_words", 35))
    if len(lines) < need_lines or n_words < need_words:
        return False, f"script too thin ({len(lines)} lines, {n_words} words; need {need_lines}+ / {need_words}+)"
    return True, ""


LIMITS = {  # content_type -> (max lines, max words per line) - a full narration script, not just reactions
    "explainer": (10, 26), "action": (10, 26), "scenic": (8, 26), "stream": (5, 22),
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
    lines = _drop_brand_lines(lines)
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


def polish_packaging(copy: dict, context: str, fmt: dict | None = None, tour: bool = False) -> dict:
    """Second opinion from the critic AI on title / banner / caption / X text (see critic.py), then the packaging
    rules on every copy field: "GTA 6" first in the title, the real countdown number, money never sounds like
    cheats, the format's search words are kept."""
    import critic
    fmt = fmt or {}
    money = fmt.get("id") == "money_tips" or critic.is_money(fmt.get("keywords", ""), copy.get("youtube_title", ""),
                                                             copy.get("banner", ""), copy.get("caption", ""))
    count = int(fmt["countdown"]) if tour and str(fmt.get("countdown", "")).isdigit() else None
    keep = critic.keyword_phrases(fmt.get("keywords", ""))
    fields = {k: copy.get(k, "") for k in ("youtube_title", "banner", "caption", "x_text") if copy.get(k)}
    better = critic.polish(fields, {"youtube_title": 70, "banner": 40, "caption": 220, "x_text": 200}, context,
                           money=money, keep=keep, count=count)
    title = better.get("youtube_title", "")
    if fmt.get("id") == "money_tips" and title and not (critic.has_term(title, "money") or critic.has_term(title, "cash")):
        rest = re.sub(r"^GTA 6\s*[:\-–|]?\s*", "", title)  # "money" is what people search: a Money Tip title says it
        better["youtube_title"] = critic.trim(f"GTA 6 Money Tip: {rest}", 70)
    if copy.get("description"):  # not worth a critic call, but the rules apply
        better.update(critic.apply_rules({"description": copy["description"]}, money))
    if "banner" in better:
        better["banner"] = re.sub(r"[^A-Z0-9 !?'.,-]", "", better["banner"].upper())[:48].strip() or copy.get("banner", "")
    return better


def packaging_note(fmt: dict | None) -> str:
    """Search words for the title / caption: the format's main search phrase + what GTA 6 players type into
    search right now (trend radar), only the phrases that fit this format."""
    import critic
    fmt = fmt or {}
    about = " ".join(str(fmt.get(k) or "") for k in ("keywords", "topic", "name"))
    phrases = critic.relevant_searches(about, limit=6, min_cover=0.5) if about.strip() else []
    if len(phrases) < 3:
        phrases += [p for p in critic.top_searches(8) if p not in phrases][: 6 - len(phrases)]
    bits = []
    main = critic.keyword_list(fmt.get("keywords", ""))
    if len(main) == 1:
        bits.append(f'Main search phrase of this format: "{main[0]}".')
    elif main:  # stacking them reads like spam ("GTA 6 police chase car chase at the airport")
        bits.append("Main search phrases of this format: " + " / ".join(f'"{k}"' for k in main)
                    + " - use ONE of them where it reads naturally, never several in a row.")
    if phrases:
        bits.append("GTA 6 players search for these right now: " + "; ".join(f'"{p}"' for p in phrases)
                    + " - use one word for word in youtube_title / caption ONLY if it honestly fits what the clip shows.")
    return ("SEARCH WORDS: " + " ".join(bits)) if bits else ""


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
    try:
        payoff_t = float(d.get("payoff_t")) if d.get("payoff_t") is not None else None
    except (TypeError, ValueError):
        payoff_t = None
    return {
        "is_gta6_footage": _yes(d.get("is_gta6_footage", True)),
        "has_facecam": _yes(d.get("has_facecam", False)),
        "has_narrator": _yes(d.get("has_narrator", False)),
        "is_spoiler": _yes(d.get("is_spoiler", False)),
        "unsafe": d.get("unsafe") is True,
        "fits_format": _yes(d.get("fits_format", True)),
        "shows_required": d.get("shows_required") is True,
        "required_t": d.get("required_t") if isinstance(d.get("required_t"), (int, float)) else None,
        "required_evidence": re.sub(r"\s+", " ", str(d.get("required_evidence") or "")).strip()[:200],
        "content_type": str(d.get("content_type", "action")).lower(),
        "hype_score": _int(d.get("hype_score"), 6),
        "scene_summary": s("scene_summary"),
        "verified_by_video": bool(d.get("verified_by_video", False)),
        "banner": re.sub(r"[^A-Z0-9 !?'.,-]", "", _strip_brands(s("banner", "INSANE GTA 6 MOMENT"), "INSANE GTA 6 MOMENT").upper())[:48].strip() or "INSANE GTA 6 MOMENT",
        "commentary_raw": raw,
        "shots": _shots(d, clip_dur),
        "angle": s("angle"),
        "clip_start": clip_start, "clip_end": clip_end, "payoff_t": payoff_t,
        "script": " ".join(ln["text"] for ln in (_shots(d, clip_dur) or raw)),
        "youtube_title": _strip_brands(s("youtube_title", "This GTA 6 moment is unreal"), "This GTA 6 moment is unreal")[:80],
        "caption": _strip_brands(s("caption", "This GTA 6 moment goes crazy. Would you try this?"), "This GTA 6 moment goes crazy. Would you try this?"),
        "x_text": _strip_brands(s("x_text", "This GTA 6 moment goes crazy"), "This GTA 6 moment goes crazy")[:240],
        "description": _strip_brands(s("description"), "")[:500],
        "hashtags": list(dict.fromkeys(tags))[:5],
    }


# --------------------------------------------------------------------------- fallback copy

_HOOKS = [
    "GTA 6 is full of details like this.", "Okay, this is NOT normal!", "Wait for it...",
    "Rockstar did NOT hold back!", "Nobody was ready for this!", "This is GTA 6?! No way!",
    "Watch this. Just watch.", "Vice City is wild, man!",
]
_REACTIONS = [
    "NO WAY!", "Look at that!", "Oh my god, the physics!", "He's not stopping!", "Rockstar cooked!",
    "That's crazy!", "Look at the detail on this!", "This looks like a movie!", "Did you see that?!",
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
    "x_text_morning": "a good-morning engagement post: ONE fun question or hot take about GTA 6 gameplay (Leonida, "
                      "cars, money, cops, side activities, characters) that people want to reply to - best about a "
                      "hot topic below",
    "x_question_nostalgia": "a nostalgia post comparing a GTA memory (GTA III, Vice City, San Andreas, GTA IV, GTA V) "
                            "with GTA 6, ending with a question, OR a 'this or that' poll-style question",
    "x_daily_recap": "an end-of-day recap post for our followers: today's GTA 6 vibe in one line + the best clip we "
                     "posted (only from the list below - never made-up clips), ending with a teaser for tomorrow",
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
    try:  # what GTA 6 players search for / talk about today (trend radar) - posts that join the conversation
        import trends
        hot = trends.context_line()
    except Exception:  # noqa: BLE001
        hot = ""
    hot = f"\nWHAT PLAYERS TALK ABOUT TODAY: {hot} (use it if it fits - a question or poll about a hot topic gets " \
          "the most replies)" if hot else ""
    import critic
    searches = critic.top_searches(8)
    searches = f"\n- if one of these search phrases fits naturally, use its words (people search them): {'; '.join(searches)}" \
        if searches else ""
    prompt = f"""You run the X account of "{load_config().get('brand', {}).get('display_name', 'GTA VI Top Moments')}".
CONTEXT: {_game_context()}{hot}
Write {TEXT_KINDS[kind]}.
Clips we posted today: {todays_clips or 'none yet'}.
Rules (X rewards short, punchy posts that start a conversation):
- max 200 characters (best 70-150); the first words are the hook; conversational, like a real GTA fan
- at most 1-2 emojis, NO hashtags (we add them), NO links, no @mentions
- say "GTA 6", never "GTA VI"; no story spoilers (plot, mission outcomes, endings, character deaths)
- never invent news, leaks, prices, numbers or release details; no glitch / exploit / mod talk{searches}
Must be clearly different from these recent posts:
{recent_texts[-12:]}
Return ONLY JSON: {{"text": "..."}}"""
    data = _generate_json(prompt, temperature=1.0)
    if data is None and ai_fallback.available():
        data = ai_fallback.chat_json(prompt, temperature=1.0)
    text = re.sub(r"\s+", " ", str((data or {}).get("text", ""))).strip()
    text = critic.norm_gta(re.sub(r"https?://\S+|www\.\S+", "", text)).strip()
    bad = critic.spoiler_words(text, story_only=True) + [w for w in critic.cheat_words(text) if not w.startswith("cheat")]
    if bad:
        log.info("[text] AI post dropped (%s): %s", ", ".join(bad), text)
    elif 15 <= len(text) <= 240 and text not in recent_texts:
        return text
    pool = [t for t in _TEXT_FALLBACK[kind] if t not in recent_texts] or _TEXT_FALLBACK[kind]
    return random.choice(pool)


# --------------------------------------------------------------------------- whole-video script (owner, 2026-11-22)
# "the writing shall occur after the AI watched the whole video, then write a good script for the whole video - not
# frame by frame". Rewriting each line from 3 still frames (ground_lines) turned scripts into captions of frames
# ("Number seven: running across the sunlit road toward a pickup truck") - posted videos made no sense.

FINAL_PROMPT = """GAME FACTS (newer than your training data - trust them): {facts}

You are the head writer of a top GTA 6 tips & guides YouTube Shorts / TikTok channel (millions of views).
WATCH THE WHOLE VIDEO ABOVE FIRST, with the sound, from the first to the last second. Then write the narration for
the WHOLE video.

The video is {length:.0f} seconds long. {lead_note}
Format: {fmt}
Banner on screen: "{banner}" | Title: "{title}"
Notes from the editor (may be wrong - trust what you watched): {summary}
{dialogue}
Write it like the best GTA guide creators:
- ONE clear explanation from start to finish: hook (the useful promise) -> what you are seeing -> how it works /
  what to do -> the payoff (what you get, the result) -> a question. Every line follows from the line before;
  someone who only HEARS the narration learns something they can use in their own game
- the first line is a hook that THIS video really pays off. Never promise anything the video doesn't show (no "how
  to steal a car" if no car gets stolen). If the video doesn't show the format's topic, write about what it DOES
  show (honesty beats the topic) and give a new banner and title that this video delivers
- every line talks about what is happening at ITS moment in the video (you watched it - use the real timeline).
  NEVER talk about anything the video does not SHOW: no item, place, number, menu or result that is not on screen
  (owner: "you can't talk about something without it being shown"). If the video doesn't show it, don't say it
- USEFUL like the top guide creators - this is what makes people watch to the end, save and share:
  * every line gives ONE concrete fact or tip about what is on screen at its moment: how it works, where it is,
    what you get, what it costs, what to avoid, how it differs from GTA 5
  * the most useful or surprising fact lands right before the closing question
  * NEVER a dry list of what is on screen ("running across the road toward a pickup truck" is a bad line) and
    NEVER jokes, roasts or hype with no information ("He chose war", "absolute chaos", "bro" - all bad).
    Good lines: "Rob the register first - the safe in the back only opens after the alarm is off." / "Those
    icons under your stars show what the cops know: your clothes, your car, even camera footage." / "Change
    your outfit and the police lose your description." (examples of the style - never copy them)
  * clear, confident, friendly spoken English - like a friend who knows the game explaining it, never slang
    like "bro" or "nah"
- talk to the viewer ("you") or about Jason / Lucia - never pretend to be the player ("I tried...", "my car")
- {numbering}
- vary the words (don't repeat the same noun in every line); say "GTA 6" at most once
- {amount}
- the last line, about 3-4 seconds before the end, is ONE short question to the viewer about THIS video (max 14
  words, nothing else in that line - never tack the question onto a long sentence)
- no story spoilers, no hashtags, no emojis
{slots}{feedback}
{hints}
{shape}"""

ONE_SHAPE = """Return ONLY JSON: {{"story": "one sentence: what really happens in this video", "lines": [{{"t": seconds,
"text": str}}], "banner": "4-7 word banner this video delivers (or the current one)", "title": "YouTube title this
video delivers, with GTA 6 in it (or the current one)"}}"""

MANY_SHAPE = """Write {n} DIFFERENT complete scripts for this video, each with its own angle (1: step-by-step guide,
2: the surprising fact first, 3: the mistake most players make -> how to do it right) - each must be good enough to
post on its own.
Return ONLY JSON: {{"story": "one sentence: what really happens in this video", "scripts": [{{"angle": str,
"lines": [{{"t": seconds, "text": str}}], "banner": "4-7 word banner this video delivers", "title": "YouTube title
this video delivers, with GTA 6 in it"}}]}}"""

SLOTS_NOTE = """THE EDIT IS FIXED: there are exactly {n} narration slots, at these times (with the current lines - rewrite
them all as one story, keep each slot's time and keep each line short enough for its slot):
{rows}
Return exactly {n} lines, in this order, with these times.
"""


def _spans_note(spans, length: float) -> str:
    spans = [(round(a, 1), round(b, 1)) for a, b in (spans or []) if b > a]
    if not spans:
        return ""
    talk = sum(b - a for a, b in spans)
    return (f"The characters in the video talk for {talk:.0f} of {length:.0f} s, at: "
            + ", ".join(f"{a}-{b}s" for a, b in spans[:20])
            + ". Place the narration in the quiet moments between them (never talk over the characters).\n")


def _fit_lines(lines: list[dict], length: float, lead: float, wps: float = 2.7) -> list[dict]:
    """Order the lines, start the hook right after the cold open and keep each line from running into the next."""
    out = []
    for ln in sorted(lines, key=lambda x: x["t"]):
        t = max(ln["t"], lead + 0.1 if not out else out[-1]["t"] + len(out[-1]["text"].split()) / wps + 0.35)
        out.append({"t": round(t, 2), "text": ln["text"]})
    if out and out[0]["t"] - lead <= 3.0:
        out[0]["t"] = round(lead + 0.1, 2)
    while len(out) > 2 and out[-1]["t"] + len(out[-1]["text"].split()) / wps > length - 0.3:
        last = out.pop()  # the closing question must fit: it replaces the line before it if needed
        if len(out) > 2:
            out.pop()
        prev_end = out[-1]["t"] + len(out[-1]["text"].split()) / wps + 0.35
        t = max(prev_end, length - len(last["text"].split()) / wps - 0.6)
        out.append({"t": round(t, 2), "text": last["text"]})
    return out


def _fewer_slots(lines: list[dict], slots: list[dict]) -> list[dict] | None:
    """The writer gave fewer lines than the fixed edit has slots (a backup wrote 5 lines for 8 shots and the clip was
    dropped, live 2026-11-25): each line takes the nearest free slot after the previous one (by its own time), the
    unused slots are left out - the caller drops their shots. None when there are too few lines (< 3) or more lines
    than slots."""
    if not 3 <= len(lines) < len(slots):
        return None
    out, nxt = [], 0
    for k, ln in enumerate(lines):
        last_ok = len(slots) - (len(lines) - k)  # leave a slot for every line still to come
        cands = range(nxt, last_ok + 1)
        i = min(cands, key=lambda j: abs(float(slots[j]["t"]) - ln["t"]))
        out.append({"t": slots[i]["t"], "text": ln["text"]})
        nxt = i + 1
    return out


def _parse_script(d: dict, length: float, lead: float, slots: list[dict] | None) -> dict | None:
    raw = d.get("lines") or d.get("commentary") or []
    lines = []
    for ln in raw if isinstance(raw, list) else []:
        try:
            t = float(ln.get("t"))
        except (TypeError, ValueError, AttributeError):
            continue
        text = re.sub(r"\s+", " ", str(ln.get("text") or ln.get("line") or "")).strip().strip('"')
        text = re.sub(r"#\w+", "", text).strip()
        if 2 <= len(text.split()) <= 32 and -0.5 <= t <= length - 0.8:
            lines.append({"t": max(0.0, t), "text": text})
    if slots:
        if len(lines) != len(slots):
            fitted = _fewer_slots(lines, slots)
            log.warning("[script] the writer returned %d lines for %d fixed slots%s", len(lines), len(slots),
                        " - each line goes to its nearest slot, the others are dropped" if fitted else "")
            if not fitted:
                return None
            lines = fitted
        else:
            lines = [{"t": s["t"], "text": ln["text"]} for s, ln in zip(slots, lines)]
    else:
        if len(lines) < 3:
            return None
        lines = _fit_lines(lines, length, lead)
    banner = re.sub(r"\s+", " ", str(d.get("banner") or "")).strip().strip('"')
    title = re.sub(r"\s+", " ", str(d.get("title") or "")).strip().strip('"')
    return {"lines": lines, "banner": banner if 2 <= len(banner.split()) <= 9 else "",
            "title": title if 3 <= len(title.split()) <= 16 and re.search(r"gta\s*(6|vi)", title, re.I) else "",
            "angle": str(d.get("angle") or "")[:40]}


def write_final_script(clip: Path, copy: dict, fmt: dict | None = None, lead: float = 0.0, spans=None,
                       feedback: str = "", slots: list[dict] | None = None, n: int = 1) -> dict | None:
    """The AI watches the WHOLE final video (with sound) and writes ONE script for it.
    slots=[{"t", "text"}]: the edit is fixed (tours / repairs) - one line per slot, same times.
    Gemini busy / out of quota: a backup AI that sees images writes it from frames (_script_from_frames).
    Returns {"lines": [{"t", "text"}], "banner", "story", "model"} or None (no AI could look at the video)."""
    global _unavailable_until
    if not env("GEMINI_API_KEY") and not ai_fallback.available():
        return None
    length = probe(clip)["duration"] or 0
    if length <= 3:
        return None
    quiet = length - lead - sum(max(0.0, b - a) for a, b in (spans or []))
    words = int(max(25, min(150, quiet * 2.3)))
    amount = (f"FILL THE VIDEO like top creators (they talk almost the whole time): about {words} words in total, "
              f"{max(5, words // 14)}-{max(6, words // 8)} lines of 1-2 spoken sentences (max 24 words each), never more "
              f"than 5 seconds without a line unless the characters are talking") if not slots else \
        "one line per slot, max 24 words each"
    slot_txt = SLOTS_NOTE.format(n=len(slots), rows="\n".join(f'{i}. t={s["t"]:.1f}s: "{s["text"]}"'
                                                              for i, s in enumerate(slots, 1))) if slots else ""
    prompt = FINAL_PROMPT.format(
        facts=_game_context(), length=length,
        lead_note=(f"The first {lead:.1f} s are a cold open (a flash of the big moment), then the story starts - the "
                   f"hook line starts at {lead + 0.1:.1f}s." if lead else "The hook line starts at 0.1s."),
        fmt=f"{(fmt or {}).get('name', 'GTA 6 Short')}. {(fmt or {}).get('brief', '')}".strip()[:600],
        banner=copy.get("banner", ""), title=copy.get("youtube_title", ""), hints=_writer_hints(fmt), summary=str(copy.get("scene_summary", ""))[:400],
        dialogue=_spans_note(spans, length), amount=amount, slots=slot_txt,
        numbering=("numbered items: the edit already has its slots - keep each slot's role; if it is a countdown, "
                   "count down to number one and say every number once" if slots else
                   "THIS IS ONE CONTINUOUS SCENE, NOT A LIST: never number anything (no \"Number 5\"), even if the "
                   "format's name says Top 5 - tell it as one story"),
        shape=MANY_SHAPE.format(n=n) if n > 1 and not slots else ONE_SHAPE,
        feedback=("\n" + feedback.strip() + "\n") if feedback else "")
    data, model = None, "gemini-video"
    if env("GEMINI_API_KEY"):
        _unavailable_until = 0.0
        try:
            up = _upload_video(clip)
        except Exception as exc:  # noqa: BLE001
            log.warning("[script] video upload failed: %.150s", exc)
            up = None
        if up is not None:  # the upload is reused by the next attempt
            data = _generate_json([up, prompt], temperature=0.8, prefer_pro=True)
    if not isinstance(data, dict):
        data = _script_from_frames(clip, prompt, length, slots)
        model = (data or {}).get("_ai_model", "")
    if not isinstance(data, dict):
        return None
    many = data.get("scripts") if isinstance(data.get("scripts"), list) else [data]
    found = [x for x in (_parse_script(d, length, lead, slots) for d in many if isinstance(d, dict)) if x]
    if not found:
        return None
    story = str(data.get("story") or many[0].get("story") or "")[:300]
    for x in found:
        x.update(story=story, model=model)
    return {**found[0], "alternatives": found[1:]}


FRAMES_SCRIPT_NOTE = """YOU CANNOT WATCH OR HEAR THIS VIDEO: you get {n} still frames from it instead, in order - {legend}.
Wherever the brief below says watch / listen, study these frames: write only about what they show (and what the
notes say), and never invent a sound, a line of dialogue or an event you can't see.

"""


def _script_from_frames(clip: Path, prompt: str, length: float, slots: list[dict] | None) -> dict | None:
    """Gemini busy / out of quota (runs of 2026-11-24 posted nothing): a backup AI that sees images writes the script
    from frames - one inside each narration slot for a fixed edit, else spread across the video. Every check after
    it (judge, picture check of each line, final QA) is the same as for a Gemini script."""
    gcfg = load_config().get("gemini", {})
    if not gcfg.get("script_backup", True) or not ai_fallback.available() or length <= 3:
        return None
    n = int(gcfg.get("script_backup_frames", 8))  # 8: what most image-capable backups take in one request
    if slots:
        pick = sorted({round(i * (len(slots) - 1) / max(1, n - 1)) for i in range(n)}) if len(slots) > n \
            else list(range(len(slots)))
        times = [min(length - 0.2, float(slots[i]["t"]) + 1.0) for i in pick]
        legend = ", ".join(f"image {k + 1} = slot {i + 1} ({t:.1f}s)" for k, (i, t) in enumerate(zip(pick, times)))
    else:
        times = [length * (k + 0.5) / n for k in range(n)]
        legend = ", ".join(f"image {k + 1} = {t:.1f}s" for k, t in enumerate(times))
    frames = ai_fallback.extract_frames(clip, times=times)
    if len(frames) != len(times):
        return None
    data = ai_fallback.chat_json(FRAMES_SCRIPT_NOTE.format(n=len(frames), legend=legend) + prompt, frames=frames,
                                 temperature=0.8, all_frames=True)
    if isinstance(data, dict):
        log.info("[script] Gemini unavailable - %s wrote the script from %d frames", data.get("_ai_model", "?"),
                 len(frames))
    return data


def _writer_hints(fmt: dict | None) -> str:
    """Our own memory: recent script mistakes + lines that passed, and what got the most / fewest views."""
    out = []
    try:
        import lessons
        out.append(lessons.prompt_hints((fmt or {}).get("id")))
    except Exception:  # noqa: BLE001
        pass
    try:
        import analytics
        out.append(analytics.view_examples())
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(x for x in out if x)

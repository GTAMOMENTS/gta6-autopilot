"""Strict QA: every finished video is checked before it is published. Nothing posts without passing.

  1. technical   - audio/video length (sync), silent stretches, frozen/choppy parts, black frames, clipping
  2. voice check - Whisper listens to the final video: is every narration line actually said, and said WHEN it is
                   on screen? (words + timing)
  3. AI editor   - Gemini watches + listens to the whole final video: montage errors, voice vs on-screen text,
                   narration over something else, sync, repeated clips, spoilers, non-English, non-GTA, unsafe
Verdict: pass / fail with a list of issues (+ timestamps). A report is written next to the video.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from common import env, load_config, probe, run

log = logging.getLogger("autopilot")


def _cfg() -> dict:
    return load_config().get("qa", {})


def _sdur(path: Path, sel: str) -> float:
    res = run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "stream=duration", "-of",
               "csv=p=0", str(path)], timeout=30)
    try:
        return float(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def _ranges(stderr: str, start_key: str, end_key: str) -> list[tuple[float, float]]:
    starts = [float(x) for x in re.findall(rf"{start_key}[:=]\s*([\d.]+)", stderr)]
    ends = [float(x) for x in re.findall(rf"{end_key}[:=]\s*([\d.]+)", stderr)]
    return list(zip(starts, ends + [starts[-1] + 999] * (len(starts) - len(ends)))) if starts else []


def _outside(r: tuple[float, float], allowed: list[tuple[float, float]], dur: float = 1e9) -> float:
    """Seconds of range r that are NOT inside the allowed (intentional) ranges (r capped at the video's end -
    a still part that lasts to the end has no 'end' timestamp)."""
    a, b = r[0], min(r[1], dur)
    covered = sum(max(0.0, min(b, y) - max(a, x)) for x, y in allowed)
    return max(0.0, (b - a) - covered)


def technical(path: Path, still_ok: list[tuple[float, float]] | None = None) -> list[dict]:
    """still_ok: ranges where a static picture / quiet audio is intentional (outro card...)."""
    c, issues = _cfg(), []
    still_ok = still_ok or []
    dur = probe(path)["duration"] or 0
    v, a = _sdur(path, "v:0"), _sdur(path, "a:0")
    if not a:
        issues.append({"severity": "high", "type": "no_audio", "detail": "the video has no sound track"})
    elif abs(v - a) > float(c.get("max_av_diff", 0.12)):
        issues.append({"severity": "high", "type": "av_length", "detail": f"audio {a:.2f}s vs video {v:.2f}s"})
    res = run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf",
               "scale=320:-2,freezedetect=n=0.003:d=1.2,blackdetect=d=0.4:pix_th=0.08",
               "-af", f"silencedetect=noise={c.get('silence_db', -45)}dB:d={c.get('silence_sec', 2.5)},volumedetect",
               "-f", "null", "-"], timeout=900)
    err = res.stderr
    for r in _ranges(err, "silence_start", "silence_end"):
        if _outside(r, still_ok, dur) > float(c.get("silence_sec", 2.5)):
            issues.append({"severity": "high", "type": "silence", "at": round(r[0], 1),
                           "detail": f"no sound for {min(r[1], dur) - r[0]:.1f}s"})
    for r in _ranges(err, "lavfi.freezedetect.freeze_start", "lavfi.freezedetect.freeze_end"):
        if _outside(r, still_ok, dur) > 1.2:
            issues.append({"severity": "high", "type": "frozen", "at": round(r[0], 1),
                           "detail": f"picture frozen for {min(r[1], dur) - r[0]:.1f}s"})
    for r in _ranges(err, "black_start", "black_end"):
        if _outside(r, still_ok, dur) > 0.4:
            issues.append({"severity": "medium", "type": "black", "at": round(r[0], 1),
                           "detail": f"black screen for {r[1] - r[0]:.1f}s"})
    peak = re.findall(r"max_volume:\s*(-?[\d.]+) dB", err)
    if peak and float(peak[-1]) >= -0.05:
        issues.append({"severity": "medium", "type": "clipping", "detail": f"audio peaks at {peak[-1]} dB (distortion)"})
    try:
        import stream_clips
        rep = stream_clips.smoothness(path)
        if rep > float(c.get("max_repeated_frames", 0.15)):
            issues.append({"severity": "high", "type": "choppy", "detail": f"{rep * 100:.0f}% repeated frames"})
    except Exception:  # noqa: BLE001
        pass
    return issues


def _norm_words(text: str) -> list[str]:
    """Same normalisation as the narration split: Whisper writes "14", the script says "fourteen"."""
    from video_processor import _norm_tokens
    return _norm_tokens(text.replace("*", ""))


_FILLER = {"the", "a", "an", "and", "to", "of", "in", "on", "it", "is", "you", "your", "this", "that", "so", "but",
           "just", "now", "then", "at", "for", "with", "i", "we", "he", "she", "they", "be", "are", "was", "if", "or"}


def _narration_stem(path: Path, lines: list[dict]) -> Path | None:
    """The narration alone, placed exactly as in the final mix (same delays, same length) - so Whisper checks
    what the narrator says and WHEN, without the streamer / game sound talking over it."""
    if not lines or not all(ln.get("path") and Path(ln["path"]).exists() for ln in lines):
        return None
    total = probe(path)["duration"] or 0
    inputs, chains = [], []
    for i, ln in enumerate(lines):
        ms = int(float(ln["start"]) * 1000)
        inputs += ["-i", str(ln["path"])]
        chains.append(f"[{i}:a]aresample=16000,aformat=channel_layouts=mono,adelay={ms}[v{i}]")
    out = path.with_name(path.stem + "_narration.wav")
    fc = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(lines))) + \
        f"amix=inputs={len(lines)}:normalize=0:duration=longest,apad[o]"
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", fc, "-map", "[o]", "-t", f"{total:.2f}",
               str(out)], timeout=300)
    return out if res.returncode == 0 and out.exists() else None


def voice_check(path: Path, lines: list[dict]) -> list[dict]:
    """lines: [{"text", "start", "dur", "path"?}] = narration placed in the final video. Whisper must hear each
    line (word match) close to where it is placed (timing). Uses the narration-only track when the voice files
    are known (the final mix has streamer / game sound under the voice)."""
    import transcribe
    if not lines:
        return []
    stem = _narration_stem(path, lines)
    try:
        heard, _ = transcribe.words(stem or path)
    finally:
        if stem:
            stem.unlink(missing_ok=True)
    if not heard:
        if transcribe._get_model() is None:
            return [{"severity": "low", "type": "voice_unchecked", "detail": "Whisper not available"}]
        return [{"severity": "high", "type": "voice_missing", "detail": "no speech heard in the final video"}]
    c, issues = _cfg(), []
    tol = float(c.get("max_voice_offset", 0.6))
    for ln in lines:
        want = _norm_words(ln["text"])
        if not want:
            continue
        a, b = ln["start"] - 1.5, ln["start"] + ln.get("dur", 3) + 1.5
        got = [w for w in heard if a <= w["start"] <= b]
        got_words = _norm_words(" ".join(w["text"] for w in got))
        covered = sum(1 for w in want if w in got_words) / len(want)
        if covered < float(c.get("min_word_match", 0.6)):
            issues.append({"severity": "high", "type": "voice_mismatch", "at": round(ln["start"], 1),
                           "detail": f"line not heard properly ({covered:.0%} of the words): \"{ln['text'][:60]}\""})
            continue
        # timing: when the line's FIRST words are actually heard vs when the line (and its caption) starts.
        # (The old estimate assumed an even speed over the whole line - "Number fourteen." + pause made every
        # line look ~1 s late.)
        # the first real word of the line (not "the" / "you" ... - those also end the previous line) and the
        # occurrence closest to where it should be
        anchors = [i for i, x in enumerate(want[:6]) if x not in _FILLER][:3]
        k = anchors[0] if anchors else None
        spoken = None
        if k is not None:
            # wide search (narration 1-3 s off must be caught - the old -1.0/+2.5 s window let it pass), the
            # occurrence closest to where the word should be wins (neighbouring lines can share words); if the
            # first real word isn't heard (Whisper spelling), the next real words are tried
            lo, hi = ln["start"] - 3.0, ln["start"] + max(3.5, 0.6 * float(ln.get("dur", 3)))
            for k in anchors:
                expect = ln["start"] + 0.32 * k
                # Whisper stretches the first word after a pause back into the silence ("Look@6.6 at@8.4"): a word
                # lasting over ~0.6 s really starts near its end
                hits = [max(w["start"], float(w.get("end", w["start"])) - 0.6) for w in heard
                        if lo <= w["start"] <= hi and want[k] in _norm_words(w["text"])]
                if hits:
                    spoken = min(hits, key=lambda t: abs(t - expect)) - 0.32 * k
                    break
            if spoken is None:
                issues.append({"severity": "high", "type": "voice_out_of_sync", "at": round(ln["start"], 1),
                               "detail": f"the line's first words aren't heard near its caption: \"{ln['text'][:50]}\""})
                continue
        if spoken is not None and abs(spoken - ln["start"]) > tol + (0.25 if k else 0):
            near = " ".join(f"{w['text']}@{w['start']:.1f}" for w in heard
                            if ln["start"] - 3.0 <= w["start"] <= ln["start"] + 2.0)[:160]
            issues.append({"severity": "high", "type": "voice_out_of_sync", "at": round(ln["start"], 1),
                           "detail": f"spoken {spoken - ln['start']:+.2f}s off the captions: \"{ln['text'][:50]}\" "
                                     f"(heard: {near})"})
    return issues


AI_PROMPT = """You are the strict quality-control editor of a GTA 6 YouTube/TikTok channel. The attached file is a
FINISHED video that is about to be published. Watch AND listen to all of it, then report every problem a viewer
would notice. Be strict - a professional channel only publishes clean edits.
GAME FACTS (newer than your training data - trust them): {game}
Do NOT reject footage just because you believe GTA 6 is unreleased - it is out. Flag "not GTA 6" only when the
footage clearly shows a DIFFERENT game (GTA V's Los Santos landmarks / characters / minimap, another game's UI).
{context}
Check for:
- montage errors: abrupt or broken cuts, a cut in the middle of a sentence or action, the same clip shown twice,
  black or frozen frames, glitches, awkward empty parts
- sound: audio out of sync with the picture, silent parts, sudden loud/quiet jumps, distortion
- narration: the voice-over not saying what the on-screen text / subtitles say, narration out of sync with the
  captions, narration talking over the streamer's important line or the punchline, narration describing
  something different from what is on screen
- LINE BY LINE: the narration lines are listed below with their start times. For EACH line, look at the picture
  at that time: it must show what the line talks about. A line that explains an action / place while the video
  shows something unrelated (e.g. "loot the safe" over a car driving down a road) is a HIGH severity montage error.
  Also flag a cut to unrelated footage in the middle of an explanation.
- content: not GTA 6, story spoilers, non-English streamer, nudity / slurs / real violence
- overall: would a professional editor publish this as is?
Then, separately, judge it against the edits of TOP gaming YouTubers / TikTokers (pro_level 1-10) and list the
concrete changes that would close the gap (improvements): hook in the first 2 seconds, pacing / dead moments,
cut rhythm, zooms / emphasis, caption style and readability, narration energy and wording, sound design and
music, thumbnail-worthy moments, ending / call to action. Be specific ("cut the first 3 s, start on the crash").
Return ONLY JSON: {{"publish": bool, "score": int (1-10), "issues": [{{"at": seconds, "severity": "high"|"medium"|"low",
 "type": str, "detail": str}}], "summary": str, "pro_level": int (1-10), "improvements": [str]}}"""


def ai_review(path: Path, context: str) -> dict | None:
    """The AI editor: Gemini watches + listens to the whole video; if Gemini can't (busy / out of quota), a backup
    AI reviews frames sampled across the video + the narration with its times."""
    import gemini_writer
    data = None
    if env("GEMINI_API_KEY"):
        try:
            gemini_writer._unavailable_until = 0.0
            up = gemini_writer._upload_video(path)
            if up is not None:
                data = gemini_writer._generate_json([up, AI_PROMPT.format(game=gemini_writer._game_context(),
                                                                          context=context[:3000])],
                                                    temperature=0.2, prefer_pro=True)
                try:
                    gemini_writer._client().files.delete(name=up.name)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            log.warning("[qa] AI review failed: %.200s", exc)
    if isinstance(data, dict) and "publish" in data:
        return data
    return _ai_review_frames(path, context)


FRAMES_NOTE = """You cannot watch or hear this video: you get {n} still frames sampled across it ({legend}) plus the
narration with its times (in the notes below). Judge what frames can show: does the picture at each narration time
fit what is said, repeated or unrelated shots, not GTA 6, spoilers, nudity / unsafe content, readable captions.
Do not guess about the sound."""


def _ai_review_frames(path: Path, context: str) -> dict | None:
    import base64
    import ai_fallback
    import gemini_writer
    if not ai_fallback.available():
        return None
    dur = probe(path)["duration"] or 0
    n = int(_cfg().get("backup_review_frames", 8))
    pairs = [(t, f) for t, f in ((dur * (0.06 + 0.88 * k / max(1, n - 1)), None) for k in range(n))]
    pairs = [(t, _frame(path, t)) for t, _ in pairs]
    pairs = [(t, f) for t, f in pairs if f]
    if not pairs:
        return None
    legend = ", ".join(f"image {i + 1} = {t:.1f}s" for i, (t, _) in enumerate(pairs))
    prompt = FRAMES_NOTE.format(n=len(pairs), legend=legend) + "\n\n" + AI_PROMPT.format(
        game=gemini_writer._game_context(), context=context[:3000])
    data = ai_fallback.chat_json(prompt, frames=[base64.b64encode(f).decode() for _, f in pairs], temperature=0.2)
    if isinstance(data, dict) and "publish" in data:
        log.info("[qa] AI editor: a backup AI reviewed %d frames + the narration (Gemini unavailable)", len(pairs))
        data["summary"] = f"(backup AI, frames only) {data.get('summary', '')}"
        return data
    return None


LINE_PROMPT = """You check whether the narration of a finished GTA 6 video matches its PICTURE. {facts}
Above, for each numbered narration line, are frames taken WHILE that line is spoken.
For EACH line decide:
- "yes": the frames show what the line describes (the action, place, object or result it talks about)
- "no": the line describes an action / place / object / result that is NOT visible in its frames - e.g.
  "grab the cash from the back alley" while the frames show someone riding a bike down a road, or "open the
  safe" when no safe is on screen. How-to steps and claims about money must be visible to count.
- "general": the line makes no visual claim (a hook like "this is the easiest money in GTA 6", a question to
  the viewer, a joke, an opinion, a number card like "Number three.")
Be strict with "no". Video type: {kind}.
Lines:
{lines}
Return ONLY JSON: {{"lines": [{{"line": number, "verdict": "yes" | "no" | "general", "reason": str}}]}}"""


def _frame(path: Path, t: float, width: int = 640) -> bytes | None:
    import subprocess
    res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                         capture_output=True, timeout=60)
    return res.stdout if res.returncode == 0 and len(res.stdout) > 1500 else None


def line_check(path: Path, lines: list[dict], kind: str) -> list[dict]:
    """Does the picture show what each narration line says? One focused AI question on frames taken while each
    line is spoken (the general AI review scored a Money Tip 9/10 whose "grab the cash from the back alley"
    line played over a bike ride - the owner caught it)."""
    import gemini_writer
    c = _cfg()
    import ai_fallback
    if not lines or not c.get("line_check", True) or not (env("GEMINI_API_KEY") or ai_fallback.available()):
        return []
    dur = probe(path)["duration"] or 0
    per_line = 2 if kind != "long" else 1
    use = lines[: int(c.get("line_check_max_lines", 40))]
    groups, listing, idx = [], [], []
    for i, ln in enumerate(use):
        a, d = float(ln["start"]), float(ln.get("dur", 2.5))
        times = [a + d * 0.5] if per_line == 1 else [a + min(0.4, d * 0.2), a + d * 0.75]
        imgs = [f for f in (_frame(path, min(t, max(0.0, dur - 0.2))) for t in times) if f]
        if not imgs:
            continue
        idx.append(i)
        groups.append((f"Line {len(idx)} frames", imgs))
        listing.append(f'{len(idx)}. "{ln["text"]}"')
    if not listing:
        return []
    import vision  # Gemini first, then the backup AIs that can see images
    data = vision.ask(groups, LINE_PROMPT.format(facts=gemini_writer._game_context(), kind=kind,
                                                 lines="\n".join(listing)), key="lines", num="line",
                      temperature=0.1, prefer_pro=True)
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        return [{"severity": "high" if c.get("require_ai_review", True) else "low", "type": "line_check_unavailable",
                 "detail": "no AI could compare the narration lines with the picture"}]
    issues = []
    for v in data["lines"]:
        try:
            i = int(v.get("line"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= i <= len(idx) and str(v.get("verdict", "")).lower().strip() == "no":
            ln = use[idx[i - 1]]
            issues.append({"severity": "high", "type": "line_not_shown", "at": round(float(ln["start"]), 1),
                           "index": idx[i - 1], "reason": str(v.get("reason", ""))[:200],
                           "detail": f"the picture doesn't show what the narrator says: \"{ln['text'][:70]}\" "
                                     f"({str(v.get('reason', ''))[:120]})"})
    return issues


def check(path: Path, kind: str, lines: list[dict] | None = None, still_ok: list | None = None,
          context: str = "") -> dict:
    """Full QA. kind: short | stream | long. Returns {"pass", "score", "issues", "report"}."""
    c = _cfg()
    if not c.get("enabled", True):
        return {"pass": True, "issues": [], "score": None, "report": "QA disabled"}
    issues = technical(path, still_ok)
    issues += voice_check(path, lines or [])
    try:
        issues += line_check(path, lines or [], kind)
    except Exception as exc:  # noqa: BLE001
        log.warning("[qa] line check failed: %.150s", exc)
        issues.append({"severity": "high", "type": "line_check_unavailable", "detail": f"line check crashed: {exc}"[:200]})
    ai = ai_review(path, context)
    score = None
    if ai is None:
        if c.get("require_ai_review", True):
            issues.append({"severity": "high", "type": "ai_review_unavailable",
                           "detail": "no AI could watch the final video (strict mode: not publishing)"})
    else:
        score = int(ai.get("score") or 0)
        for it in ai.get("issues") or []:
            if isinstance(it, dict):
                issues.append({**it, "type": f"ai:{it.get('type', 'issue')}"})
        if not ai.get("publish", False):
            issues.append({"severity": "high", "type": "ai_verdict", "detail": ai.get("summary", "AI editor said no")})
        if score < int(c.get("min_score", 7)):
            issues.append({"severity": "high", "type": "ai_score", "detail": f"AI editor score {score}/10"})
    high = [i for i in issues if str(i.get("severity")) == "high"]
    ok = not high
    lines_out = [f"QA {'PASSED' if ok else 'FAILED'} ({kind}) - AI score: {score if score is not None else 'n/a'}"]
    lines_out += [f"- [{i.get('severity')}] {i.get('type')}" + (f" at {i['at']}s" if i.get("at") is not None else "")
                  + f": {i.get('detail', '')}" for i in issues]
    tips = [str(t)[:220] for t in ((ai or {}).get("improvements") or []) if str(t).strip()][:8]
    pro = (ai or {}).get("pro_level")
    if tips or pro is not None:  # not blocking - the to-do list for making the edits look like top YouTubers'
        lines_out.append(f"PRO LEVEL: {pro}/10 - to look like a top YouTuber's edit:")
        lines_out += [f"  * {t}" for t in tips]
    report = "\n".join(lines_out)
    log.info("[qa] %s", report.replace("\n", " | ")[:2500])
    try:
        Path(str(path).rsplit(".", 1)[0] + "_QA.txt").write_text(report + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return {"pass": ok, "score": score, "issues": issues, "report": report, "pro_level": pro, "improvements": tips}

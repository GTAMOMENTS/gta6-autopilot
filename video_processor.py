import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
import edge_tts
from gemini_writer import generate_clip_copy

BASE_DIR = Path(__file__).resolve().parent
VO_DIR = BASE_DIR / "02_Voiceovers"
READY_DIR = BASE_DIR / "03_Ready_To_Post"
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return max(10.0, float(res.stdout.strip()))
    except ValueError:
        return 48.0


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def build_fallback_srt(text: str, total_duration: float) -> str:
    words = text.split()
    chunk_size = 6
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    if not chunks:
        return ""
    step = total_duration / len(chunks)
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        start_t = format_srt_time((idx - 1) * step)
        end_t = format_srt_time(min(total_duration, idx * step))
        lines.append(f"{idx}\n{start_t} --> {end_t}\n{chunk}\n")
    return "\n".join(lines)


async def _try_edge_tts(text: str, voice: str, audio_out: Path) -> str:
    communicate = edge_tts.Communicate(text, voice, rate="+12%")
    submaker = edge_tts.SubMaker()
    with open(audio_out, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
    return submaker.get_srt()


def generate_voice_and_srt(text: str, audio_out: Path, srt_out: Path):
    srt_content = ""
    voices_to_try = [
        os.getenv("TTS_VOICE", "").strip(),
        "en-US-AndrewNeural",
        "en-US-GuyNeural",
        "en-US-ChristopherNeural"
    ]
    voices_to_try = [v for v in dict.fromkeys(voices_to_try) if v]

    generated = False
    for voice in voices_to_try:
        try:
            srt_content = asyncio.run(_try_edge_tts(text, voice, audio_out))
            if audio_out.exists() and audio_out.stat().st_size > 5000:
                logging.info("[video_processor] Voiceover generated with %s", voice)
                generated = True
                break
        except Exception as exc:
            logging.warning("[video_processor] Voice %s failed (%s), trying next...", voice, exc)

    if not generated:
        logging.info("[video_processor] Using gTTS backup voice engine...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gTTS"], check=False)
        from gtts import gTTS
        raw_tts = audio_out.with_suffix(".raw.mp3")
        gTTS(text=text, lang="en", slow=False).save(str(raw_tts))
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_tts), "-filter:a", "atempo=1.14", str(audio_out)],
            capture_output=True, check=True
        )
        raw_tts.unlink(missing_ok=True)

    duration = get_audio_duration(audio_out)
    if not srt_content.strip():
        srt_content = build_fallback_srt(text, duration)
    srt_out.write_text(srt_content, encoding="utf-8")
    return duration


def render_vertical_clip(raw_path: Path, clip_id: str, hook_banner: str, voice_script: str) -> Path | None:
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path = VO_DIR / f"{clip_id}.mp3"
    srt_path = VO_DIR / f"{clip_id}.srt"
    out_mp4 = READY_DIR / f"{clip_id}_vertical.mp4"

    audio_dur = generate_voice_and_srt(voice_script, mp3_path, srt_path)
    target_dur = min(58.0, round(audio_dur + 1.2, 2))

    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    safe_banner = re.sub(r"[^a-zA-Z0-9 !?.-]", "", hook_banner)

    vf_filter = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=22:6[bg];"
        "[0:v]scale=1340:-2,crop=1080:ih[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"drawtext=text='{safe_banner}':fontcolor=white:fontsize=54:box=1:boxcolor=black@0.80:boxborderw=20:x=(w-text_w)/2:y=230,"
        f"subtitles='{srt_escaped}':force_style='Alignment=2,MarginV=430,Fontsize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2'[vout];"
        "[0:a]volume=0.22[game_a];"
        "[1:a]volume=1.15[vo_a];"
        "[game_a][vo_a]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(raw_path),
        "-i", str(mp3_path),
        "-filter_complex", vf_filter,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(target_dur),
        str(out_mp4)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_mp4.exists():
        return out_mp4
    logging.error("[video_processor] FFmpeg error: %s", res.stderr[-500:])
    return None


def process_pending_clips(max_to_render: int = 1) -> int:
    if not QUEUE_FILE.exists():
        return 0
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rendered_count = 0

    for item in manifest["items"]:
        if rendered_count >= max_to_render:
            break
        if item.get("status") != "downloaded":
            continue

        banner, script = generate_clip_copy(item.get("title", "GTA 6 Highlight"), item.get("creator", "@RockstarGames"))
        out_path = render_vertical_clip(Path(item["raw_path"]), item["id"], banner, script)
        if out_path:
            item["rendered_path"] = str(out_path)
            item["hook_banner"] = banner
            item["voice_script"] = script
            item["status"] = "ready"
            rendered_count += 1

    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return rendered_count

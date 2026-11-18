import asyncio
import json
import os
import subprocess
from pathlib import Path
import edge_tts
from gemini_writer import generate_clip_copy

BASE_DIR = Path(__file__).resolve().parent
VO_DIR = BASE_DIR / "02_Voiceovers"
READY_DIR = BASE_DIR / "03_Ready_To_Post"
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"


async def generate_tts_with_srt(text: str, voice: str, audio_out: Path, srt_out: Path):
    # Enthusiastic, fast-paced energetic delivery
    communicate = edge_tts.Communicate(text, voice, rate="+14%", pitch="+3Hz")
    submaker = edge_tts.SubMaker()
    with open(audio_out, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
    with open(srt_out, "w", encoding="utf-8") as srt_file:
        srt_file.write(submaker.get_srt())


def render_vertical_clip(raw_path: Path, clip_id: str, hook_banner: str, voice_script: str, voice: str) -> Path | None:
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path = VO_DIR / f"{clip_id}.mp3"
    srt_path = VO_DIR / f"{clip_id}.srt"
    out_mp4 = READY_DIR / f"{clip_id}_vertical.mp4"

    asyncio.run(generate_tts_with_srt(voice_script, voice, mp3_path, srt_path))

    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    safe_banner = hook_banner.replace("'", "").replace(":", "")

    # Loop raw video seamlessly if voiceover is ~50s, crop wider foreground for immersive 9:16 framing, and mix game audio at 22%
    vf_filter = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=22:6[bg];"
        "[0:v]scale=1340:-2,crop=1080:ih[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"drawtext=text='{safe_banner}':fontcolor=white:fontsize=54:box=1:boxcolor=black@0.80:boxborderw=20:x=(w-text_w)/2:y=230,"
        f"subtitles='{srt_escaped}':force_style='Alignment=2,MarginV=430,Fontsize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2'[vout];"
        "[0:a]volume=0.22[game_a];"
        "[1:a]volume=1.15[vo_a];"
        "[game_a][vo_a]amix=inputs=2:duration=shortest:dropout_transition=2[aout]"
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
        "-t", "54",
        "-shortest",
        str(out_mp4)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_mp4.exists():
        return out_mp4
    print(f"[video_processor] FFmpeg error: {res.stderr[-500:]}")
    return None


def process_pending_clips(max_to_render: int = 1) -> int:
    if not QUEUE_FILE.exists():
        return 0
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    voice = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")
    rendered_count = 0

    for item in manifest["items"]:
        if rendered_count >= max_to_render:
            break
        if item.get("status") != "downloaded":
            continue

        banner, script = generate_clip_copy(item.get("title", "GTA 6 Highlight"), item.get("creator", "@RockstarGames"))
        out_path = render_vertical_clip(Path(item["raw_path"]), item["id"], banner, script, voice)
        if out_path:
            item["rendered_path"] = str(out_path)
            item["hook_banner"] = banner
            item["voice_script"] = script
            item["status"] = "ready"
            rendered_count += 1

    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return rendered_count

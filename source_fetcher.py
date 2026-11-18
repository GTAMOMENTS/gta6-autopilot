import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import feedparser

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "01_Raw_Clips"
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"
CONFIG_FILE = BASE_DIR / "config.json"
COOKIES_PATH = Path("/tmp/yt_cookies.txt")

# Official Rockstar Games Akamai CDN MP4 streams (supports HTTP range seeking without login)
OFFICIAL_ROCKSTAR_STREAMS = [
    {
        "prefix": "gtavi_launch",
        "title": "Grand Theft Auto VI Official Launch Trailer Highlight",
        "creator": "@RockstarGames",
        "url": "https://media-rockstargames-com.akamaized.net/VI/downloads/videos/GTAVI_Launch_Trailer/GTAVI_Launch_Trailer.mp4",
        "offsets": [5, 30, 55, 80, 105]
    },
    {
        "prefix": "gtavi_extended",
        "title": "Grand Theft Auto VI Official Extended Gameplay Look Highlight",
        "creator": "@RockstarGames",
        "url": "https://media-rockstargames-com.akamaized.net/VI/downloads/videos/GTAVI_An_Extended_Look/GTAVI_An_Extended_Look.mp4",
        "offsets": list(range(15, 1545, 30))
    }
]


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest():
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not QUEUE_FILE.exists():
        return {"items": []}
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data):
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_safe_title(title: str, blocked: list[str]) -> bool:
    lower = title.lower()
    return not any(word in lower for word in blocked)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60]


def slice_official_stream(stream_url: str, start_sec: int, clip_id: str, duration: int = 24) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{clip_id}.mp4"
    if out_path.exists():
        return out_path

    # HTTP range seek directly to timestamp and downscale to 1080p for fast cloud rendering
    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", stream_url,
        "-vf", "scale=1920:-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50_000:
        return out_path
    print(f"[source_fetcher] Stream slice failed ({clip_id}): {res.stderr[-300:]}")
    return None


def download_youtube_clip(video_url: str, video_id: str, max_duration: int = 28) -> Path | None:
    cookies_txt = os.getenv("YT_COOKIES_TXT", "").strip()
    if not cookies_txt:
        return None
    COOKIES_PATH.write_text(cookies_txt + "\n", encoding="utf-8")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{video_id}.mp4"
    if out_path.exists():
        return out_path

    cmd = [
        "yt-dlp",
        "--cookies", str(COOKIES_PATH),
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*0:00-0:{max_duration + 5}",
        "--no-playlist",
        "-o", str(out_path),
        video_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_path.exists():
        return out_path
    return None


def fetch_new_sources(limit_per_run: int = 1) -> int:
    cfg = load_config()
    manifest = load_manifest()
    known_ids = {item["id"] for item in manifest["items"]}
    blocked = cfg["blocked_keywords"]
    added = 0

    # 1. Optional YouTube RSS if YT_COOKIES_TXT is configured
    for src in cfg["rss_sources"]:
        if added >= limit_per_run:
            break
        feed = feedparser.parse(src["url"])
        for entry in feed.entries[:8]:
            if added >= limit_per_run:
                break
            vid_id = sanitize_filename(entry.get("yt_videoid", entry.get("id", "")))
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not vid_id or vid_id in known_ids:
                continue
            if "gta" not in title.lower() and "grand theft auto" not in title.lower():
                continue
            if not is_safe_title(title, blocked):
                continue

            local_file = download_youtube_clip(link, vid_id, cfg["video_specs"]["max_duration_sec"])
            if local_file:
                manifest["items"].append({
                    "id": vid_id,
                    "title": title,
                    "url": link,
                    "creator": src["creator_tag"],
                    "official": True,
                    "raw_path": str(local_file),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(vid_id)
                added += 1

    # 2. Primary reliable source: Official Rockstar Games CDN streams
    for stream in OFFICIAL_ROCKSTAR_STREAMS:
        if added >= limit_per_run:
            break
        for sec in stream["offsets"]:
            if added >= limit_per_run:
                break
            clip_id = f"{stream['prefix']}_{sec}s"
            if clip_id in known_ids:
                continue

            local_file = slice_official_stream(stream["url"], sec, clip_id)
            if local_file:
                manifest["items"].append({
                    "id": clip_id,
                    "title": f"{stream['title']} (Scene at {sec}s)",
                    "url": stream["url"],
                    "creator": stream["creator"],
                    "official": True,
                    "raw_path": str(local_file),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(clip_id)
                added += 1

    save_manifest(manifest)
    return added

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import feedparser

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "01_Raw_Clips"
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"
CONFIG_FILE = BASE_DIR / "config.json"


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


def download_clip(video_url: str, video_id: str, max_duration: int = 28) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{video_id}.mp4"
    if out_path.exists():
        return out_path

    cmd = [
        "yt-dlp",
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


def fetch_new_sources(limit_per_run: int = 2) -> int:
    cfg = load_config()
    manifest = load_manifest()
    known_ids = {item["id"] for item in manifest["items"]}
    blocked = cfg["blocked_keywords"]
    added = 0

    for src in cfg["rss_sources"]:
        feed = feedparser.parse(src["url"])
        for entry in feed.entries[:10]:
            if added >= limit_per_run:
                break
            vid_id = sanitize_filename(entry.get("yt_videoid", entry.get("id", "")))
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not vid_id or vid_id in known_ids:
                continue
            # Prefer GTA entries from official feed
            if "gta" not in title.lower() and "grand theft auto" not in title.lower():
                continue
            if not is_safe_title(title, blocked):
                continue

            local_file = download_clip(link, vid_id, cfg["video_specs"]["max_duration_sec"])
            if local_file:
                manifest["items"].append({
                    "id": vid_id,
                    "title": title,
                    "url": link,
                    "creator": src["creator_tag"],
                    "official": src.get("official", False),
                    "raw_path": str(local_file),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(vid_id)
                added += 1

    save_manifest(manifest)
    return added

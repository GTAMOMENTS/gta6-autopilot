import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "01_Raw_Clips"
LOCAL_CLIPS_DIR = BASE_DIR / "clips"
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"
CONFIG_FILE = BASE_DIR / "config.json"

# Verified public Dailymotion mirrors of official Rockstar GTA VI trailers & gameplay reveals
VERIFIED_GTA6_SOURCES = [
    {
        "id_prefix": "gtavi_trailer2",
        "title": "Grand Theft Auto VI Official Trailer 2 - Jason & Lucia in Leonida",
        "creator": "@RockstarGames",
        "url": "https://www.dailymotion.com/video/x9j2dp6",
        "segments": [(12, 66), (68, 122), (110, 164)]
    },
    {
        "id_prefix": "gtavi_trailer1",
        "title": "Grand Theft Auto VI Official Trailer 1 - Welcome to Vice City",
        "creator": "@RockstarGames",
        "url": "https://www.dailymotion.com/video/x8q8okj",
        "segments": [(8, 62), (35, 89)]
    }
]

SEARCH_QUERIES = [
    "dmsearch8:Grand Theft Auto VI Official Trailer",
    "dmsearch8:GTA 6 Gameplay Reveal 4K"
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
    extra_blocked = ["podcast", "crigler", "episode", "interview", "emulator", "rom"]
    return not any(word in lower for word in (blocked + extra_blocked))


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60]


def download_and_slice(video_url: str, clip_id: str, start_sec: int, end_sec: int) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    full_tmp = RAW_DIR / f"{clip_id}_full.mp4"
    out_path = RAW_DIR / f"{clip_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 200_000:
        return out_path

    if not full_tmp.exists():
        cmd_dl = [
            "yt-dlp",
            "-f", "best[height<=1080][ext=mp4]/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--max-filesize", "180M",
            "-o", str(full_tmp),
            video_url
        ]
        res = subprocess.run(cmd_dl, capture_output=True, text=True, timeout=180)
        if res.returncode != 0 or not full_tmp.exists():
            print(f"[source_fetcher] Download failed for {video_url}: {res.stderr[-300:]}")
            return None

    duration = max(48, end_sec - start_sec)
    cmd_cut = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(full_tmp),
        "-vf", "scale=1920:-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path)
    ]
    cut_res = subprocess.run(cmd_cut, capture_output=True, text=True, timeout=120)
    if full_tmp.exists():
        full_tmp.unlink(missing_ok=True)

    if cut_res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 200_000:
        return out_path
    return None


def fetch_new_sources(limit_per_run: int = 1) -> int:
    cfg = load_config()
    manifest = load_manifest()
    known_ids = {item["id"] for item in manifest["items"]}
    blocked = cfg["blocked_keywords"]
    added = 0

    # 1. Custom MP4 clips dropped into /clips folder
    if LOCAL_CLIPS_DIR.exists():
        for mp4 in sorted(LOCAL_CLIPS_DIR.glob("*.mp4")):
            if added >= limit_per_run:
                break
            clip_id = sanitize_filename(f"local_{mp4.stem}")
            if clip_id not in known_ids:
                manifest["items"].append({
                    "id": clip_id,
                    "title": f"GTA VI Top Moment - {mp4.stem}",
                    "url": "https://www.rockstargames.com/VI",
                    "creator": "@RockstarGames",
                    "official": True,
                    "raw_path": str(mp4),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(clip_id)
                added += 1

    # 2. Verified GTA VI Official Trailers (54-second high-action segments)
    for src in VERIFIED_GTA6_SOURCES:
        if added >= limit_per_run:
            break
        for start_s, end_s in src["segments"]:
            if added >= limit_per_run:
                break
            clip_id = f"{src['id_prefix']}_{start_s}_{end_s}"
            if clip_id in known_ids:
                continue
            local_file = download_and_slice(src["url"], clip_id, start_s, end_s)
            if local_file:
                manifest["items"].append({
                    "id": clip_id,
                    "title": src["title"],
                    "url": src["url"],
                    "creator": src["creator"],
                    "official": True,
                    "raw_path": str(local_file),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(clip_id)
                added += 1

    # 3. Dynamic Dailymotion GTA 6 gameplay/trailer search once trailer segments are used
    for query in SEARCH_QUERIES:
        if added >= limit_per_run:
            break
        probe = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-single-json", query],
            capture_output=True, text=True, timeout=60
        )
        if probe.returncode != 0:
            continue
        entries = json.loads(probe.stdout).get("entries", [])
        for entry in entries:
            if added >= limit_per_run:
                break
            vid_id = sanitize_filename(f"dm_{entry.get('id', '')}")
            title = entry.get("title", "")
            webpage = entry.get("url") or entry.get("webpage_url", "")
            dur = entry.get("duration") or 0
            if not vid_id or vid_id in known_ids or not webpage:
                continue
            if dur < 55 or dur > 1800 or not is_safe_title(title, blocked):
                continue

            local_file = download_and_slice(webpage, vid_id, 5, 59)
            if local_file:
                manifest["items"].append({
                    "id": vid_id,
                    "title": title,
                    "url": webpage,
                    "creator": "@RockstarGames",
                    "official": False,
                    "raw_path": str(local_file),
                    "rendered_path": None,
                    "status": "downloaded",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                })
                known_ids.add(vid_id)
                added += 1

    save_manifest(manifest)
    return added

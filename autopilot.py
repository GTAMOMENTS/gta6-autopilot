import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from source_fetcher import fetch_new_sources
from video_processor import process_pending_clips
from publisher import publish_video_to_platforms, publish_x_text_slot

BASE_DIR = Path(__file__).resolve().parent
QUEUE_FILE = BASE_DIR / "queue" / "clips_manifest.json"
HISTORY_FILE = BASE_DIR / "queue" / "posted_history.json"
CONFIG_FILE = BASE_DIR / "config.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_next_valid_ready_item(manifest: dict) -> dict | None:
    """Finds a ready item whose rendered MP4 actually exists on disk, cleaning up stale cache entries."""
    changed = False
    chosen = None
    for item in manifest.get("items", []):
        if item.get("status") == "ready":
            rpath = item.get("rendered_path")
            if rpath and Path(rpath).exists():
                if chosen is None:
                    chosen = item
            else:
                item["status"] = "expired"
                changed = True
    if changed:
        save_json(QUEUE_FILE, manifest)
    return chosen


def run_cycle():
    load_dotenv(BASE_DIR / ".env")
    cfg = load_json(CONFIG_FILE, {})
    tz_name = os.getenv("TIMEZONE", "Europe/Istanbul")
    now = datetime.now(ZoneInfo(tz_name))
    is_manual_test = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"

    logging.info("Autopilot run at %s (%s) | manual_test=%s", now.strftime("%Y-%m-%d %H:%M"), tz_name, is_manual_test)

    fetched = fetch_new_sources(limit_per_run=1)
    rendered = process_pending_clips(max_to_render=1)
    logging.info("Pipeline status: fetched=%d, rendered=%d", fetched, rendered)

    manifest = load_json(QUEUE_FILE, {"items": []})
    history = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})

    if is_manual_test:
        ready_item = get_next_valid_ready_item(manifest)
        if not ready_item:
            logging.warning("No valid rendered MP4 found on disk.")
            return
        logging.info("Publishing newly rendered clip %s (%s)...", ready_item["id"], ready_item["rendered_path"])
        ok = publish_video_to_platforms(ready_item, ["youtube", "instagram", "x", "tiktok"])
        if ok:
            ready_item["status"] = "published"
            save_json(QUEUE_FILE, manifest)
            logging.info("Manual test run published clip %s!", ready_item["id"])
        else:
            logging.error("Publishing failed for clip %s — check API responses above.", ready_item["id"])
        return

    today_str = now.strftime("%Y-%m-%d")
    current_minutes = now.hour * 60 + now.minute

    for slot in cfg.get("schedule_trt", []):
        slot_id = f"{today_str}_{slot['time']}_{slot['type']}"
        if slot_id in history["completed_slots"]:
            continue

        hh, mm = map(int, slot["time"].split(":"))
        slot_minutes = hh * 60 + mm
        if 0 <= (current_minutes - slot_minutes) <= 45:
            logging.info("Triggering scheduled slot: %s -> %s", slot_id, slot["platforms"])
            if slot["type"] in ("vertical_clip", "x_video_clip"):
                ready_item = get_next_valid_ready_item(manifest)
                if ready_item and publish_video_to_platforms(ready_item, slot["platforms"]):
                    ready_item["status"] = "published"
                    save_json(QUEUE_FILE, manifest)
                    history["completed_slots"].append(slot_id)
                    save_json(HISTORY_FILE, history)
            else:
                if publish_x_text_slot():
                    history["completed_slots"].append(slot_id)
                    save_json(HISTORY_FILE, history)


if __name__ == "__main__":
    run_cycle()

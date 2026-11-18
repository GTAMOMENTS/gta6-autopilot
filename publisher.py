import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "queue" / "posted_history.json"
CONFIG_FILE = BASE_DIR / "config.json"

UPLOAD_POST_VIDEO_URL = "https://api.upload-post.com/api/upload"
UPLOAD_POST_TEXT_URL = "https://api.upload-post.com/api/upload_text"
ZERNIO_BASE_URL = "https://zernio.com/api/v1"

X_ENGAGEMENT_POSTS = [
    "Which city area in Leonida are you driving to first in GTA 6?",
    "13 years of Los Santos.\n\nNext stop: Vice City.",
    "What is the very first car you are customizing in GTA 6?",
    "Drop your boldest GTA 6 launch-week prediction below."
]


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_caption(platform: str, hook: str, creator: str, hashtags: list[str]) -> str:
    if platform == "x":
        return f"{hook}\n\n{creator}"
    tag_line = " ".join(hashtags)
    return f"{hook}\n\n{creator}\n\n{tag_line}"


def publish_tiktok_via_zernio(video_path: Path, caption: str) -> bool:
    zernio_key = os.getenv("ZERNIO_API_KEY", "").strip()
    if not zernio_key:
        print("[publisher] Missing ZERNIO_API_KEY")
        return False

    headers = {
        "Authorization": f"Bearer {zernio_key}",
        "Content-Type": "application/json"
    }

    acc_resp = requests.get(f"{ZERNIO_BASE_URL}/accounts", headers=headers, timeout=30)
    if acc_resp.status_code != 200:
        print(f"[publisher] Zernio accounts error: {acc_resp.status_code} {acc_resp.text}")
        return False

    acc_data = acc_resp.json()
    accounts = acc_data.get("accounts", acc_data.get("data", acc_data if isinstance(acc_data, list) else []))
    tiktok_acc = next((a for a in accounts if a.get("platform") == "tiktok"), None)
    if not tiktok_acc:
        print("[publisher] No TikTok account linked in Zernio")
        return False
    account_id = tiktok_acc.get("_id") or tiktok_acc.get("id")

    presign_resp = requests.post(
        f"{ZERNIO_BASE_URL}/media/presign",
        headers=headers,
        json={
            "filename": video_path.name,
            "contentType": "video/mp4",
            "size": video_path.stat().st_size
        },
        timeout=30
    )
    if presign_resp.status_code not in (200, 201):
        print(f"[publisher] Zernio presign error: {presign_resp.status_code} {presign_resp.text}")
        return False

    pdata = presign_resp.json().get("data", presign_resp.json())
    upload_url = pdata["uploadUrl"]
    public_url = pdata["publicUrl"]

    with open(video_path, "rb") as vf:
        put_resp = requests.put(upload_url, headers={"Content-Type": "video/mp4"}, data=vf, timeout=120)
        if put_resp.status_code not in (200, 201):
            print(f"[publisher] Zernio upload error: {put_resp.status_code}")
            return False

    post_resp = requests.post(
        f"{ZERNIO_BASE_URL}/posts",
        headers=headers,
        json={
            "content": caption,
            "mediaItems": [{"url": public_url, "type": "video"}],
            "platforms": [{"platform": "tiktok", "accountId": account_id}],
            "publishNow": True
        },
        timeout=60
    )
    if post_resp.status_code not in (200, 201):
        print(f"[publisher] Zernio post error: {post_resp.status_code} {post_resp.text}")
        return False
    return True


def publish_video_to_platforms(item: dict, platforms: list[str]) -> bool:
    api_key = os.getenv("UPLOAD_POST_API_KEY", "").strip()
    profile = os.getenv("UPLOAD_POST_PROFILE", "default").strip()

    cfg = load_json(CONFIG_FILE, {})
    hashtags = random.choice(cfg.get("hashtags_pool", [["#GTA6", "#GTAVI"]]))
    video_path = Path(item["rendered_path"])
    if not video_path.exists():
        return False

    any_ok = False

    for plat in platforms:
        caption = build_caption(plat, item.get("hook_banner", "GTA 6 top moment"), item["creator"], hashtags)

        if plat == "tiktok":
            if publish_tiktok_via_zernio(video_path, caption):
                print("[publisher] Published to TikTok via Zernio")
                any_ok = True
            continue

        if not api_key:
            continue

        headers = {"Authorization": f"Apikey {api_key}"}
        with open(video_path, "rb") as vf:
            files = {"video": (video_path.name, vf, "video/mp4")}
            data = {
                "user": profile,
                "platform[]": plat,
                "title": caption[:95] if plat == "youtube" else caption,
                "description": caption
            }
            resp = requests.post(UPLOAD_POST_VIDEO_URL, headers=headers, data=data, files=files, timeout=120)
            if resp.status_code in (200, 201):
                print(f"[publisher] Published to {plat} via Upload-Post")
                any_ok = True
            else:
                print(f"[publisher] Upload-Post {plat} error: {resp.status_code} {resp.text}")

    return any_ok


def publish_x_text_slot() -> bool:
    api_key = os.getenv("UPLOAD_POST_API_KEY", "").strip()
    profile = os.getenv("UPLOAD_POST_PROFILE", "default").strip()
    if not api_key:
        return False

    history = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})
    recent_texts = {p.get("text", "") for p in history.get("posts", [])[-10:]}
    candidates = [t for t in X_ENGAGEMENT_POSTS if t not in recent_texts] or X_ENGAGEMENT_POSTS
    text = random.choice(candidates)

    headers = {"Authorization": f"Apikey {api_key}"}
    data = {
        "user": profile,
        "platform[]": "x",
        "title": text
    }
    resp = requests.post(UPLOAD_POST_TEXT_URL, headers=headers, data=data, timeout=60)
    if resp.status_code in (200, 201):
        history.setdefault("posts", []).append({
            "type": "x_text",
            "text": text,
            "posted_at": datetime.now(timezone.utc).isoformat()
        })
        save_json(HISTORY_FILE, history)
        return True
    print(f"[publisher] Upload-Post X text error: {resp.status_code} {resp.text}")
    return False

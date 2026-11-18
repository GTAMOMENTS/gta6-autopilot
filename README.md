# GTA VI Top Moments — 24/7 Autopilot

> **Not a programmer? Start with [SETUP_GUIDE.md](SETUP_GUIDE.md)**: click-by-click setup, no code needed.

Fully automated faceless GTA 6 clip channel for YouTube Shorts, Instagram Reels, TikTok and X,
running free on GitHub Actions.

## What each run does

1. **Finds recent footage from anywhere** (last 14 days, min 720p), mixing two types: **gameplay**
   and **stream moments** (viral Twitch clips, recorded streams, YouTube stream highlights with streamer reactions).
   Gameplay sources: gameplay walkthroughs, missions, story cutscenes, police chases,
   heists and stunts. It searches Dailymotion and YouTube with about 20 rotating queries, plus anything
   you drop in `/clips` or paste into `config.json → sources.extra_urls`. Official trailers are
   only a last resort.
2. **Filters out junk**: GTA 5 mods posing as GTA 6, fan "concept" trailers, AI fakes, leaks,
   reaction/news/talk videos, livestreams.
3. **Downloads only a ~2 min section** of long videos (YouTube's "most replayed" peak when available).
   **The AI acts as the editor**: it watches the whole section and picks the most gripping 25–50 seconds
   (starting right before the action), with motion/loudness analysis as hints and as fallback.
4. **The AI judges the clip** (min 7/10 hype; a 6/10 clip is only used if nothing better is found) and (if Gemini is busy, a free OpenRouter/Groq/Mistral model checks frames of the clip instead) and rejects it if it isn't real GTA VI, has a facecam, or is boring.
   It then writes a banner, 3-5 short hype reaction lines timed to the big moments (not a narration), a YouTube title,
   captions and an X post.
5. **Renders a ~48 s 1080×1920 video**: blurred background, zoomed gameplay, wrapped hook banner, big
   word-by-word captions, short hype voice lines (Gemini TTS, edge-tts fallback) over loud game audio, -14 LUFS.
6. **Publishes** through the cheapest working route per platform (see below), and records history
   so nothing is posted twice.

The run fails (red ❌ in Actions, email notification) if nothing could be posted. Each run's
summary page shows the source, AI verdict, script and post links.

## Publishing routes (free first)

| Platform  | Route (in order)                    | Cost |
|-----------|-------------------------------------|------|
| YouTube   | `youtube_api` → zernio → upload_post | **Free** (official API, 100 uploads/day) |
| Instagram | `zernio` → upload_post               | **Free** (Zernio: first 2 accounts free, unlimited posts) |
| TikTok    | `zernio` → upload_post               | **Free** (the 2nd free Zernio account) |
| X         | `x_api` → `x_browser` → zernio → upload_post | `x_browser` is **free**: a hidden browser posts with your login cookies (`X_COOKIES`). It breaks X's automation rules, so there's a risk of lock/suspension. The official API is paid (about $3/month) |

Change the order in `config.json → publishing.routes`. A provider without secrets is skipped automatically.

## GitHub secrets

| Secret | Needed for |
|--------|-----------|
| `GEMINI_API_KEY` | AI check + scripts (free at aistudio.google.com). Watches the whole clip |
| `OPENROUTER_API_KEY` | **Free backup AI** when Gemini is busy (openrouter.ai → Keys). Free vision models look at 6 frames of the clip |
| `GROQ_API_KEY` | Optional 2nd free backup (console.groq.com), text only |
| `NVIDIA_API_KEY` | Optional 3rd free backup (build.nvidia.com, no card) |
| `ZERNIO_API_KEY` | Instagram + TikTok (connect exactly these 2 accounts in Zernio) |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | YouTube. See SETUP_GUIDE.md section C (browser only) or run `tools/youtube_auth.py` |
| `X_COOKIES` | Free X posting via browser (SETUP_GUIDE.md section E) |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | X (optional, paid). From console.x.com, app with Read+Write, OAuth 1.0a |
| `UPLOAD_POST_API_KEY`, `UPLOAD_POST_PROFILE` | Optional fallback (10 free uploads/month) |
| `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | Optional, free: viral Twitch clips + recorded streams (SETUP_GUIDE.md section H) |
| `YTDLP_COOKIES` | Optional. YouTube blocks GitHub's servers ("not a bot" check). Paste a Netscape-format `cookies.txt` exported from a **throwaway** Google account to unlock YouTube as a source. Without it, Dailymotion and the other sources still work |

### YouTube one-time setup
1. https://console.cloud.google.com → new project → *APIs & Services → Library* → enable **YouTube Data API v3**.
2. *OAuth consent screen* → External → add yourself as test user → **Publish app** (status *In production*).
   In *Testing* status the token expires every 7 days.
3. *Credentials → Create OAuth client ID → Desktop app* → download the JSON.
4. On your PC: `pip install requests` then `python tools/youtube_auth.py path/to/client_secret.json`,
   log in with the account that owns the channel, and copy the 3 printed values into GitHub secrets.

## Checking your setup

Run the workflow with `slot = check_setup`. It tests every secret and shows a ✅/❌ table with fixes. It posts nothing.

## Running manually

*Actions → GTA6 Top Moments 24/7 Autopilot → Run workflow*:
- `slot`: what to run (`vertical_clip`, `x_video_clip`, `test_all`, or a text post type).
- `dry_run` (default **on**): finds and renders the video **without posting**. Download it from the
  run's *Artifacts* to check the quality.

Locally: `python autopilot.py --slot vertical_clip --dry-run` (needs ffmpeg, plus a `.env` with the secrets).

## Files

| File | Role |
|------|------|
| `autopilot.py` | entry point: maps each cron run to its slot, runs it, records history |
| `source_fetcher.py` | footage discovery, filtering, ranking, section download |
| `highlight.py` | finds the most action-packed seconds |
| `gemini_writer.py` | AI quality gate, scripts, captions, X posts (with fallbacks) |
| `ai_fallback.py` | free backup AIs (OpenRouter / Groq / Mistral), models auto-discovered |
| `video_processor.py` | TTS, word-timed captions, vertical render |
| `publisher.py` | YouTube API / Zernio / X API / Upload-Post with automatic fallback |
| `config.json` | schedule, routes, search queries, filters, style |

State (posted history + used sources) lives in the Actions cache (`queue/`), not in git.

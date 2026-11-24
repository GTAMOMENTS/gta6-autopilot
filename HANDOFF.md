# HANDOFF - GTA 6 autopilot (read this first)

This file is everything a new AI assistant (or developer) needs to continue the work exactly where it stopped.
Written 2026-11-19 by the previous assistant. Repo: https://github.com/GTAMOMENTS/gta6-autopilot

---------------------------------------------------------------------------------------------------------
## 1. The owner and the goals

- **Goal: the highest possible views** on a faceless GTA 6 channel, **100% free** (no paid APIs, no paid plans), **100% automated**
  (runs on GitHub Actions on a schedule, nobody touches it).
- Channels: YouTube `@GTAVI6TopMoments`, Instagram `@gtavi_topmoments`, TikTok `@gta.vi.top.moment`, X `@GTAVIMomentz`.
  Brand name in config: "GTA VI Top Moments". Owner's timezone: **Turkey (TRT, UTC+3)**. GTA 6 launched 2026-11-19.
- Owner writes short, casual messages (often typos), wants **things done, merged and published**, then a short status.
  He tests by downloading the videos and checking them himself.

### Owner's rules / decisions (do NOT undo these)
| Topic | Decision |
|---|---|
| Voice | **Human voice only**: Gemini TTS voice **"Fenrir"**. Never robotic (edge-tts/gTTS) - `voice.robot_fallback=false`. If the voice fails, post nothing (Shorts) / post without narration (long video). |
| TTS prompt | Gemini TTS gets **ONLY the script text** (style instructions were read out loud once - big bug). Never put instructions in the TTS request. |
| Script | Full narration script written by the AI (8-14 lines for Shorts), captions must match the voice exactly. |
| Spoilers | **No story spoilers ever** (AI `is_spoiler` gate + blocked title words). |
| Brands | Naming cars/brands in scripts/titles is **OK** (`quality_gate.block_brand_names=false`). Brand **hashtags** (#RockstarGames, #PS5...) stay removed (Instagram flagged "branded content" once). |
| Length | **No minimum length** for Shorts (a 1:15 minimum was tried and rejected - it dropped good short clips). |
| Streamers | **Only famous English-speaking streamers** (no Turkish/Spanish/etc.). |
| Schedule | **Owner's plan (2026-11-24), see section 17**: 2 narrated Shorts a day (Video A + Video B) on every platform (YouTube Shorts, Instagram, TikTok, X), 3 X text posts, 1 long YouTube video every 3 days, published automatically after QA. Streamer clips only as the fallback when a Short slot posts nothing. |
| Quality | Must not post junk: script gate (>=4 lines/35 words, written by an AI that saw the clip), hype gate, choppy-clip filter. |
| Merging | **Owner (2026-11-20): "merge automatically when you deem it necessary"** - test (local + Actions dry run where it matters), then merge the PR yourself (REST: `gh api -X PUT repos/GTAMOMENTS/gta6-autopilot/pulls/N/merge -f merge_method=merge`; GraphQL is blocked in the session). Tell the owner what was merged and why. |

---------------------------------------------------------------------------------------------------------
## 2. What runs automatically

### Workflow `.github/workflows/main.yml` ("GTA6 Top Moments 24/7 Autopilot") - one queue for real runs
| TRT | UTC cron | Slot | What |
|---|---|---|---|
| 15:10 | 10 12 | **long_video** (every 3 days) | 16:9 long video: stream-moments countdown (trend guide when streams are quiet). Published automatically after QA (owner 2026-11-24; `long_video.require_approval` true = waits for the owner) |
| 16:50 | 50 13 | x_text_morning | X text: question / poll (uses today's trends) |
| 19:40 | 40 16 | vertical_clip = **Video A** | rotation format (narrated) -> Instagram, X, TikTok, YouTube Shorts |
| 22:10 | 10 19 | x_question_nostalgia | X text: hot take / this-or-that |
| 01:40 | 40 22 | video_b = **Video B** | second video from the normal rotation -> Instagram, X, TikTok, YouTube Shorts |
| 04:20 | 20 1 | x_daily_recap | X text: recap |
Manual runs: Actions -> Run workflow -> `slot` (check_setup, voice_samples, vertical_clip, money_clip, trends, test_all, remake,
stream_clip, long_video, approve_long, text slots), `dry_run`, `format` (Shorts formats, or long kinds moments / money_guide /
trend_guide), `remake_clip`, `providers`. **Dry runs get their own queue** (they never save state) - they no longer block or
cancel scheduled runs. Real runs: 1 running + 1 pending. GitHub cron is often 30-60 min late and sometimes skips runs.

### Workflow `.github/workflows/streams.yml` ("GTA6 Viral Stream Clips")
No longer on a timer (owner's plan 2026-11-20): viral stream clips are posted as Video A (format `stream_reactions` ->
`stream_clips.run_for_slot`). Can still be run by hand.

---------------------------------------------------------------------------------------------------------
## 3. Code map

| File | Role |
|---|---|
| `autopilot.py` | entry point; maps cron -> slot; Short pipeline (`run_video_slot`: sources -> AI copy -> gates -> cut -> voice -> render -> publish), `run_remake`, text slots, history, run summary, learning refresh |
| `source_fetcher.py` | footage discovery (YouTube search, Dailymotion, Twitch, X accounts, local, trailers), ranking, downloads (yt-dlp sections, YouTube client fallbacks, direct mp4 / HLS for X/Kick), blocked news/official channels |
| `gemini_writer.py` | Gemini: watches clips (video+audio upload), writes copy (prompts CLIP_PROMPT, TOUR_PROMPT), model discovery (Flash, Flash-Lite, Pro), retries, `polish_packaging` (critic) |
| `ai_fallback.py` | free backup AIs (frames only, no audio): **Gemma** on the Gemini key, OpenRouter, NVIDIA, Groq, Mistral |
| `critic.py` | second AI rewrites titles/banners/captions/long-video title+thumbnail text for CTR; rejects invented numbers/names |
| `formats.py` | Video A rotation (top5_funny, car_chases, top5_things_to_do, secrets, stunts_crashes, stream_reactions, trending) + **learning** + series numbers; slot formats (Video B = `money_tips`, `slot_only`), money topic rotation, `trending` filled from the trend radar, stream-health switch |
| `trends.py` | **trend radar** (free): YouTube/Google autocomplete, this week's biggest GTA 6 YouTube videos, Reddit, Twitch viewers -> AI topics in `queue/trends.json` (6 h cache); `stream_health()`, `pick()`, `context_line()` for X posts |
| `analytics.py` | reads YouTube views/likes/comments of recent videos with yt-dlp (free) for learning; marks deleted videos |
| `video_processor.py` | Gemini TTS (Fenrir; one request + alignment, per-line fallback, length checks), vertical render (blurred bg, zoom, ASS captions, cold open, zoom punch-in, generated whoosh/boom SFX, optional `music/`), `place_lines` |
| `highlight.py` | energy curve, best window, `cut_segments` |
| `publisher.py` | YouTube API upload (+ first comment, custom thumbnail), Zernio (IG/TikTok), X browser/API, Upload-Post; captions per platform (3-5 hashtags, keywords, series) |
| `stream_clips.py` | viral stream clips: Twitch top live streams + their clips (official API), Kick top English live channels + channel clips (public web API via curl_cffi), YouTube streamers' recordings (most-replayed peak), X clip accounts (IShowSpeedHQ); `is_famous`, `smoothness` (choppy check), AI review, render, post |
| `long_video.py` | long video, kinds `moments` (below) and `money_guide` / `trend_guide` (gather_guide -> review_guide -> package_guide -> render_long mode guide); moments: gather (famous English, max 2/streamer) -> download (+no-sound/choppy filter) -> AI review in batches of 6 (Pro first) -> Whisper smart cuts -> package (title/thumbnail text/intro + critic) -> 16:9 render (cold open, title cards, Fenrir lines, whoosh, outro) -> YouTube upload + thumbnail + chapters/credits |
| `transcribe.py` | **Whisper** (faster-whisper "small", local on runner, model cached via actions/cache `~/.cache/huggingface`): word timestamps, censoring, language detection, `smart_cut` (sentence/scene aligned cuts) |
| `x_source.py` | reads X accounts' media tab with Playwright + X_COOKIES (GraphQL parse, fallback: post links from the page + yt-dlp); cached |
| `x_browser.py` | posts to X through a logged-in headless browser (X_COOKIES) |
| `setup_check.py` | `check_setup` slot: tests every secret, YouTube quality check |
| `config.json` | everything tunable (schedule, formats, sources, quality_gate, voice, video.fx, publishing, learning, stream_clips, long_video, whisper, critic, gemini, ai_fallback) |

State (not in git): Actions cache `queue/` (manifest of sources/posts, history, format rotation, long_videos.json, x cache) and
`queue_streams/` (stream clip state). Dry runs don't save state.

### Secrets (GitHub -> Settings -> Secrets -> Actions)
GEMINI_API_KEY, ZERNIO_API_KEY, YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN, X_COOKIES, YTDLP_COOKIES, TWITCH_CLIENT_ID,
TWITCH_CLIENT_SECRET, OPENROUTER_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY, MISTRAL_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY
(+ optional X_API_*, UPLOAD_POST_*). Text-only AI jobs use `ai_fallback.text_providers` order (Cerebras, Groq first); vision jobs `providers`.
(GitHub Models / GITHUB_MODELS_TOKEN was removed: GitHub retired the service on 2026-07-30.)

---------------------------------------------------------------------------------------------------------
## 4. Limits & lessons learned (important)

- **YouTube API quota**: 10,000 units/day, upload = 1,600 -> ~6 uploads/day. **Plus a channel upload limit**
  (`uploadLimitExceeded`, rolling ~24 h) - hit on 2026-11-19 after many test uploads. Avoid live test uploads; use dry runs.
- **Gemini free tier**: Flash models run out of daily quota quickly (each Short tries several clips); Flash-Lite usually still
  works; 503/504 "high demand" is common -> one retry after 40 s. `gemini-2.5-pro` returns 404 "no longer available to new users";
  newer Pro models may work (discovered automatically). The owner's Google One / Gemini app subscription does NOT apply to the API.
- **Hugging Face / YouTube / GitHub artifact storage are blocked in the assistant's dev sandbox** - test those parts via Actions dry runs.
- Download real files from runs: GitHub artifacts (`video-<run_id>`, 5 days) - the owner can download them; the assistant can't.
- **X**: reading/posting via browser automation can trigger Cloudflare bot checks - open only one page per visit; X can lock accounts.
- **Instagram**: branded-content violation once (brand hashtags + credit lines). **2026-11-20: flagged "This content may not be original to you" - profile not recommended to non-followers** (all posts reuse other creators' footage / stream clips; deleting posts doesn't lift it). Owner's choice: **Instagram paused** (`publishing.paused_platforms`, PR #29) until there is original footage (owner can't record gameplay right now); request the review later with a cleaner profile. The stream-clip timer (streams.yml) was disabled in GitHub the same day. TikTok's 0 views are probably the same originality rule (unoriginal reposts are kept out of For You).
- **TikTok**: all videos had 0 views on 2026-11-19 - unresolved; owner should check post visibility / Account status (maybe private via Zernio).
- Kick: clip pages need Cloudflare impersonation; use the clip's HLS stream (`clip_url`) - works. Kick's category-wide clip list mixes languages (removed).
- Twitch broadcaster clips can be from another game - filtered by `game_id`.
- Choppy video came from source clips (laggy streams), not from our pipeline -> `smoothness()` drops clips with >20% repeated frames.
- Audio out of sync in the long video came from concat-demuxer copy + per-clip length differences -> fixed in PR #26 (see below).

---------------------------------------------------------------------------------------------------------
## 5. Status at handoff (2026-11-19 ~12:40 UTC)

Merged to main: PRs #9-#25 (schedule, formats, voice fixes, script gate, free AI backups, X source, countdowns, brand rules,
charm pack, learning, stream clips, long video, English-only + choppy filter, Gemini Pro/Whisper/smart cuts/critic).

**Open / next steps (in order):**
1. **PR #26 - long video audio sync fix** (dry run 44525111317 PASSED: 7:50, 15 moments, Whisper cuts, critic, thumbnail;
   owner should download the run's artifact and confirm the sound is in sync) (+ excludes `deep-research-pro` from Pro discovery, + this HANDOFF.md):
   https://github.com/GTAMOMENTS/gta6-autopilot/pull/26 -> run a `long_video` **dry run** on the branch, check the log
   (`[long]`, `[cut]`, `[whisper]`, `[critic]`), ask the owner to download the artifact and confirm sound is in sync, then merge.
2. **YouTube `uploadLimitExceeded` (CONFIRMED)**: "The user has exceeded the number of videos they may upload." First seen
   2026-11-19 07:24 UTC after many test uploads, still active at 15:40 UTC (17:40 Short). Shorts still post to IG/TikTok/X;
   the long video is YouTube-only, so it is lost while the limit lasts. Don't do live test uploads. Consider: keep a failed
   long video (Actions cache/artifact) and retry the upload at the next run.
   The first live long video (run 44515332047) rendered (5:46, 11 moments) but was rejected for this reason. `run_long` does not log YouTube's error text -> add logging, and consider keeping a
   failed long video to retry at the next run. Also include the thumbnail `.jpg` in the uploaded artifacts (workflow only uploads mp4/png/mp3).
3. AI review rejected many Twitch clips as "not GTA 6" (17 of ~40) with Flash-Lite - check if those are false rejections
   (maybe ask Flash/Pro when available, or show more context in the prompt).
4. YouTube stream-recording source finds 0 (recent streams have no "most replayed" data yet) - low priority.
5. First-comment permission: owner must re-create YT_REFRESH_TOKEN with scope `youtube.force-ssl` (OAuth Playground steps were given).
6. The "secrets" Short format kept failing: "GTA 6 secrets/details" videos almost always have a YouTuber narrating (gate
   rejects narration). Queries now ask for "no commentary" footage (PR #26) - check it finds usable clips; if not, consider
   dropping "secrets" from the rotation or letting the AI find details in plain free-roam footage.
7. Watch learning data (`[learn] format scores`) after a few days; the rotation starts favouring the best formats automatically.

## 6. How to work on this repo
- Branch per change -> PR -> dry run in Actions (`gh api -X POST repos/GTAMOMENTS/gta6-autopilot/actions/workflows/main.yml/dispatches -f ref=<branch> -f 'inputs[slot]=long_video' -f 'inputs[dry_run]=true'`)
  -> read the job log (`gh api repos/.../actions/jobs/<job_id>/logs`) -> merge when the owner agrees.
- Local tests: mock the network parts (sources, Gemini, TTS) and render with real ffmpeg (ffmpeg 6.1 = same as runners).
- Commit messages end with the session attribution lines the environment asks for; never put model names in commits/PRs.

## 7. Strict QA + pro editing (PR #28)
- `qa.py` runs on EVERY finished video before publishing (Shorts, stream clips, long): technical checks (audio vs video
  length, silence, frozen/choppy, black frames, clipping), Whisper voice check (each narration line heard + in sync with
  its captions), Gemini AI editor watching the final video (montage errors, voice vs on-screen text, spoilers...).
  Any high-severity issue = not published. Report: `<video>_QA.txt` in the run's Artifacts. Config `qa`.
- Long videos: jump cuts (dead air removed), REPLAY (slow-mo) / REACTION (facecam close-up) inserts after the top 3
  moments, hook + COMING UP preview (see section 9), audio polish (noise reduction, compression, limiter), light colour grade +
  sharpening, optional `music/` bed, 3 thumbnails (Test & Compare), facecam cut-out on thumbnails.
- Long videos were not auto-published until 2026-11-24 (`long_video.require_approval`, now false - section 17): with
  it on, they wait in `queue/pending_long/` for a manual upload or the workflow's slot `approve_long`.
- Stream-clip Shorts: facecam-on-top / gameplay-below layout when the AI finds a webcam (`stream_clips.facecam_layout`).
- QA caught on real footage (dry run 1): the narrator said "Number four" while the card showed "#13" (the AI wrote the
  lines before the countdown order existed) -> numbers are now added AFTER ordering (`_strip_rank` / `_ranked`), and
  frozen stretches in source streams (lag) are cut out by `_jump_keep` (freezedetect).
- Run time: a long-video run is ~40-60 min (40 downloads + 7 Gemini review batches + render + QA). Dry run 2 hit the old
  60 min job limit -> downloads run 4 at a time (`long_video.download_workers`), a Gemini model that answers 503/504
  is skipped for 8 min (`gemini.busy_cooldown_sec`), job timeout is 120 min (repo is public: Actions minutes are free).

## 8. Owner's views plan + trend radar (2026-11-20, merged: PRs #28 strict QA, #29 pause Instagram, #30 plan, #31 QA fixes)
- Schedule above (Video A + Video B daily, X texts 16:50 / 22:10 / 04:20, long every 3 days). The first **money guide** long
  video was made as a dry run on 2026-11-20 for the owner to publish by hand; `not_before` = 2026-11-23, then every 3 days.
- **Money Tip (Video B)**: format `money_tips` (`slot_only`): one money method per Short, topics rotate (config `topics`),
  a hot money trend (demand >= 7) goes first; amounts only if visible on screen; no story missions / glitches; series
  "GTA 6 Money Tip #N". Footage = "no commentary" gameplay (narrated footage is rejected by the gate).
- **Trend radar** (`trends.py`): refreshed every 6 h on any run. Feeds: format `trending` (Video A), money topics, long
  `trend_guide`, X text prompts. **Stream health**: Twitch English GTA 6 viewers < 20k or < 35% of the 7-day peak = weak ->
  `stream_reactions` becomes `trending`, long `moments` becomes `trend_guide`. If no viral stream clip passes, Video A
  follows a trend instead.
- **Guide long videos**: footage = YouTube/Dailymotion GTA 6 gameplay videos (one ~110 s section each, glitch/mod/GTA 5
  titles filtered, max 2 per channel); the AI researches each clip (item, facts, 3-5 narration lines, usefulness, has_narrator);
  lines with numbers the clip never showed are dropped; the same item from several creators is merged (counts as
  confirmed); countdown weakest -> best; other creators' voices muted, Fenrir explains; credits in the description.
- Test: `slot=trends` prints today's topics; `slot=long_video format=money_guide dry_run=true`; `slot=money_clip dry_run=true`.

## 9. Quality push with 5 specialist agents (2026-11-20 evening, PRs #33-#41, all merged)
- **QA pro review (#33):** every QA report ends with "PRO LEVEL x/10" + concrete improvements (non-blocking backlog).
- **Music (#34, #35):** `music.py` - CC0 beats from Freesound (`FREESOUND_API_KEY` secret is set; 12 beats cached in
  `queue/music`), ducked under the narration; own files in `music/` take priority.
- **Test suite + CI (#36):** `tests/` (~270 offline tests incl. Short + guide smoke renders), `.github/workflows/tests.yml`
  on every PR + push to main. Run `python -m pytest -q -m "not slow"` locally (~2 min). conftest blocks network,
  redirects queue/ + work/, resets the AI verdict cache per test.
- **Sources & efficiency (#37):** Dailymotion format picking, YouTube slow-download salvage + client switch, parallel
  searches, approximate upload dates (cutoff 2026-11-18), `ai_cache.py` (authenticity per source+window, review
  verdicts, fake-channel memory; `queue/ai_verdicts.json`), Gemini per-minute 429 backoff, one decode per clip.
- **Packaging & SEO (#38):** `critic.py` rules (GTA 6 first, true countdown numbers, no cheat words in money titles,
  keep search keywords), per-platform `build_captions`, `build_long_captions`, prompts use trend top searches.
- **Shorts pro edit (#39):** word-by-word karaoke captions (Whisper per voice piece, else pause-based), safe zones,
  bouncing hook banner, question card + CTA, eased zooms / punch-ins / payoff shake+flash, generated pop/riser/hit,
  music lift on action, harder game ducking, dialogue-aware line placement, smart_cut at speech pauses,
  native-res cuts (fixed a ~10 GB memory spike in multi-piece cuts).
- **Long-video pro edit (#40):** hook of #1 + COMING UP preview, #N slam cards, corner badge, lower-third credits,
  progress bar, swipe transitions, 12 s end screen for YouTube end-screen elements, 3 stronger thumbnails,
  chapters always valid (no chapter < 10 s), `_segment` holds the last frame if a source ends early.
- **QA context (#41):** QA is told these edit features are intentional; the AI-review upload copy must be full length;
  long videos log "timeline: last narration ends X, end screen Y-Z, file W".
- **Night of 2026-11-20 (PRs #43-#44):** Money Tips need concrete on-screen money proof (`required_evidence` with money
  words, `formats.money_tips.evidence_terms`) + how-to script rules (`tutorial`), QA judges usefulness; narration
  pieces keep a safe tail (+0.35 s, last line +0.6 s - Whisper word ends are early); money trend topics with
  glitch/exploit/cheat/hack are skipped; voice_out_of_sync details list the words heard.
- First real Money Tip with all upgrades posted 2026-11-21 ~02:30 TRT (YouTube https://youtube.com/shorts/EsXseWiLRF0,
  X, TikTok; QA 9, pro 8) - real GTA 6 but a weak tip (mostly a chase) -> the proof rules above.
- Open: money guide QA said the end was "cut off" (timeline log showed the outro inside the end screen -> likely
  clipped last syllables, fixed in #44 - verify); Gemini free quota runs out after ~6 test runs + the schedule;
  usage limits: 5 agents in parallel hit the 5-hour limit in ~90 min - run at most 3 at once.

## 10. Money content dropped (owner, 2026-11-21)
- The owner dropped money content for now: Money Tips kept narrating things the picture didn't show (a posted one said
  "grab the cash from the back alley" over a bike ride). Video B (01:40 TRT, slot `video_b`) is a second video from the
  normal rotation; long videos are stream-moment countdowns with a trend guide as fallback (`long_video.rotation`,
  `long_video.fallbacks`).
- The money code stays (format `money_tips`, `money_guide`): add `"format": "money_tips"` to the Video B slot or
  `money_guide` to `long_video.rotation` to bring it back. Manual tests: slot `money_clip`, long format `money_guide`.
- New strict check for ALL videos (PR #46): `qa.line_check` - frames taken while each narration line is spoken, one
  focused AI question per line ("does the picture show what this line says?"); any "no" blocks the video.

## 11. Instagram resumed (owner, 2026-11-21)
- The owner's Instagram Profile status shows Recommendation eligibility OK again (Monetization "Established presence"
  only means the account is too new to earn money). `publishing.paused_platforms` is empty again.
- Raw stream clips (streamer's own voice + subtitles, no narration) skip Instagram (`publishing.no_raw_clips_on`) - they
  were most of what got flagged; narrated Video A / B go to Instagram. If the originality flag returns: put
  "instagram" back into `paused_platforms`.

## 12. Prevent, repair, learn (owner: "it should have learnt not to make mistakes", 2026-11-21)
- **Prevent:** `gemini_writer.ground_lines` writes each narration line from 3 frames of the exact footage it plays over;
  `_prepare_lines` (autopilot) then runs `qa.line_check` on frames + text BEFORE voice + render; failing lines get one
  targeted rewrite (told why); still wrong -> next clip with no voice/render spent.
- **Repair:** if final QA's only high issues are lines not matching the picture, `repair_moment` / `repair_tour` rewrite +
  re-voice just those lines and re-render once (no new download / review / script).
- **Learn:** `lessons.py` logs every QA result (`queue/qa_lessons.json`, real runs only); recurring script mistakes (real
  examples) + lines that passed go into the writer's prompts (`formats.prompt_note`, `ground_lines`); formats that keep
  failing are rested (`lessons.should_rest`); each run logs "[learn] QA history: first-pass rate ...".

## 13. Backup AIs for every image check + learning what gets views (owner, 2026-11-21)
- Owner asked why the other AI keys weren't used when Gemini's free quota ran out: every image / video check was
  Gemini-only. `vision.ask(groups, prompt, key, num)` now tries Gemini first, then the backups that can see images
  (`ai_fallback.chat_json(..., frames=)`) - frames flattened with a legend, split into chunks of <= 8 images, numbers
  mapped back. Used by `authentic._ask` (is it GTA 6?), `qa.line_check`, `gemini_writer.ground_lines` and as
  `long_video._batch_verdicts`' last resort. `qa.ai_review` falls back to `_ai_review_frames` (8 frames + timed
  narration; summary starts "(backup AI, frames only)").
- `analytics.view_examples()`: with >= `learning.view_examples_min` (4) videos that have stats, the best ones (views
  per sqrt(age in days)) become examples for the writer (banner, opening line, title) and the weakest ones a style to
  avoid. Injected in `formats.prompt_note` and `critic.polish`. Switch off: `learning.view_examples: false`.
- 2026-11-21 Video A failed (nothing posted): Gemini Flash out of quota (test runs earlier that day), only Flash-Lite
  left; the one good clip lost its script over one line failing the pre-check twice, then the 25-min clip budget ran out
  after 4 clips (3 were fakes). Fixes: `sources.time_budget_min` 55 (job timeout is 120); `video.precheck_attempts` 3 -
  the last rewrite asks for a plain description of what is on screen.
- Owner 2026-11-21: "if nothing got posted u should push a short clip". A video slot (Video A / B) that posts nothing
  now posts the most viral famous-streamer clip in the same run (`autopilot._fallback_stream_clip` ->
  `stream_clips.run_for_slot`, not on Instagram). Only started while < `fallback.max_elapsed_min` (80) have passed;
  the format rotation still records the narrated format's failure. Off: `fallback.stream_clip_on_fail: false`.

## 14. Whole-video script + script judge (owner, 2026-11-22: "the script is so inconsistent ... 0/10")
- Two posted Shorts (dirt bike, "steal a cop car") had QA 9/10 but scripts that made no sense: `ground_lines`
  rewrote every line alone from 3 still frames -> captions of frames ("Number seven: running across the sunlit road
  toward a pickup truck"); one script was "rescued" from a scene description; QA judged only picture + edit.
- Owner: "the writing shall occur after the AI watched the whole video, then write a good script for the whole video".
  Now (`autopilot._write_checked`): `gemini_writer.write_final_script` - Gemini watches the WHOLE final cut (cold
  open + story, with sound, dialogue gaps given) and writes ONE story script (+ banner / title the video delivers)
  -> `script_judge.judge` (a different AI, text models first: makes sense? promise kept? entertaining? + certain
  checks: numbering, fake "I", same word in every line, closing question; min 7/10) -> `qa.line_check` -> any
  failure goes back to the writer with the reasons for a NEW whole script (`video.script_attempts` 3), else next clip.
  Tours keep one line per shot (`slots`), repairs rewrite the whole script too. `ground_lines` is no longer used.
- Footage where characters talk > `video.max_dialogue_share` (60%) is skipped for narration.
- QA editor now scores the script (`script_score`, high issue below `qa.min_script_score` 7) and the `_QA.txt` report
  lists the narration + the judge's score.
- Dry run 2026-11-22: the judge (cerebras/gpt-oss-120b) rejected all 3 scripts of the first clip (5, 5, 4/10:
  numbered list over one scene, "lacks humor and punchlines"). Added: comedy craft in the writer prompt (set-up ->
  punchline, roast decisions, NPC inner voice, a laugh every two lines), "ONE CONTINUOUS SCENE, NOT A LIST" for
  single moments, and 3 different scripts per attempt (`video.script_candidates`) - the judge picks the best passing.
- Dry run 2: scripts became genuinely funny (roast angle) but the judge scored 4-6: it demanded "Top 5" numbering
  (the writer is told not to number one edit), penalised the required closing question and wanted place names; the
  writer then followed that feedback and got worse. The judge prompt now carries the channel rules (never counted as
  problems) + calibration anchors (the two bad posted scripts = 2-3/10, a roast script = 8/10).
- Dry run 3 made the first video with the new system (judge 7/10, QA 9, script 10/10) - artifact of run 45030379337.
  It still had 8 s and 11 s without narration: `script_judge.dead_air` now rejects gaps > `script_judge.max_silence_sec`
  (6 s, game dialogue counts as talking) and narration under `min_coverage` (60%) of the quiet time; the writer is told
  to fill the video.

## 15. Faster, stricter pipeline (2026-11-24) - what the run logs showed and what changed
Evidence (Actions logs): the last posted Short (https://youtube.com/shorts/TPyb3eDoyiY, run 45082855217) was 27 s with 4
lines, a walking hook, a 22-word last line with the question tacked on and the stuffed title "GTA 6 police chase car
chase at Vice City airport goes crazy" - yet judge 8/10, QA 9/10 (Flash-Lite; gemini-2.5-pro answers 404). Time goes
to downloads and AI outages, not QA (~26 s): run 45228468044 spent ~30 of 53 min downloading slices of videos that
were then rejected (9.5 min on one the check called GTA V 99%); in both 45228468044 and 45448945826 Gemini was
overloaded (503) when it mattered (45448945826 dropped a hype-9 clip that had passed every gate) and the fallback
stream clip never asked Gemini at all (see below) - nothing was posted.
- **Approved script = posted script** (`qa.script_integrity`): `place_lines` silently dropped lines that didn't fit,
  and tours cut lines past `tour_max_sec`. A missing approved line is a high QA issue; it is repairable
  (`autopilot.repairable` / `repair_note`: rewrite shorter, re-voice, re-render once) and `lessons` tells the writer
  when scripts run long. A test scenario was silently losing 2 of 5 lines before this.
- **QA fail-fast** (`qa.fail_fast`): script -> technical -> Whisper -> picture -> AI editor; once something blocks,
  the rest is skipped (same verdicts, less quota). `_QA.txt` lists TIMINGS per check and, for passes, a 60 s HUMAN CHECK.
- **One Gemini upload per file per run** (`gemini_writer._upload_video` / `release_uploads`, keyed by path + size +
  mtime; a re-render is a new file): the posted Short's 5-min reel was encoded + uploaded 3 times (~50 s each).
  Uploads are deleted at exit (atexit) or right after single-use reviews (QA, stream clips, long batches).
- **Tour downloads stop after a fake first slice** (`sources.tour.check_first_slice`, `source_fetcher._first_slice_fake`):
  only a confident "not GTA 6" stops it (same rule that already blacklists the video); unsure -> download on.
- **Fallback stream clip asks Gemini again** (`stream_clips.ai_review` resets `_unavailable_until`, like the other
  whole-video checks): it used to upload the clip and give up without one request after an earlier outage.
- **Search phrases, not stuffing**: `formats.*.keywords` are comma-separated phrases ("GTA 6 police chase, car
  chase"); the writer is told to use ONE; `critic.keyword_phrases` never joins two phrases ("chase car") and a rewrite
  must keep one of them (not all); YouTube tags get one tag per phrase (`critic.keyword_list`).
- **Closing line**: `script_judge.static_problems` rejects a closing question buried in a long line
  (`script_judge.max_closing_words` 14; countdowns exempt - their last slot is "Number one").
- Tests: `python -m pytest -m "not slow"` is the quick loop (~1 min; the two real-time download tests are now `slow`);
  CI still runs everything.
- Not done / open: (GitHub Models: solved in section 16 - the service was retired.) A "hook" criterion in the script judge was considered but held back: it
  changes a carefully calibrated judge and may lower the post rate - try it in a dry run first. Verify all of the
  above with an Actions dry run (`slot=vertical_clip`, `dry_run=true`) and read `[qa] ... TIMINGS` + `[gemini] reusing`.

## 16. Surviving AI outages + parallel slice downloads (2026-11-24, owner: "merge then 1 2 then live test and publish")
- **An approved clip waits for a busy Gemini** (`autopilot.patient`, `video.ai_busy_retries` 2 x `ai_busy_wait_sec`
  60, within `sources.time_budget_min`): when no AI can watch the clip to write / repair its script, the run waits
  and asks again instead of dropping a clip that passed every gate (run 45448945826 dropped a hype-9 clip after one
  busy spell; the logs show Gemini's 503s come and go within minutes).
- **GitHub Models removed**: GitHub retired it on 2026-07-30 (https://github.blog/changelog/2026-07-30-github-models-is-now-retired/);
  its endpoint answers every request with HTTP 200 text/plain "OK". Gone from `ai_fallback.PROVIDERS`, config and
  the workflows (`models: read`, `GITHUB_MODELS_TOKEN`).
- **Backup AIs waste less time**: a model that answers twice without usable JSON (thinking models writing prose,
  ~45 s each) is skipped for the rest of the run (`ai_fallback.MAX_STRIKES`); `ai_fallback.max_total_sec` 240 -> 180
  (answers that came at all came within ~30 s; failing chains used the whole 240 s).
- **Tour slices download together** (`sources.tour.parallel_slices`, `source_fetcher._download_slices`): slice 1
  alone (it gets the GTA 6 check), then the others at the same time (long_video already downloads 4 at once).

## 17. Two Shorts a day everywhere + auto-published long video (owner, 2026-11-24)
- Owner: "push two vids a day on each platform and 1 long every 3 days". Video A's YouTube start date
  (`platform_from` 2026-11-27) is removed: Video A + Video B both go to YouTube Shorts, Instagram, TikTok, X.
- Instagram keeps refusing raw streamer clips (`publishing.no_raw_clips_on`, the "not original" flag of 2026-11-20),
  so the owner chose **narrated videos only**: `stream_reactions` is `slot_only` (out of the rotation; still
  available as `format=stream_reactions` on a manual run, and streamer clips stay the fallback when a Short slot posts
  nothing - that fallback still skips Instagram).
- The long video (every 3 days, counted from the last one that posted) **publishes itself after QA**
  (`long_video.require_approval: false`). Videos already waiting for approval stay untouched (they need
  `approve_long`) and expire after 6 days.
- YouTube API quota: 2 Shorts + a long every 3 days = at most 3 uploads a day (~4,800 of 10,000 units).
- Live test of the new code (run 45481334656, manual Video A, 2026-11-24 23:30 TRT): posted a streamer clip (the
  rotation was on `stream_reactions`) to X + TikTok in ~8 min, QA 9/10 with the new TIMINGS line. The first
  narrated Short on the new code is the next scheduled run.
- Seen in that run, not fixed yet: Gemini Flash's daily quota was used up (429) by evening; `_generate_json` waited
  40 s even though a model on cooldown answered right after (round 1 only tried the 404 Pro model).

## 18. Fewer wasted downloads + only footage where something happens (owner, 2026-11-24)
- A live run downloaded 3 slices of 3 videos and then the AI said "someone already talks over this footage" each
  time. Owner: "the code must find the exciting moment worth clipping - you can't do narration on a walking clip".
- **Speech check** (`sources.speech_precheck`, `source_fetcher._talks_over`, `transcribe.speech_share`): Whisper on
  the first 60 s of tour slice 1 (before the other slices) and of single sections (before any AI watches them);
  over 75 % speech = rejected (`TALKY_REASON`). Free, local. Not for streams / formats with `allow_narrated`.
- **Talky channels** (`ai_cache.talky_*`, `sources.talky_channel_after` 2): uploaders with 2 such rejections (speech
  check, the AI gate's narrator rejection, dialogue-heavy cuts) are skipped before download.
- **Nothing happens** (`sources.action_check`, `highlight.action_events`): sudden loud moments (crash, shot,
  explosion) are counted before any AI watches a section / reel; under 0.5 a minute = rejected (`CALM_REASON`).
  Formats with `calm_ok` (secrets, things to do) are exempt.
- **No calm tour shots** (`autopilot.drop_calm_shots`, `video.tour_calm_energy` 0.25): shots in the calmest stretches
  (walking, roaming, calm driving) are dropped while `tour_min_shots` remain; the tour prompt says the same.
- Title scores: "wtf" -3, "moments #" -2.5 (commentary compilations), "free roam" 2 -> 1; action words decide.
- Thresholds (75 % speech, 0.5 loud moments a minute, 0.25 energy) are first guesses from synthetic tests - read
  `[gate] ✋ rejected ... before any AI watched it` lines in real runs and tune them in config.json.

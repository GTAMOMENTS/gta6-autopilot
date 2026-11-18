# Setup guide (no programming needed)

Everything here is done in your web browser. Each part ends with a **secret** to save in GitHub.

---

## A. How to save a "secret" in GitHub

A secret is like a saved password. **You first get the value from another website** (sections C–G
tell you where), **then** paste it into GitHub. Each secret has 2 boxes:
- **Name**: type the exact label from this guide, for example `X_COOKIES` (capital letters, underscores).
- **Secret**: paste (Ctrl+V) the value you copied.

Example: for free X posting you copy your cookies with Cookie-Editor on x.com (section E), then
Name = `X_COOKIES`, Secret = Ctrl+V → **Add secret**.

1. Open https://github.com/GTAMOMENTS/gta6-autopilot
2. Click **Settings** (top menu of the repo) → in the left menu **Secrets and variables** → **Actions**.
3. Click the green **New repository secret** button.
4. **Name** = the exact name from this guide (for example `YT_REFRESH_TOKEN`), **Secret** = the value. Click **Add secret**.
5. To change a secret later: click the pencil ✏️ next to it.

## B. How to check that everything works

1. In the repo click **Actions** → on the left **GTA6 Top Moments 24/7 Autopilot** → **Run workflow** (grey button on the right).
2. **Use workflow from**: pick the branch `claude/admiring-babbage-m28400` (until it is merged into `main`, after that use `main`).
3. **What to run**: `check_setup` → click the green **Run workflow**.
4. Wait about 1–2 minutes, click the run, and look at the **Setup check** table:
   ✅ works · ❌ broken (the last column tells you what to do) · ⚪ not set up yet.

To **see a test video without posting anything**: same steps, choose `vertical_clip` and leave **Dry run** ticked.
When it finishes, open the run and scroll down to **Artifacts** → download `video-…` (a zip with the mp4).

---

## C. YouTube uploads (free) — about 10 minutes, one time only

You'll create a free Google "app" that lets the bot upload to your channel.

**Part 1 – Google Cloud (create the app)**
1. Go to https://console.cloud.google.com and sign in with the **Google account that owns the YouTube channel**.
   Accept the terms if asked.
2. At the top left click the **project selector** (next to "Google Cloud") → **New project** → name it `gta6-autopilot` → **Create**.
   Make sure the new project is selected at the top.
3. In the search bar at the top type **YouTube Data API v3** → click it → click **Enable**.
4. In the search bar type **Google Auth Platform** (or menu ☰ → APIs & Services → OAuth consent screen) → **Get started**:
   - App name: `GTA6 Autopilot` · User support email: your email → **Next**
   - Audience: **External** → **Next**
   - Contact email: your email → **Next** → tick the agreement → **Create**
5. In the left menu click **Audience** → click **Publish app** → **Confirm**.
   It must say **In production**. (If it stays "Testing", YouTube uploads stop working after 7 days.)
6. In the left menu click **Clients** → **+ Create client**:
   - Application type: **Web application** · Name: `autopilot`
   - Under **Authorized redirect URIs** click **+ Add URI** and paste exactly:
     `https://developers.google.com/oauthplayground`
   - Click **Create**. A window shows **Client ID** and **Client secret** — keep this tab open, you need both.

**Part 2 – OAuth Playground (get the token)**
7. Open https://developers.google.com/oauthplayground in a new tab.
8. Click the **⚙️ gear** at the top right → tick **Use your own OAuth credentials** → paste your **Client ID** and **Client secret** → click the gear again to close.
9. On the left, scroll to the bottom to the box **Input your own scopes** and paste:
   `https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly`
   then click **Authorize APIs**.
10. Choose your Google account. If it asks which channel, choose the **GTA VI Top Moments** channel.
    You'll see "Google hasn't verified this app" → click **Advanced** → **Go to GTA6 Autopilot (unsafe)** → tick all boxes → **Continue**.
    (It's your own app, so this is safe.)
11. You're back in the Playground at **Step 2**. Click **Exchange authorization code for tokens**.
    Copy the **Refresh token** (long text that usually starts with `1//`).

**Part 3 – save 3 secrets** (see section A)
| Name | Value |
|---|---|
| `YT_CLIENT_ID` | the Client ID from step 6 |
| `YT_CLIENT_SECRET` | the Client secret from step 6 |
| `YT_REFRESH_TOKEN` | the Refresh token from step 11 |

Then run `check_setup` — the YouTube row should show ✅ with your channel name.

---

## D. Instagram + TikTok (free with Zernio)

1. Log in at https://zernio.com → connect **only Instagram and TikTok** (the first 2 accounts are free).
2. Create an API key and save it as the secret `ZERNIO_API_KEY` (you already have this one).

---

## E. X / Twitter — free option (browser login)

X no longer has a free API. The free option lets a hidden browser log in **as you** and post like a person.

⚠️ **Risk:** X's rules don't allow automating the website. X might ask you to log in again, lock the
account, or (rarely) suspend it. It's your choice. If you don't add this secret, X posts are simply skipped.

1. Install the free browser extension **Cookie-Editor** (Chrome Web Store / Firefox Add-ons).
2. Open https://x.com and make sure you're logged in as **@GTAVIMomentz**.
3. Click the Cookie-Editor icon (puzzle piece 🧩 → Cookie-Editor) → at the bottom click **Export** → **Export as JSON**
   (it's copied to your clipboard).
4. Save it as the secret `X_COOKIES` (paste everything).
5. **Don't click "Log out" on X in that browser** — logging out kills the cookies. Closing the browser is fine.
6. Run `check_setup` — the X row should say "logged in". If X ever asks for a login again, repeat steps 2–4.

---

## F. YouTube videos as clip sources (optional)

YouTube blocks GitHub's servers unless the bot looks like a logged-in viewer.

1. Use a **spare Google account**, not your main one or the channel owner (it could get flagged).
2. In Chrome: open Extensions → Cookie-Editor → **Details** → turn on **Allow in Incognito**.
3. Open an **Incognito/private window**, go to https://youtube.com and log in with the spare account.
4. Cookie-Editor → **Export** → **Export as JSON**.
5. **Close the incognito window without logging out** (otherwise YouTube cancels the cookies right away).
6. Save it as the secret `YTDLP_COOKIES` (replace the old value).
7. Run `check_setup` — the "YouTube as clip source" row should be ✅. Redo this if it turns ❌ later.

---

## H. Twitch clips + recorded streams (free, optional, ~3 minutes)

Lets the bot grab the **most-viewed GTA 6 Twitch clips** of the last 3 days and wild moments from
recorded streams (streamer reactions). Without it, the bot still finds stream highlights on YouTube.

1. Go to https://dev.twitch.tv/console and log in with any Twitch account (create one if needed; you may
   have to turn on two-factor authentication in Twitch settings first).
2. Click **Register Your Application**:
   - Name: anything unique, e.g. `gta6-autopilot-yourname`
   - OAuth Redirect URLs: `http://localhost`
   - Category: **Other** · Client Type: **Confidential** → **Create**
3. Click **Manage** next to your app → copy the **Client ID** → click **New Secret** → copy the **Client Secret**.
4. Save 2 secrets (section A): `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`.
5. Run `check_setup` — the Twitch row shows how many viral clips it found.

To change the gameplay/stream balance, edit `config.json` → `sources` → `content_mix`
(default 60% gameplay, 40% stream moments).

---

## G. AI keys

| Secret | Where | Needed? |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com → Get API key | yes (main AI) |
| `OPENROUTER_API_KEY` | https://openrouter.ai → Keys → Create key | yes (free backup when Gemini is busy) |
| `GROQ_API_KEY` | https://console.groq.com → API Keys | optional extra backup |
| `NVIDIA_API_KEY` | https://build.nvidia.com → sign in (free NVIDIA account) → open any model → **Get API Key** (starts with `nvapi-`) | optional extra backup, free, no card |

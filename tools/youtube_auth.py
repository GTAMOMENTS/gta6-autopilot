"""One-time helper: get a YouTube refresh token for the YT_REFRESH_TOKEN secret.

Run this ON YOUR OWN COMPUTER (it opens a browser to log in):

    pip install requests
    python tools/youtube_auth.py path/to/client_secret.json

Then copy the three printed values into GitHub -> Settings -> Secrets and variables
-> Actions: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.

Before running: in Google Cloud Console
  1. create a project, enable "YouTube Data API v3"
  2. OAuth consent screen: External, add your Google account as a test user, then press
     "PUBLISH APP" (status "In production"). In "Testing" status Google expires the
     refresh token after 7 days and the autopilot would stop uploading.
  3. Credentials -> Create OAuth client ID -> type "Desktop app" -> download the JSON.
When logging in, pick the Google account / brand channel that owns @GTAVI6TopMoments.
If Google warns "app isn't verified", click Advanced -> Go to (your app).
"""
from __future__ import annotations

import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser

import requests

SCOPES = ("https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly "
          "https://www.googleapis.com/auth/youtube.force-ssl")  # force-ssl: post the first comment


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        raw = json.load(f)
    conf = raw.get("installed") or raw.get("web") or raw
    client_id, client_secret = conf["client_id"], conf["client_secret"]

    result: dict = {}
    state = secrets.token_urlsafe(16)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if qs.get("state", [""])[0] == state and "code" in qs:
                result["code"] = qs["code"][0]
                msg = "Done! You can close this tab and go back to the terminal."
            else:
                msg = "Login failed: " + str(qs.get("error", ["unknown"])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(msg.encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{server.server_port}"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect, "response_type": "code", "scope": SCOPES,
        "access_type": "offline", "prompt": "consent", "state": state,
    })
    def serve():
        while "code" not in result:
            server.handle_request()

    threading.Thread(target=serve, daemon=True).start()
    print("Opening browser for Google login...\nIf it does not open, visit:\n" + url)
    webbrowser.open(url)
    while "code" not in result:
        threading.Event().wait(1)

    tok = requests.post("https://oauth2.googleapis.com/token", data={
        "code": result["code"], "client_id": client_id, "client_secret": client_secret,
        "redirect_uri": redirect, "grant_type": "authorization_code",
    }, timeout=30).json()
    if "refresh_token" not in tok:
        print("No refresh token returned:", tok)
        sys.exit(1)
    ch = requests.get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                      headers={"Authorization": f"Bearer {tok['access_token']}"}, timeout=30).json()
    names = [i["snippet"]["title"] for i in ch.get("items", [])]
    print("\nAuthorised channel(s):", names or "(none found - did you pick the right account?)")
    print("\nAdd these GitHub secrets:\n")
    print("YT_CLIENT_ID     =", client_id)
    print("YT_CLIENT_SECRET =", client_secret)
    print("YT_REFRESH_TOKEN =", tok["refresh_token"])


if __name__ == "__main__":
    main()

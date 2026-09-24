"""Cloudflare Workers AI embeddings, for building the index from this machine."""
import os
import re
import time

import subprocess
import threading

import requests

_refresh_lock = threading.Lock()


def refresh_login():
    """wrangler renews its OAuth token (it lasts about an hour) on any command."""
    with _refresh_lock:
        subprocess.run("npx -y wrangler whoami", shell=True, capture_output=True, timeout=120)

from scripts import env  # noqa: F401  (loads .env)

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "")
WRANGLER_CONFIG = os.path.expandvars(r"%APPDATA%\xdg.config\.wrangler\config\default.toml")


def token():
    if os.environ.get("CF_API_TOKEN"):
        return os.environ["CF_API_TOKEN"]
    with open(WRANGLER_CONFIG, encoding="utf-8") as f:
        return re.search(r'^oauth_token\s*=\s*"([^"]+)"', f.read(), re.M).group(1)


def run(model, body, retries=8):
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/{model}"
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {token()}"}, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(2 + attempt)
            continue
        if r.status_code == 200:
            return r.json()["result"]
        if r.status_code == 401 and not os.environ.get("CF_API_TOKEN"):
            refresh_login()
            continue
        if r.status_code in (400, 403) and attempt >= 1:
            raise RuntimeError(f"{r.status_code} {r.text[:300]}")
        time.sleep(2 + attempt * 2)
    raise RuntimeError(f"{model} kept failing: {r.status_code} {r.text[:300]}")

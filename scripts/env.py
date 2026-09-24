"""Read Source Code/.env (git-ignored) into os.environ for the local scripts.

It holds machine-specific values that must not reach the public repo, such
as the relay URL (GEMINI_BASE) and the Cloudflare account (CF_ACCOUNT_ID).
"""
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

if os.path.exists(_PATH):
    with open(_PATH, encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))

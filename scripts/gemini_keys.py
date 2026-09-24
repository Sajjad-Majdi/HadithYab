"""Find the Gemini keys in the environment that the API actually accepts."""
import os
import re

import requests

from scripts import env  # noqa: F401  (loads .env)

# Google blocks Iran, so local tools go through a relay set in .env as GEMINI_BASE.
BASE = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com").rstrip("/")


def valid_keys():
    names = sorted(k for k in os.environ if re.fullmatch(r"GEMINI_API_KEY(_\d+)?|GOOGLE_API_KEY", k))
    good = []
    for name in names:
        key = os.environ[name]
        for _ in range(3):
            try:
                r = requests.post(
                    f"{BASE}/v1beta/models/gemini-embedding-2:embedContent",
                    headers={"x-goog-api-key": key},
                    json={"content": {"parts": [{"text": "a"}]}, "output_dimensionality": 128},
                    timeout=20,
                )
                break
            except requests.RequestException:
                r = None
        if r is not None and r.status_code == 200 and key not in good:
            good.append(key)
    return good

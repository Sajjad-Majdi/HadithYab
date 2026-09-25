"""Run the app locally: python scripts/dev.py [--test-index]

Google is blocked from Iran, so set GEMINI_BASE in .env to a relay.
Workers AI calls use this machine's wrangler login; under .venv-gpu the
queries are embedded locally instead.
"""
import os
import re
import sys

import uvicorn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts import env  # noqa: F401,E402  (.env: relay URL, Cloudflare account)

if not os.environ.get("GEMINI_API_KEYS"):
    keys = [v for k, v in sorted(os.environ.items()) if re.fullmatch(r"GEMINI_API_KEY_\d+", k)]
    os.environ["GEMINI_API_KEYS"] = ",".join(keys)
if "--test-index" in sys.argv:
    os.environ["HADITHYAB_INDEX"] = os.path.join(ROOT, "data", "test_index")

from scripts.cf_ai import ACCOUNT, token  # noqa: E402

os.environ.setdefault("CF_ACCOUNT_ID", ACCOUNT)
os.environ.setdefault("CF_API_TOKEN", token())

# Run from .venv-gpu (torch) and queries are embedded on this machine with the
# downloaded weights, identical to Workers AI, so its daily cap does not apply.
MODEL = os.path.join(ROOT, "data", "models", "embeddinggemma-300m")
try:
    import torch  # noqa: F401
    if os.path.exists(MODEL):
        os.environ.setdefault("EMBED_LOCAL", MODEL)
except ImportError:
    pass

uvicorn.run("hadithyab.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", 8765)))

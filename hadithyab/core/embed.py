"""Query embeddings. The index and the queries must come from the same model.

The default is EmbeddingGemma on Cloudflare Workers AI: free, no daily cap
worth worrying about, top-10 hit rate 85% on our hadith benchmark (MRR 0.63).
gemini-embedding-2 scored 100% (MRR 0.92) but its free quota is ~1000 texts
a day per Google project, too small for the 38k-hadith corpus; it stays wired
in for a paid key. Research mode makes up much of the gap by searching again.
"""
import itertools
import os
import threading
from functools import lru_cache

import numpy as np
import requests

GEMINI_MODEL = "gemini-embedding-2"
GEMINI_RAW_DIM = 1536
GEMINI_DIM = 768

CF_MODELS = {
    # key: (Workers AI model, dimension, query body, document body)
    "qwen3-embedding-0.6b": ("@cf/qwen/qwen3-embedding-0.6b", 1024,
                             lambda t: {"queries": t, "instruction": "Given a question about Islamic teachings, retrieve the hadith that answers it"},
                             lambda t: {"documents": t}),
    "embeddinggemma-300m": ("@cf/google/embeddinggemma-300m", 768,
                            lambda t: {"text": [f"task: search result | query: {x}" for x in t]},
                            lambda t: {"text": [f"title: none | text: {x}" for x in t]}),
    "bge-m3": ("@cf/baai/bge-m3", 1024, lambda t: {"text": t}, lambda t: {"text": t}),
}

EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma-300m")
MODEL = EMBED_MODEL  # recorded in the index so a mismatch is caught at start-up
DIM = GEMINI_DIM if EMBED_MODEL == GEMINI_MODEL else CF_MODELS[EMBED_MODEL][1]

# From Iran Google needs the relay; on Render it is reachable directly.
GEMINI_BASE = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
CF_ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "")
CF_TOKEN = os.environ.get("CF_API_TOKEN", "")

_session = requests.Session()
_lock = threading.Lock()


def _gemini_keys():
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


_ring = itertools.cycle(_gemini_keys() or [""])


_current = [""]


def next_key():
    with _lock:
        _current[0] = next(_ring)
        return _current[0]


def _last_key():
    return _current[0]


def doc_text(doc):
    """The text of one hadith as it is embedded: translation, then the Arabic."""
    return f"{doc['fa']}\n{doc['ar']}"[:6000]


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def cf_run(model, body, token=None, account=None, timeout=20):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account or CF_ACCOUNT}/ai/run/{model}"
    r = _session.post(url, headers={"Authorization": f"Bearer {token or CF_TOKEN}"}, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Workers AI {r.status_code}: {r.text[:200]}")
    return r.json()["result"]["data"]


def _gemini_query(text):
    """Try each key under both model names: the preview name has its own quota."""
    body = {"content": {"parts": [{"text": f"task: search result | query: {text}"}]},
            "output_dimensionality": GEMINI_RAW_DIM}
    tries = max(2, 2 * len(_gemini_keys()))
    for n in range(min(tries, 12)):
        model = GEMINI_MODEL if n % 2 == 0 else GEMINI_MODEL + "-preview"
        r = _session.post(f"{GEMINI_BASE}/v1beta/models/{model}:embedContent",
                          headers={"x-goog-api-key": next_key() if n % 2 == 0 else _last_key()},
                          json=body, timeout=15)
        if r.status_code == 200:
            return r.json()["embedding"]["values"][:GEMINI_DIM]
        if r.status_code not in (429, 503):
            break
    raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")


@lru_cache(maxsize=4096)
def _embed_cached(text):
    last = None
    for _ in range(2):
        try:
            if EMBED_MODEL == GEMINI_MODEL:
                vec = _gemini_query(text)
            else:
                model, _dim, query_body, _doc_body = CF_MODELS[EMBED_MODEL]
                vec = cf_run(model, query_body([text]))[0]
            return tuple(_unit(vec).tolist())
        except (requests.RequestException, RuntimeError) as e:
            last = e
    raise last


def embed_query(text):
    return np.asarray(_embed_cached(text.strip()), dtype=np.float32)

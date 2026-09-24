"""Embed the whole corpus and write the app's index.

Run from Source Code/:  python scripts/build_index.py [--docs-only]
The model is EMBED_MODEL (see hadithyab/core/embed.py). It resumes: finished
batches stay in data/build/<model>/ and are skipped, so a run cut short by a
quota simply continues the next day.

Output, read by the app at start-up:
  hadithyab/index/vectors.npy   float16, (N, DIM), unit length
  hadithyab/index/docs.json.gz  the cleaned hadiths, same order
  hadithyab/index/meta.json     which model made the vectors
  hadithyab/index/lexical.npz   BM25 postings, with vocab.json
"""
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hadithyab.core import embed, lexical  # noqa: E402
from scripts.corpus import load_corpus  # noqa: E402

OUT = os.path.join(ROOT, "hadithyab", "index")
GEMINI = embed.EMBED_MODEL == embed.GEMINI_MODEL
BATCH = 20 if GEMINI else 32


# ---- Gemini: 1000 embeddings a day per Google project, per model name --------
# The preview name returns identical vectors (cosine 1.000) on its own quota,
# so every key is worth two slots. A slot that answers 429 is spent for today.

class Slots:
    def __init__(self, keys):
        self.live = [(k, m) for m in (embed.GEMINI_MODEL, embed.GEMINI_MODEL + "-preview") for k in keys]
        self.lock = threading.Lock()
        self.turn = 0

    def take(self):
        with self.lock:
            if not self.live:
                return None
            self.turn += 1
            return self.live[self.turn % len(self.live)]

    def drop(self, slot):
        with self.lock:
            if slot in self.live:
                self.live.remove(slot)
                print(f"key spent for today, {len(self.live)} slots left", flush=True)


def gemini_batch(slots, texts):
    from scripts.gemini_keys import BASE
    for attempt in range(60):
        slot = slots.take()
        if slot is None:
            return None
        key, model = slot
        body = {"requests": [{"model": f"models/{model}",
                              "content": {"parts": [{"text": f"title: none | text: {t}"}]},
                              "output_dimensionality": embed.GEMINI_RAW_DIM} for t in texts]}
        try:
            r = requests.post(f"{BASE}/v1beta/models/{model}:batchEmbedContents",
                              headers={"x-goog-api-key": key}, json=body, timeout=180)
        except requests.RequestException:
            time.sleep(3)
            continue
        if r.status_code == 200:
            a = np.array([e["values"] for e in r.json()["embeddings"]], dtype=np.float32)[:, :embed.GEMINI_DIM]
            return a / np.linalg.norm(a, axis=1, keepdims=True)
        if (r.status_code == 429 and "PerDay" in r.text) or "API_KEY" in r.text:
            slots.drop(slot)
        else:
            time.sleep(min(20, 2 + attempt))
    return None


# ---- Workers AI ---------------------------------------------------------------

def cloudflare_batch(texts):
    from scripts.cf_ai import ACCOUNT, token
    model, _dim, _query_body, doc_body = embed.CF_MODELS[embed.EMBED_MODEL]
    for attempt in range(12):
        try:
            a = np.asarray(embed.cf_run(model, doc_body(texts), token=token(), account=ACCOUNT, timeout=180),
                           dtype=np.float32)
            return a / np.linalg.norm(a, axis=1, keepdims=True)
        except Exception:
            time.sleep(min(60, 3 * (attempt + 1)))
    return None


def write_docs(docs):
    """The hadith texts and the keyword index; no API calls, so it is cheap to redo."""
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "docs.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, separators=(",", ":"))
    lexical.build([embed.doc_text(d) for d in docs], OUT)


def main():
    docs = load_corpus()
    if "--docs-only" in sys.argv:
        write_docs(docs)
        print("docs and keyword index written", flush=True)
        return 0
    work = os.path.join(ROOT, "data", "build", embed.EMBED_MODEL)
    os.makedirs(work, exist_ok=True)
    n_batches = (len(docs) + BATCH - 1) // BATCH
    todo = [(n, [embed.doc_text(d) for d in docs[n * BATCH:(n + 1) * BATCH]])
            for n in range(n_batches) if not os.path.exists(os.path.join(work, f"{n:05d}.npy"))]
    print(f"{len(docs)} hadiths with {embed.EMBED_MODEL}; {len(todo)} of {n_batches} batches to go", flush=True)

    if GEMINI:
        from scripts.gemini_keys import valid_keys
        slots = Slots(valid_keys())
        worker = lambda texts: gemini_batch(slots, texts)  # noqa: E731
    else:
        worker = cloudflare_batch

    done = missed = 0
    with ThreadPoolExecutor(6) as pool:
        jobs = {pool.submit(worker, texts): n for n, texts in todo}
        for job in as_completed(jobs):
            vecs = job.result()
            if vecs is None:
                missed += 1
                continue
            np.save(os.path.join(work, f"{jobs[job]:05d}.npy"), vecs.astype(np.float16))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    left = len(todo) - done
    print(f"embedded {done} batches today; {left} left", flush=True)
    if left:
        return 1

    full = np.concatenate([np.load(os.path.join(work, f"{n:05d}.npy")) for n in range(n_batches)])
    assert full.shape == (len(docs), embed.DIM), full.shape
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "vectors.npy"), full.astype(np.float16))
    write_docs(docs)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": embed.EMBED_MODEL, "dim": embed.DIM, "count": len(docs), "embedded": len(docs)}, f)
    print("index written", full.shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

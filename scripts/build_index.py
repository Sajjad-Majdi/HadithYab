"""Embed the whole corpus and write the app's index.

Run from Source Code/:  python scripts/build_index.py [--docs-only | --local | --check-local]
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

_spent = threading.Event()  # set once Workers AI says today's allowance is gone


def cloudflare_batch(texts):
    from scripts.cf_ai import ACCOUNT, refresh_login, token
    model, _dim, _query_body, doc_body = embed.CF_MODELS[embed.EMBED_MODEL]
    for attempt in range(12):
        if _spent.is_set():
            return None
        try:
            a = np.asarray(embed.cf_run(model, doc_body(texts), token=token(), account=ACCOUNT, timeout=180),
                           dtype=np.float32)
            return a / np.linalg.norm(a, axis=1, keepdims=True)
        except Exception as e:
            if "daily free allocation" in str(e):
                if not _spent.is_set():
                    print("Workers AI daily allowance used up; the rest waits for tomorrow", flush=True)
                _spent.set()
                return None
            if "401" in str(e):
                refresh_login()
            time.sleep(min(60, 3 * (attempt + 1)))
    return None


# ---- Local: the same open EmbeddingGemma weights, on this machine's CPU --------
# Workers AI serves google/embeddinggemma-300m; running the same weights here
# (python scripts/build_index.py --local) gives vectors the live site's
# Workers AI queries can be compared with. check_local() measures that first.

_local_model = None
_local_lock = threading.RLock()  # local_batch holds it while local_model() takes it again


def local_model():
    global _local_model
    with _local_lock:
        if _local_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            # Downloaded once (with IDM) into data/models; the HF id is only a fallback.
            path = os.path.join(ROOT, "data", "models", "embeddinggemma-300m")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _local_model = SentenceTransformer(path if os.path.exists(path) else "google/embeddinggemma-300m",
                                               device=device)
            print(f"local model on {device}", flush=True)
        return _local_model


def local_batch(texts):
    _model, _dim, _query_body, doc_body = embed.CF_MODELS[embed.EMBED_MODEL]
    prompts = doc_body(texts)["text"]  # the exact strings Workers AI was given
    with _local_lock:
        a = local_model().encode(prompts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return a.astype(np.float32)


def check_local(sample=64):
    """Cosine between local and Workers AI vectors for texts embedded both ways."""
    docs = load_corpus()[:sample]
    work = os.path.join(ROOT, "data", "build", embed.EMBED_MODEL)
    cf = np.concatenate([np.load(os.path.join(work, f"{n:05d}.npy")) for n in range(sample // BATCH)]).astype(np.float32)
    mine = local_batch([embed.doc_text(d) for d in docs])
    cos = (cf * mine).sum(1) / np.linalg.norm(cf, axis=1) / np.linalg.norm(mine, axis=1)
    print(f"local vs Workers AI cosine: min {cos.min():.4f}, mean {cos.mean():.4f}", flush=True)
    return float(cos.min())


def write_docs(docs):
    """The hadith texts and the keyword index; no API calls, so it is cheap to redo."""
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "docs.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, separators=(",", ":"))
    lexical.build([embed.doc_text(d) for d in docs], OUT)


def collections():
    """Every book in index order. Hadith ids come first and never move, so old
    permalinks (/h/<id>) keep pointing at the same hadith."""
    from scripts.books import fetch, load_nahj, load_quran
    fetch()
    return [("hadith", load_corpus()), ("quran", load_quran()), ("nahj", load_nahj())]


def embed_collection(name, docs, worker):
    """Embed what is not on disk yet; return the vectors, zeros where missing."""
    work = os.path.join(ROOT, "data", "build", embed.EMBED_MODEL + ("" if name == "hadith" else f"-{name}"))
    os.makedirs(work, exist_ok=True)
    n_batches = (len(docs) + BATCH - 1) // BATCH
    todo = [(n, [embed.doc_text(d) for d in docs[n * BATCH:(n + 1) * BATCH]])
            for n in range(n_batches) if not os.path.exists(os.path.join(work, f"{n:05d}.npy"))]
    print(f"{name}: {len(docs)} texts, {len(todo)} of {n_batches} batches to go", flush=True)
    done = 0
    if todo:
        with ThreadPoolExecutor(1 if worker is local_batch else 6) as pool:
            jobs = {pool.submit(worker, texts): n for n, texts in todo}
            for job in as_completed(jobs):
                vecs = job.result()
                if vecs is None:
                    continue
                np.save(os.path.join(work, f"{jobs[job]:05d}.npy"), vecs.astype(np.float16))
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
    full = np.zeros((len(docs), embed.DIM), dtype=np.float16)
    have = 0
    for n in range(n_batches):
        path = os.path.join(work, f"{n:05d}.npy")
        if os.path.exists(path):
            part = np.load(path)
            full[n * BATCH:n * BATCH + len(part)] = part
            have += len(part)
    print(f"  {name}: {have}/{len(docs)} embedded", flush=True)
    return full, have


def main():
    books = collections()
    docs = []
    for _name, part in books:
        for d in part:
            d["id"] = len(docs)
            docs.append(d)
    if "--docs-only" in sys.argv:
        write_docs(docs)
        print("docs and keyword index written", flush=True)
        return 0

    if "--check-local" in sys.argv:
        return 0 if check_local() > 0.99 else 1
    if GEMINI:
        from scripts.gemini_keys import valid_keys
        slots = Slots(valid_keys())
        worker = lambda texts: gemini_batch(slots, texts)  # noqa: E731
    elif "--local" in sys.argv:
        worker = local_batch
    else:
        worker = cloudflare_batch

    # The index is written even when a daily quota cut the run short: missing
    # vectors stay zero (those texts still show up in keyword and BM25 search)
    # and the next run fills them in.
    # Small books first, so a short quota still finishes whole books.
    made = {name: embed_collection(name, part, worker) for name, part in sorted(books, key=lambda b: len(b[1]))}
    vectors, have = zip(*(made[name] for name, _part in books))
    full = np.concatenate(vectors)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "vectors.npy"), full)
    write_docs(docs)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": embed.EMBED_MODEL, "dim": embed.DIM, "count": len(docs), "embedded": sum(have),
                   "books": {name: len(part) for name, part in books}}, f)
    with open(os.path.join(OUT, "NOTICE.txt"), "w", encoding="utf-8") as f:
        f.write(NOTICE)
    print(f"index written: {sum(have)}/{len(docs)} embedded", flush=True)
    return 0 if sum(have) == len(docs) else 1


NOTICE = """Quran text: Tanzil Quran Text (Simple, Version 1.1), Copyright (C) 2007-2026
Tanzil Project, Creative Commons Attribution 3.0, https://tanzil.net
Permission is granted to copy and distribute verbatim copies of this text, but
changing it is not allowed. Persian translation: Naser Makarem Shirazi, via
tanzil.net, for non-commercial use.

Nahj al-Balagha: Arabic text with the Persian translation of Mohammad Dashti,
from https://github.com/WWGTX/NahjulBalaghah.

Hadith: https://github.com/IslamShia/shia-hadith
"""


if __name__ == "__main__":
    sys.exit(main())

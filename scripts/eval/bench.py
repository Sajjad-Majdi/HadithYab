"""Score one embedding model on the hadith benchmark.

Usage: python scripts/eval/bench.py <model-key> [fa|fa_ar]

Every query has one target hadith inside a pool of 3000. We report how
often the target lands at rank 1 and in the top 10, and the MRR@10.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scripts.corpus import load_corpus  # noqa: E402

CACHE = os.path.join(ROOT, "data", "bench")


def st_model(name, query_kw, doc_kw, **load_kw):
    def run(queries, docs):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, device="cpu", **load_kw)
        q = model.encode(queries, batch_size=16, normalize_embeddings=True, **query_kw)
        d = model.encode(docs, batch_size=16, normalize_embeddings=True, show_progress_bar=True, **doc_kw)
        return q, d
    return run


def gemini(dim):
    from scripts.gemini_keys import BASE, valid_keys
    keys = valid_keys()
    url = f"{BASE}/v1beta/models/gemini-embedding-2:batchEmbedContents"

    def embed_batch(args):
        i, texts = args
        for attempt in range(12):
            key = keys[(i + attempt) % len(keys)]
            body = {"requests": [{"model": "models/gemini-embedding-2", "content": {"parts": [{"text": t}]},
                                  "output_dimensionality": dim} for t in texts]}
            try:
                r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=120)
            except requests.RequestException:
                time.sleep(2 + attempt)
                continue
            if r.status_code == 200:
                return [e["values"] for e in r.json()["embeddings"]]
            time.sleep(1 + attempt)
        raise RuntimeError("embedding batch kept failing")

    def embed(texts):
        batches = [(i, texts[s:s + 50]) for i, s in enumerate(range(0, len(texts), 50))]
        with ThreadPoolExecutor(4) as pool:
            out = [v for part in pool.map(embed_batch, batches) for v in part]
        a = np.array(out, dtype=np.float32)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    def run(queries, docs):
        q = embed([f"task: search result | query: {t}" for t in queries])
        d = embed([f"title: none | text: {t}" for t in docs])
        return q, d
    return run


def cloudflare(model, batch, q_body, d_body):
    from scripts.cf_ai import run

    def embed(texts, make):
        chunks = [texts[s:s + batch] for s in range(0, len(texts), batch)]
        with ThreadPoolExecutor(6) as pool:
            out = [v for part in pool.map(lambda c: run(model, make(c))["data"], chunks) for v in part]
        a = np.array(out, dtype=np.float32)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    def go(queries, docs):
        return embed(queries, q_body), embed(docs, d_body)
    return go


MODELS = {
    "cf-qwen3-0.6b": lambda: cloudflare("qwen/qwen3-embedding-0.6b", 32,
                                        lambda t: {"queries": t, "instruction": "Given a question about Islamic teachings, retrieve the hadith that answers it"},
                                        lambda t: {"documents": t}),
    "cf-embeddinggemma": lambda: cloudflare("google/embeddinggemma-300m", 100,
                                            lambda t: {"text": [f"task: search result | query: {x}" for x in t]},
                                            lambda t: {"text": [f"title: none | text: {x}" for x in t]}),
    "cf-bge-m3": lambda: cloudflare("baai/bge-m3", 100, lambda t: {"text": t}, lambda t: {"text": t}),
    "jina-v3": lambda: st_model("jinaai/jina-embeddings-v3",
                                dict(task="retrieval.query", prompt_name="retrieval.query"),
                                dict(task="retrieval.passage", prompt_name="retrieval.passage"),
                                trust_remote_code=True),
    "jina-v5-small": lambda: st_model("jinaai/jina-embeddings-v5-text-small",
                                      dict(task="retrieval", prompt_name="query"),
                                      dict(task="retrieval", prompt_name="document"),
                                      trust_remote_code=True),
    "jina-v5-nano": lambda: st_model("jinaai/jina-embeddings-v5-text-nano",
                                     dict(task="retrieval", prompt_name="query"),
                                     dict(task="retrieval", prompt_name="document"),
                                     trust_remote_code=True),
    "nemotron-1b": lambda: st_model("nvidia/Nemotron-3-Embed-1B-BF16",
                                    dict(prompt_name="query"), dict(prompt_name="document")),
    "embeddinggemma": lambda: st_model("google/embeddinggemma-300m",
                                       dict(prompt_name="query"), dict(prompt_name="document")),
    "gemini-2": lambda: gemini(1536),
}


def doc_text(doc, variant):
    if variant == "fa_ar":
        return f"{doc['fa']}\n{doc['ar']}"[:3000]
    return doc["fa"][:3000]


def score(q, d, targets):
    sims = q @ d.T
    order = np.argsort(-sims, axis=1)[:, :10]
    r1 = r10 = mrr = 0.0
    for row, t in zip(order, targets):
        hits = np.where(row == t)[0]
        if hits.size:
            rank = hits[0] + 1
            r1 += rank == 1
            r10 += 1
            mrr += 1 / rank
    n = len(targets)
    return r1 / n, r10 / n, mrr / n


def main():
    key = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "fa"
    corpus = load_corpus()
    with open(os.path.join(ROOT, "data", "eval_set.json"), encoding="utf-8") as f:
        ev = json.load(f)
    pool = ev["pool"]
    index = {doc_id: i for i, doc_id in enumerate(pool)}
    docs = [doc_text(corpus[i], variant) for i in pool]
    queries = [it["query"] for it in ev["items"]]

    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}.{variant}.npz")
    if os.path.exists(path):
        saved = np.load(path)
        q, d = saved["q"], saved["d"]
        secs = float(saved["secs"])
    else:
        t0 = time.time()
        q, d = MODELS[key]()(queries, docs)
        secs = time.time() - t0
        np.savez(path, q=q, d=d, secs=secs)

    report = {"model": key, "docs": variant, "seconds": round(secs)}
    for lang in ("fa", "ar", "all"):
        rows = [i for i, it in enumerate(ev["items"]) if lang == "all" or it["lang"] == lang]
        targets = [index[ev["items"][i]["target"]] for i in rows]
        r1, r10, mrr = score(np.asarray(q)[rows], np.asarray(d), targets)
        report[lang] = {"R@1": round(r1, 3), "R@10": round(r10, 3), "MRR": round(mrr, 3)}
    print(json.dumps(report))
    with open(os.path.join(CACHE, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()

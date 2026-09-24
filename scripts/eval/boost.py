"""Measure ways to lift EmbeddingGemma's accuracy on the hadith benchmark.

Usage: python scripts/eval/boost.py
Compares, on the same 80 queries and 3000-hadith pool:
  base      the query embedding alone
  bm25      keyword ranking alone
  hybrid    base and bm25 fused by reciprocal rank
  hyde      an LLM writes the hadith it expects; that text is embedded too
  rerank    an LLM reorders the top 20 of the best first-stage ranking
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hadithyab.core.text import fold  # noqa: E402
from scripts.cf_ai import run as cf  # noqa: E402
from scripts.corpus import load_corpus  # noqa: E402
from scripts.gemini_keys import BASE, valid_keys  # noqa: E402

LLM = "gemini-3.5-flash-lite"
CACHE = os.path.join(ROOT, "data", "bench", "boost_cache.json")

STOP = set(fold("""از به با در که را و یا این آن است بود شود می کند کرد های ها برای تا بر هم نیز چه چرا کسی
چیست کیست درباره حدیث احادیث روایت روایات امام پیامبر علیه السلام فرمود قال عن في من على الى ان ما لا""").split())


def tokens(text):
    return [t for t in fold(text).split() if len(t) > 1 and t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.4, b=0.7):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        s = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(tokens(query)):
            idf = self.idf.get(w)
            if not idf:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(*rankings, k=60, weights=None):
    total = Counter()
    for r, w in zip(rankings, weights or [1] * len(rankings)):
        for pos, i in enumerate(r):
            total[i] += w / (k + pos + 1)
    return [i for i, _ in total.most_common()]


def metrics(rankings, targets):
    r1 = r10 = mrr = 0
    for r, t in zip(rankings, targets):
        top = list(r[:10])
        if t in top:
            p = top.index(t) + 1
            r1 += p == 1
            r10 += 1
            mrr += 1 / p
    n = len(targets)
    return {"R@1": round(r1 / n, 3), "R@10": round(r10 / n, 3), "MRR": round(mrr / n, 3)}


_keys = None


def llm(prompt, temperature=0.2):
    global _keys
    _keys = _keys or valid_keys()
    for attempt in range(10):
        key = _keys[attempt % len(_keys)]
        try:
            r = requests.post(f"{BASE}/v1beta/models/{LLM}:generateContent", headers={"x-goog-api-key": key}, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "thinkingConfig": {"thinkingLevel": "minimal"}}},
                timeout=60)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        time.sleep(1 + attempt)
    raise RuntimeError(r.text[:300])


HYDE = """یک کاربر در پایگاه احادیث شیعه این را جستجو کرده: «{q}»
حدیثی را بنویس که به احتمال زیاد دنبالش است: یک جمله کوتاه عربی به سبک روایات، و ترجمه فارسی آن در خط بعد.
فقط همین دو خط را بنویس، بدون نام راوی و منبع."""

RERANK = """پرسش کاربر: «{q}»
این‌ها احادیث نامزدند، هر کدام با شماره:
{cands}
شماره‌ها را از مرتبط‌ترین به کم‌ارتباط‌ترین با پرسش مرتب کن. فقط شماره‌ها را با ویرگول جدا کن، مثل: 4,1,7"""


def main():
    corpus = load_corpus()
    ev = json.load(open(os.path.join(ROOT, "data", "eval_set.json"), encoding="utf-8"))
    pool = ev["pool"]
    index = {d: i for i, d in enumerate(pool)}
    items = ev["items"]
    queries = [it["query"] for it in items]
    targets = [index[it["target"]] for it in items]
    docs = [f"{corpus[i]['fa']}\n{corpus[i]['ar']}" for i in pool]

    saved = np.load(os.path.join(ROOT, "data", "bench", "cf-embeddinggemma.fa_ar.npz"))
    qv, dv = saved["q"], saved["d"]
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

    base = [list(np.argsort(-(dv @ q))[:100]) for q in qv]
    bm = BM25(docs)
    lex = [list(np.argsort(-bm.scores(q))[:100]) for q in queries]
    hybrid = [rrf(b, l, weights=[1, 0.5]) for b, l in zip(base, lex)]
    print("base  ", metrics(base, targets))
    print("bm25  ", metrics(lex, targets))
    print("hybrid", metrics(hybrid, targets))

    # HyDE: embed an imagined answer and fuse it with the query itself.
    hyde_txt = cache.setdefault("hyde", {})
    todo = [q for q in queries if q not in hyde_txt]
    with ThreadPoolExecutor(6) as pool_:
        for q, t in zip(todo, pool_.map(lambda q: llm(HYDE.format(q=q)), todo)):
            hyde_txt[q] = t
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    hv = np.array(cf("google/embeddinggemma-300m",
                     {"text": [f"title: none | text: {hyde_txt[q]}" for q in queries]})["data"], dtype=np.float32)
    hv /= np.linalg.norm(hv, axis=1, keepdims=True)
    hyde = [list(np.argsort(-(dv @ h))[:100]) for h in hv]
    mix = [rrf(b, h) for b, h in zip(base, hyde)]
    mix_lex = [rrf(b, h, l, weights=[1, 1, 0.5]) for b, h, l in zip(base, hyde, lex)]
    print("hyde  ", metrics(hyde, targets))
    print("base+hyde", metrics(mix, targets))
    print("base+hyde+bm25", metrics(mix_lex, targets))

    # LLM rerank of a first stage's top TOP (env FIRST=base|mix, TOP=20).
    first = base if os.environ.get("FIRST", "mix") == "base" else mix_lex
    top = int(os.environ.get("TOP", 20))
    rr = cache.setdefault("rerank", {})

    def rerank(n):
        q, cand = queries[n], first[n][:top]
        key = q + "|" + ",".join(map(str, cand))
        if key not in rr:
            listing = "\n".join(f"{j + 1}. {docs[i][:300]}" for j, i in enumerate(cand))
            rr[key] = llm(RERANK.format(q=q, cands=listing), temperature=0)
        order = [int(x) - 1 for x in re.findall(r"\d+", rr[key]) if 0 < int(x) <= len(cand)]
        seen = list(dict.fromkeys(order)) + [j for j in range(len(cand)) if j not in order]
        return [cand[j] for j in seen] + first[n][top:]

    with ThreadPoolExecutor(6) as pool_:
        reranked = list(pool_.map(rerank, range(len(queries))))
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"+rerank {os.environ.get('FIRST', 'mix')} top{top}", metrics(reranked, targets))
    print("first-stage recall@top", sum(t in f[:top] for f, t in zip(first, targets)) / len(targets))
    fa_rows = [i for i, it in enumerate(items) if it["lang"] == "fa"]
    print("+rerank fa only", metrics([reranked[i] for i in fa_rows], [targets[i] for i in fa_rows]))


if __name__ == "__main__":
    main()

"""BM25 over folded Persian and Arabic words, stored as compact arrays.

The postings are built once by scripts/build_index.py and loaded read-only,
so the live app never holds the per-document word counts in Python objects.
"""
import json
import os
from collections import Counter

import numpy as np

from .text import fold

K1, B = 1.4, 0.7

STOP = set(fold("""از به با در که را و یا این آن است بود شود می کند کرد های ها برای تا بر هم نیز چه چرا کسی
چیست کیست درباره حدیث احادیث روایت روایات امام پیامبر علیه السلام فرمود قال عن في من على الى ان ما لا""").split())


def tokens(text):
    return [t for t in fold(text).split() if len(t) > 1 and t not in STOP]


def build(texts, directory):
    counts = [Counter(tokens(t)) for t in texts]
    vocab = sorted({w for c in counts for w in c})
    where = {w: i for i, w in enumerate(vocab)}
    postings = [[] for _ in vocab]
    for doc, c in enumerate(counts):
        for w, f in c.items():
            postings[where[w]].append((doc, f))
    indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(p) for p in postings])
    docs = np.fromiter((d for p in postings for d, _ in p), dtype=np.int32, count=int(indptr[-1]))
    tf = np.fromiter((f for p in postings for _, f in p), dtype=np.float32, count=int(indptr[-1]))
    lengths = np.array([sum(c.values()) for c in counts], dtype=np.float32)
    np.savez_compressed(os.path.join(directory, "lexical.npz"), indptr=indptr, docs=docs, tf=tf, lengths=lengths)
    with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))


class BM25:
    def __init__(self, directory):
        z = np.load(os.path.join(directory, "lexical.npz"))
        self.indptr, self.docs, self.tf, lengths = z["indptr"], z["docs"], z["tf"], z["lengths"]
        with open(os.path.join(directory, "vocab.json"), encoding="utf-8") as f:
            self.where = {w: i for i, w in enumerate(json.load(f))}
        self.n = len(lengths)
        self.norm = (K1 * (1 - B + B * lengths / lengths.mean())).astype(np.float32)

    def scores(self, query):
        out = np.zeros(self.n, dtype=np.float32)
        for w in set(tokens(query)):
            i = self.where.get(w)
            if i is None:
                continue
            a, b = self.indptr[i], self.indptr[i + 1]
            docs, tf = self.docs[a:b], self.tf[a:b]
            idf = np.log1p((self.n - (b - a) + 0.5) / ((b - a) + 0.5))
            out[docs] += idf * tf * (K1 + 1) / (tf + self.norm[docs])
        return out

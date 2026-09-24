"""The in-memory hadith index: semantic search, keyword search, similar hadiths.

Everything lives in RAM. The whole corpus is ~38k hadiths, so one matrix
product answers a query in a few milliseconds, and nothing can go to sleep
the way the old hosted database did.
"""
import bisect
import gzip
import json
import os
import threading

import numpy as np

from .embed import EMBED_MODEL, embed_query
from .lexical import BM25
from .text import BOOKS, SPEAKER_LABELS, fold

INDEX_DIR = os.environ.get("HADITHYAB_INDEX") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index")
DUPLICATE = 0.965  # cosine above which two hits are the same hadith, retold


class Index:
    def __init__(self, directory=INDEX_DIR):
        with gzip.open(os.path.join(directory, "docs.json.gz"), "rt", encoding="utf-8") as f:
            self.docs = json.load(f)
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta["model"] != EMBED_MODEL:
            raise RuntimeError(f"index was built with {meta['model']}, but EMBED_MODEL is {EMBED_MODEL}")
        self.vectors = np.load(os.path.join(directory, "vectors.npy")).astype(np.float32)
        self.bm25 = BM25(directory)
        self.speakers = np.array([d["speaker"] for d in self.docs])
        self.kinds = np.array([d.get("kind", "hadith") for d in self.docs])
        # Bare text for keyword search, Persian and Arabic side by side.
        self.folded = [f" {fold(d['fa'])} | {fold(d['ar'])} " for d in self.docs]
        self.joined = "\n".join(self.folded)
        self.starts, at = [], 0
        for text in self.folded:
            self.starts.append(at)
            at += len(text) + 1
        self.counts = {k: int((self.speakers == k).sum()) for k in SPEAKER_LABELS}
        self.book_counts = {k: int((self.kinds == k).sum()) for k in BOOKS}

    def __len__(self):
        return len(self.docs)

    # ---- ranking -----------------------------------------------------------

    def _mask(self, scope):
        """scope is a speaker key, a book key (quran, nahj, hadith) or all."""
        if not scope or scope == "all":
            return None
        if scope in BOOKS:
            return self.kinds == scope
        return self.speakers == scope

    def _rank(self, scores, speaker, limit, exclude=None):
        mask = self._mask(speaker)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        if exclude is not None:
            scores[exclude] = -np.inf
        take = min(len(scores), limit * 3 + 10)
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [int(i) for i in top if np.isfinite(scores[i])], scores

    def _group(self, ids, scores, limit):
        """Fold retellings of the same hadith into the first hit that has it."""
        picked, variants = [], {}
        for i in ids:
            if len(picked) >= limit:
                break
            if picked:
                sims = self.vectors[picked] @ self.vectors[i]
                j = int(np.argmax(sims))
                if sims[j] >= DUPLICATE:
                    variants.setdefault(picked[j], []).append(i)
                    continue
            picked.append(i)
        return [self.card(i, float(scores[i]), variants.get(i, [])) for i in picked]

    # ---- public API ----------------------------------------------------------

    def semantic(self, query, speaker=None, limit=20):
        qv = embed_query(query)
        ids, scores = self._rank(self.vectors @ qv, speaker, limit)
        return self._group(ids, scores, limit)

    def lexical(self, query, speaker=None, limit=20):
        """BM25 ranking: rare words shared with the query weigh most."""
        scores = self.bm25.scores(query)
        if not scores.any():
            return []
        scores = np.where(scores > 0, scores, -np.inf)
        ids, scores = self._rank(scores, speaker, limit)
        return self._group(ids, scores, limit)

    def similar(self, doc_id, speaker=None, limit=10):
        vec = self.vectors[doc_id]
        if not vec.any():  # not embedded yet: fall back to its words
            d = self.docs[doc_id]
            return [c for c in self.lexical(f"{d['fa']} {d['ar']}"[:600], speaker, limit + 1) if c["id"] != doc_id][:limit]
        ids, scores = self._rank(self.vectors @ vec, speaker, limit, exclude=doc_id)
        return self._group(ids, scores, limit)

    def keyword(self, terms, speaker=None, limit=20):
        """Hadiths whose Persian or Arabic text holds every word of `terms`.

        Words match as whole words; a phrase in quotes must appear as written.
        """
        words, phrases = [], []
        for n, part in enumerate(terms.split('"')):
            if n % 2:
                if fold(part):
                    phrases.append(f" {fold(part)} ")
            else:
                words += [f" {w} " for w in fold(part).split() if w]
        needles = phrases + words
        if not needles:
            return [], 0
        # Scan the joined corpus for the longest needle at C speed, then check
        # the other needles only in the hadiths it hit.
        needles.sort(key=len, reverse=True)
        first, pos, cand = needles[0], 0, set()
        while (pos := self.joined.find(first, pos)) != -1:
            cand.add(bisect.bisect_right(self.starts, pos) - 1)
            pos += 1
        mask = self._mask(speaker)
        hits = [i for i in sorted(cand)
                if (mask is None or mask[i]) and all(n in self.folded[i] for n in needles[1:])]
        # Shorter texts first: the match is the heart of them, not a passing word.
        hits.sort(key=lambda i: len(self.folded[i]))
        return [self.card(i, None, []) for i in hits[:limit]], len(hits)

    def card(self, i, score=None, variants=()):
        d = self.docs[i]
        return {
            "id": i,
            "fa": d["fa"],
            "ar": d["ar"],
            "source": d["source"],
            "from": d["from"],
            "speaker": d["speaker"],
            "speaker_label": SPEAKER_LABELS.get(d["speaker"], ""),
            "kind": d.get("kind", "hadith"),
            "score": None if score is None else round(score, 4),
            "variants": [int(v) for v in variants],
        }


_index = None
_lock = threading.Lock()


def get_index():
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = Index()
    return _index

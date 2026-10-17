"""Crawlable pages: the Quran by surah, Nahj al-Balagha by item, hadiths by speaker.

Search engines and AI readers do not type queries, so every text also needs a
plain link path from the home page. Built once from the loaded index.
"""
import re
from functools import lru_cache

from .core.index import get_index
from .core.text import BOOKS, SPEAKERS, SPEAKER_LABELS

PAGE_SIZE = 40
# section: (one, plural, plural before a noun: «خطبه‌های نهج‌البلاغه»)
NAHJ_SECTIONS = {"sermon": ("خطبه", "خطبه‌ها", "خطبه‌های"), "letter": ("نامه", "نامه‌ها", "نامه‌های"),
                 "saying": ("حکمت", "حکمت‌ها", "حکمت‌های")}
SPEAKER_KEYS = [k for k, _, _ in SPEAKERS] + ["other"]
SCOPES = set(SPEAKER_KEYS) | set(BOOKS)  # the ?s= filter values
_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(n):
    return str(n).translate(_DIGITS)


@lru_cache(maxsize=1)
def tree():
    docs = get_index().docs
    surahs, nahj, speakers = {}, {k: {} for k in NAHJ_SECTIONS}, {k: [] for k in SPEAKER_KEYS}
    for i, d in enumerate(docs):
        kind = d.get("kind", "hadith")
        if kind == "quran":
            s = surahs.setdefault(d["surah"], {"n": d["surah"], "ids": [],
                                               "name": re.sub(r"^سوره\s*|،.*$", "", d["source"]).strip()})
            s["ids"].append(i)
        elif kind == "nahj":
            item = nahj[d["section"]].setdefault(d["number"], {"n": d["number"], "ids": [], "title": d.get("title", "")})
            item["ids"].append(i)
        else:
            speakers.setdefault(d["speaker"], []).append(i)
    return {"surahs": [surahs[k] for k in sorted(surahs)],
            "nahj": {sec: [items[k] for k in sorted(items)] for sec, items in nahj.items()},
            "speakers": speakers}


def surah(n):
    s = tree()["surahs"]
    return s[n - 1] if 1 <= n <= len(s) else None


def nahj_item(section, n):
    for item in tree()["nahj"].get(section, []):
        if item["n"] == n:
            return item
    return None


def speaker_pages(key):
    ids = tree()["speakers"].get(key, [])
    return max(1, -(-len(ids) // PAGE_SIZE))


def speaker_slice(key, page):
    ids = tree()["speakers"].get(key, [])
    return ids[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]


def speaker_label(key):
    return SPEAKER_LABELS.get(key, "")


def gist(card, size=70):
    """The saying itself, without the 'X said:' lead the translation opens with."""
    text = card["fa"]
    head, sep, rest = text.partition(":")
    if sep and len(head) < 90 and rest.strip():
        text = rest
    text = " ".join(text.replace("«", "").replace("»", "").split())
    return text if len(text) <= size else text[:size].rsplit(" ", 1)[0] + "…"


def heading(card):
    """A short human title for one text: «سوره بقره، آیه ۲۵۵», «حدیث امام صادق»…"""
    if card["kind"] in ("quran", "nahj"):
        return card["source"]
    return f"حدیث {card['speaker_label']}" if card["speaker_label"] and card["speaker"] != "other" else "حدیث"

"""Build a small retrieval benchmark from the real corpus.

Picks a pool of hadiths, then asks an LLM to write the kind of question a
user would type to find some of them. The question must paraphrase the
meaning, not copy its words, so the test measures semantic search.
"""
import json
import os
import random
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scripts.corpus import load_corpus  # noqa: E402
from scripts.gemini_keys import BASE, valid_keys  # noqa: E402

POOL = 3000
TARGETS_FA = 90
TARGETS_AR = 30
LLMS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]
OUT = os.path.join(ROOT, "data", "eval_set.json")

PROMPT_FA = """این یک حدیث است:
{text}

یک جستجوی کوتاه (۳ تا ۱۲ کلمه) به فارسی بنویس که یک کاربر عادی برای پیدا کردن همین حدیث تایپ می‌کند.
شرط‌ها: معنا را بازگو کن، واژه‌های خاص و کم‌کاربرد متن را تکرار نکن، نام راوی و منبع را نیاور.
مثل آدم واقعی بنویس: گاهی سؤال، گاهی عبارت. فقط خود جستجو را بنویس."""

PROMPT_AR = """هذا حديث:
{text}

اكتب عبارة بحث قصيرة بالعربية (٣ إلى ١٢ كلمة) يكتبها مستخدم عادي ليجد هذا الحديث.
أعد صياغة المعنى، ولا تنسخ كلمات النص، ولا تذكر الراوي أو المصدر. اكتب عبارة البحث فقط."""


def ask(prompt, key_ring):
    for attempt in range(16):
        key = key_ring[attempt % len(key_ring)]
        llm = LLMS[(attempt // 2) % len(LLMS)]
        try:
            r = requests.post(
                f"{BASE}/v1beta/models/{llm}:generateContent",
                headers={"x-goog-api-key": key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=90,
            )
        except requests.RequestException as e:
            print("net", e, flush=True)
            time.sleep(3)
            continue
        if r.status_code == 200:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip().strip('"«»')
        time.sleep(2)
    raise RuntimeError("LLM kept failing")


def main():
    corpus = load_corpus()
    rng = random.Random(7)
    usable = [d for d in corpus if 40 <= len(d["fa"]) <= 1200]
    pool = rng.sample(usable, POOL)
    ring = valid_keys()
    rng.shuffle(ring)

    items = []
    for i, doc in enumerate(pool[:TARGETS_FA]):
        items.append({"lang": "fa", "target": doc["id"], "query": ask(PROMPT_FA.format(text=doc["fa"]), ring[i:] + ring[:i])})
        print(len(items), items[-1]["query"].encode("utf-8", "replace").decode("utf-8"), flush=True)
    for i, doc in enumerate(pool[TARGETS_FA:TARGETS_FA + TARGETS_AR]):
        items.append({"lang": "ar", "target": doc["id"], "query": ask(PROMPT_AR.format(text=doc["ar"]), ring[i:] + ring[:i])})
        print(len(items), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"pool": [d["id"] for d in pool], "items": items}, f, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()

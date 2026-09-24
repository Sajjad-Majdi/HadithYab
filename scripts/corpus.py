"""Load and clean the IslamShia hadith dump.

The dump repeats many hadiths, reuses the same `id` across its parts, mixes
Arabic and Persian letter forms, and stretches words with tatweel. This
module turns it into one clean list with stable ids.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "hadiths.json")

from hadithyab.core.text import clean, speaker_of  # noqa: E402


def load_corpus(path=RAW):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    seen = set()
    docs = []
    for item in raw:
        fa = clean(item.get("farsiTranslation"), persian=True)
        ar = clean(item.get("hadithText"))
        if not fa and not ar:
            continue
        key = (fa, ar)
        if key in seen:
            continue
        seen.add(key)
        raw_from = clean(item.get("from"), persian=True)
        docs.append({
            "id": len(docs),
            "fa": fa,
            "ar": ar,
            "source": clean(item.get("source")),
            "from": raw_from,
            # Many records leave `from` empty or generic ("عنه عليه السلام"), but
            # the text itself opens with the speaker's name.
            "speaker": next((k for k in (speaker_of(raw_from), speaker_of(fa[:70]), speaker_of(ar[:70]))
                             if k != "other"), "other"),
        })
    return docs


if __name__ == "__main__":
    from collections import Counter
    docs = load_corpus()
    print(len(docs))
    print(Counter(d["speaker"] for d in docs).most_common())

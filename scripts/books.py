"""The Quran and Nahj al-Balagha as searchable documents.

Sources (fetched into data/sources/ by `fetch()`):
  Quran text    Tanzil Project, simple text with diacritics, CC BY 3.0.
                The text must stay verbatim, so it is stored as downloaded;
                only the search index folds it.
  Translation   Naser Makarem Shirazi, via tanzil.net (non-commercial use).
  Nahj          WWGTX/NahjulBalaghah nahj.db: 241 sermons, 79 letters,
                480 sayings, Arabic with Mohammad Dashti's Persian.
"""
import json
import os
import re
import sqlite3
import urllib.request

from hadithyab.core.text import clean

SOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sources")
URLS = {
    "quran-simple.txt": "https://tanzil.net/pub/download/index.php?quranType=simple&outType=txt-2"
                        "&marks=true&sajdah=true&tatweel=true&agree=true",
    "fa.makarem.txt": "https://tanzil.net/trans/fa.makarem",
    "nahj.db": "https://raw.githubusercontent.com/WWGTX/NahjulBalaghah/main/nahj.db",
}

SURAHS = """فاتحه بقره آل‌عمران نساء مائده انعام اعراف انفال توبه یونس هود یوسف رعد ابراهیم حجر نحل اسراء کهف مریم طه
انبیاء حج مؤمنون نور فرقان شعراء نمل قصص عنکبوت روم لقمان سجده احزاب سبأ فاطر یس صافات ص زمر غافر فصلت شوری زخرف
دخان جاثیه احقاف محمد فتح حجرات ق ذاریات طور نجم قمر الرحمن واقعه حدید مجادله حشر ممتحنه صف جمعه منافقون تغابن طلاق
تحریم ملک قلم حاقه معارج نوح جن مزمل مدثر قیامت انسان مرسلات نبأ نازعات عبس تکویر انفطار مطففین انشقاق بروج طارق اعلی
غاشیه فجر بلد شمس لیل ضحی شرح تین علق قدر بینه زلزال عادیات قارعه تکاثر عصر همزه فیل قریش ماعون کوثر کافرون نصر مسد
اخلاص فلق ناس""".split()
assert len(SURAHS) == 114

NAHJ_CHUNK = 600  # Persian characters per Nahj chunk
NAHJ_SECTION = {"sermon": "خطبه", "letter": "نامه", "saying": "حکمت"}
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fetch():
    os.makedirs(SOURCES, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(SOURCES, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


def _verses(name):
    out = {}
    with open(os.path.join(SOURCES, name), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) == 3 and parts[0].isdigit():
                out[(int(parts[0]), int(parts[1]))] = parts[2]
    return out


def load_quran():
    arabic, persian = _verses("quran-simple.txt"), _verses("fa.makarem.txt")
    assert len(arabic) == 6236 and len(persian) == 6236, (len(arabic), len(persian))
    docs = []
    for (s, a), ar in arabic.items():
        where = f"سوره {SURAHS[s - 1]}، آیه {a}".translate(FA_DIGITS)
        # Tanzil marks words the translator added with *...*; the stars only clutter.
        docs.append({"kind": "quran", "fa": clean(persian[(s, a)].replace("*", ""), persian=True), "ar": ar,
                     "source": where, "from": "قرآن کریم", "speaker": "quran", "surah": s, "ayah": a})
    return docs


def _blocks(raw):
    return [b["v"].strip() for b in json.loads(raw) if b.get("t") == "p" and b.get("v", "").strip()]


_SENTENCE = re.compile(r"(?<=[.!?؟:])\s+")


def _units(paras, most=320):
    """Arabic sentences, so a long paragraph can still be shared out. Many
    sermons run for pages without a full stop, so long stretches are also cut
    at the space nearest every `most` characters."""
    out = []
    for p in paras:
        for u in _SENTENCE.split(p):
            u = u.strip()
            while len(u) > most * 1.5:
                cut = u.rfind(" ", 0, most) + 1 or most
                out.append(u[:cut].strip())
                u = u[cut:].strip()
            if u:
                out.append(u)
    return out


def _pieces(paras, target):
    """Persian paragraphs as (paragraph number, text) pieces no longer than
    about `target`: a long paragraph is cut at sentence ends, so one chunk never
    has to swallow a whole page."""
    out = []
    for i, p in enumerate(paras):
        for u in (_units([p], most=target) if len(p) > target else [p]):
            out.append((i, u))
    return out


def _chunk(pieces, target):
    """Group pieces into chunks near `target` characters; pieces of one
    paragraph are joined with a space, paragraphs with a new line."""
    chunks, cur = [], []
    for piece in pieces:
        if cur and sum(len(t) for _, t in cur) + len(piece[1]) > target:
            chunks.append(cur)
            cur = []
        cur.append(piece)
    if cur:
        chunks.append(cur)
    return [["\n".join(" ".join(t for j, t in c if j == i) for i in dict.fromkeys(j for j, _ in c))]
            for c in chunks]


def _align(units, weights):
    """Share the Arabic out at the same length fractions as the Persian chunks."""
    n = len(weights)
    if n == 1:
        return [units]
    total, bounds, run = sum(weights) or 1, [], 0
    for w in weights[:-1]:
        run += w
        bounds.append(run / total)
    size = sum(map(len, units)) or 1
    parts, done, k = [[] for _ in range(n)], 0, 0
    for u in units:
        while k < n - 1 and (done + len(u) / 2) / size > bounds[k]:
            k += 1
        parts[k].append(u)
        done += len(u)
    return parts


def load_nahj():
    con = sqlite3.connect(os.path.join(SOURCES, "nahj.db"))
    rows = con.execute("select category, number, title, body_ar, body_fa from entries "
                       "order by case category when 'sermon' then 0 when 'letter' then 1 else 2 end, number").fetchall()
    docs = []
    for cat, num, title, body_ar, body_fa in rows:
        fa_paras = [clean(p, persian=True) for p in _blocks(body_fa)]
        ar_paras = [clean(p) for p in _blocks(body_ar)]
        if not fa_paras and not ar_paras:
            continue
        units = _units(ar_paras)
        # Small chunks: Nahj is dense, and one vector for a page-long stretch
        # blurs the many separate points in it.
        target = NAHJ_CHUNK
        chunks = _chunk(_pieces(fa_paras, target), target) or [[]]
        while len(chunks) > max(1, len(units)):  # never more parts than Arabic sentences
            target *= 1.5
            chunks = _chunk(_pieces(fa_paras, target), target)
        ar_parts = _align(units, [sum(map(len, c)) or 1 for c in chunks])
        name = f"نهج‌البلاغه، {NAHJ_SECTION[cat]} {num}".translate(FA_DIGITS)
        for i, (fa_part, ar_part) in enumerate(zip(chunks, ar_parts)):
            part = f" (بخش {i + 1} از {len(chunks)})".translate(FA_DIGITS) if len(chunks) > 1 else ""
            docs.append({"kind": "nahj", "fa": "\n".join(fa_part), "ar": " ".join(ar_part),
                         "source": name + part, "title": clean(title or "", persian=True),
                         "from": "امام علی", "speaker": "ali", "section": cat, "number": num})
    return docs


if __name__ == "__main__":
    fetch()
    q, n = load_quran(), load_nahj()
    print(len(q), "verses;", len(n), "nahj chunks")
    from collections import Counter
    print(Counter(d["section"] for d in n))
    print(n[0]["source"], "|", n[0]["fa"][:120])
    print(q[254]["source"], "|", q[254]["fa"][:120])

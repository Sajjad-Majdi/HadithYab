"""Text cleaning shared by the index builder and the live app."""
import re

_TATWEEL = "ـ"
_ZW = re.compile("[​‎‏﻿]")
_SPACES = re.compile(r"\s+")
# Harakat, the dagger alif, and the Quran's pause and recitation marks.
_HARAKAT = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")
# Letters only: Arabic-script punctuation (، ؛ ؟ « » ٪ ٫ ٬ ۔) sits inside
# U+0600-U+06FF too, so it is cut out explicitly.
_NON_WORD = re.compile(r"[^\w؀-ۿ]+|[،؛؟٪-٬۔«»]+")

# Persian text: Arabic letter forms become the Persian ones.
_FA_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"})
# Keyword matching also ignores hamza seats, so إيمان, ايمان and ایمان meet.
_FOLD_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
                           "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ء": ""})

# Who spoke the hadith, found by the earliest name in the text.
SPEAKERS = [
    ("prophet", "پیامبر اکرم", ["پیامبر", "رسول", "نبی", "النبی", "حضرت محمد"]),
    ("ali", "امام علی", ["امام علی", "امیرالمؤمنین", "امیر المؤمنین", "امیرمؤمنان", "علی علیه"]),
    ("fatima", "حضرت زهرا", ["فاطمه", "فاطمة", "زهرا"]),
    ("hasan", "امام حسن", ["امام حسن علیه", "مجتبی"]),
    ("husayn", "امام حسین", ["امام حسین"]),
    ("sajjad", "امام سجاد", ["سجاد", "زین العابدین", "امام زین", "علی بن الحسین"]),
    ("baqir", "امام باقر", ["باقر"]),
    ("sadiq", "امام صادق", ["صادق"]),
    ("kazim", "امام کاظم", ["کاظم", "موسی بن جعفر"]),
    ("rida", "امام رضا", ["رضا علیه", "امام رضا"]),
    ("jawad", "امام جواد", ["جواد", "محمد تقی"]),
    ("hadi", "امام هادی", ["هادی", "علی النقی", "امام دهم"]),
    ("askari", "امام عسکری", ["عسکری"]),
    ("mahdi", "امام زمان", ["مهدی", "امام زمان", "امام عصر", "قائم", "صاحب الزمان"]),
]
SPEAKER_LABELS = {key: label for key, label, _ in SPEAKERS}
SPEAKER_LABELS["other"] = "دیگران"
SPEAKER_LABELS["quran"] = "قرآن کریم"

# Books sit beside the hadith collection; a filter value can name one of them.
BOOKS = {"quran": "قرآن کریم", "nahj": "نهج‌البلاغه", "hadith": "سایر احادیث"}


def clean(text, persian=False):
    text = _ZW.sub("", (text or "").replace(_TATWEEL, ""))
    if persian:
        text = text.translate(_FA_MAP)
    return _SPACES.sub(" ", text).strip()


def fold(text):
    """Reduce Persian or Arabic text to a bare form for keyword matching."""
    text = _HARAKAT.sub("", _ZW.sub("", (text or "").replace(_TATWEEL, "")))
    text = text.translate(_FOLD_MAP).replace("‌", " ")
    return _SPACES.sub(" ", _NON_WORD.sub(" ", text)).strip().lower()


def speaker_of(text):
    """The infallible named first in `text` ("other" when none is)."""
    plain = _HARAKAT.sub("", text.translate(_FA_MAP))
    best, where = "other", len(plain) + 1
    for key, _label, needles in SPEAKERS:
        for n in needles:
            at = plain.find(n)
            if -1 < at < where:
                best, where = key, at
    return best


# In a query only unambiguous titles count, so "صادق بودن" (being truthful)
# never turns into a filter for Imam Sadiq.
_QUERY_NAMES = {
    "prophet": ["پیامبر اکرم", "پیامبر خدا", "پیغمبر", "پیامبر", "رسول خدا", "رسول الله", "حضرت محمد", "النبی"],
    "ali": ["امیرالمؤمنین", "امیر المؤمنین", "امیرمؤمنان", "امام علی", "حضرت علی"],
    "fatima": ["حضرت زهرا", "حضرت فاطمه", "فاطمه زهرا"],
    "hasan": ["امام حسن مجتبی", "امام حسن"],
    "husayn": ["امام حسین", "سیدالشهدا", "سید الشهدا"],
    "sajjad": ["امام سجاد", "امام زین العابدین"],
    "baqir": ["امام باقر", "امام محمد باقر"],
    "sadiq": ["امام صادق", "امام جعفر صادق"],
    "kazim": ["امام کاظم", "امام موسی کاظم"],
    "rida": ["امام رضا"],
    "jawad": ["امام جواد", "امام محمد تقی"],
    "hadi": ["امام هادی", "امام علی النقی"],
    "askari": ["امام حسن عسکری", "امام عسکری"],
    "mahdi": ["امام زمان", "امام مهدی", "امام عصر", "حضرت مهدی"],
}
_FILLER = re.compile(r"^(?:(?:احادیث|حدیث|روایات|روایت|سخنان|سخن|کلام|گفتار|از|های|ها)\s+)+|\s+(?:از|در|درباره)$")


def detect_speaker(query):
    """("ali", "مرگ") for "احادیث امام علی درباره مرگ"; (None, query) when no title is named."""
    plain = _HARAKAT.sub("", query.translate(_FA_MAP))
    best = None
    for key, names in _QUERY_NAMES.items():
        for name in names:
            at = plain.find(name)
            # Longest title at the earliest place wins: امام حسن عسکری beats امام حسن.
            if at != -1 and (best is None or (at, -len(name)) < (best[1], -len(best[2]))):
                best = (key, at, name)
    if not best:
        return None, query
    key, at, name = best
    rest = _SPACES.sub(" ", (plain[:at] + " " + plain[at + len(name):]))
    rest = re.sub(r"\s*\((?:ع|ص|س|عج)\)|\s*(?:علیه السلام|علیها السلام|صلی الله علیه و آله)", "", rest).strip()
    rest = re.sub(r"^(?:درباره|در باره|در مورد|راجع به)\s+", "", _FILLER.sub("", rest).strip()).strip()
    return key, rest or query

"""Smart search (جستجوی هوشواره‌ای): a Gemini agent that searches the index and answers with citations.

To keep it fast, the question is searched once before the model is called,
so a simple question is answered in a single model round. The model can
still call the tools for more rounds when the first results are not enough.
Every round streams, so text reaches the reader as soon as it is written.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests

from ..core.embed import GEMINI_BASE as BASE, next_key
from ..core.index import get_index
from ..core.text import SPEAKER_LABELS, SPEAKERS, detect_speaker

MODELS = [m.strip() for m in os.environ.get(
    "AGENT_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-latest").split(",") if m.strip()]
MAX_ROUNDS = 3
SEED_HITS = 8

SPEAKER_ENUM = ["all"] + [k for k, _, _ in SPEAKERS] + ["other"]
SPEAKER_HELP = "، ".join(f"{k}={SPEAKER_LABELS[k]}" for k in SPEAKER_ENUM[1:])

SYSTEM = f"""تو دستیار پژوهشی «حدیث‌یاب» هستی و به طلبه‌ها و پژوهشگران کمک می‌کنی. پایگاه تو حدود ۳۸ هزار حدیث شیعه با متن عربی و ترجمه فارسی است.

قانون‌ها:
۱. فقط بر پایه احادیثی جواب بده که ابزارها برگردانده‌اند. هرگز حدیث، منبع یا شماره‌ای از حافظه نساز.
۲. هر ادعا را با شماره حدیث به شکل [#شماره] مستند کن، مثل [#1203]. چند شماره پشت هم هم درست است: [#12][#40].
۳. نتیجه اولیه فقط نقطه شروع است. پیش از جواب، دست‌کم یک نوبت ۳ تا ۵ جستجوی هم‌زمان و متفاوت بفرست تا پرسش از همه زاویه‌ها پوشش داده شود:
   الف) search_hadith با مفهوم اصلی به فارسی، و جداگانه با بیان عربی آن به سبک روایات (مثلاً «ذكر الموت» برای «یاد مرگ»).
   ب) bm25_search با واژه‌های کلیدی عربی و فارسی و هم‌ریشه‌ها و مترادف‌ها (مثلاً «الموت الأجل القبر مرگ»).
   ج) اگر پرسش چند جنبه دارد، برای هر جنبه جستجوی جدا (مثلاً «آمادگی برای مرگ»، «سختی جان دادن»، «حال مؤمن هنگام مرگ»).
   د) keyword_search فقط برای عبارت دقیق، نام یا اصطلاحی که مطمئنی در متن هست.
۴. نتیجه‌ها را بخوان. اگر جنبه‌ای هنوز بی‌شاهد مانده یا نتیجه‌ها کم‌ربط بودند، نوبت دوم با واژه‌های تازه بزن. برای یافتن نقل‌های هم‌سوی یک حدیث مهم، similar_hadith را صدا بزن.
۵. فیلتر گوینده را در هر جستجو رعایت کن اگر کاربر معصوم خاصی خواسته است.
۶. جواب فارسی، روشن و فشرده باشد. احادیث را بر اساس موضوع دسته‌بندی کن. عبارت کلیدی عربی را اگر به فهم کمک می‌کند کوتاه نقل کن.
۷. این ابزار سند و اعتبار حدیث را بررسی نمی‌کند. فتوا نده. اگر احادیث با هم فرق دارند یا شاهد کم است، صادقانه بگو.
۸. اگر در نتیجه‌ها چیزی مرتبط نیافتی، همین را بگو و پیشنهاد جستجوی دیگر بده.
۹. برای پیامبر (ص)، برای حضرت زهرا (س)، برای امامان (ع) و برای امام زمان (عج) بنویس.
گوینده‌ها: {SPEAKER_HELP}"""

_SPEAKER_PARAM = {"type": "string", "enum": SPEAKER_ENUM,
                  "description": "محدود کردن به یک معصوم. all یعنی همه. " + SPEAKER_HELP}

TOOLS = [{"functionDeclarations": [
    {
        "name": "search_hadith",
        "description": "جستجوی معنایی. کل query به یک بردار معنا تبدیل می‌شود و احادیث هم‌معنا برمی‌گردند، حتی با واژه‌های متفاوت. "
                       "پس query باید یک عبارت طبیعی با فقط یک مفهوم باشد، مثل یک جمله کوتاه. فهرست واژه نده: "
                       "«الموت اطفال برزخ قیامت» چهار مفهوم را قاطی می‌کند و به هیچ‌کدام نمی‌رسد؛ برای هر مفهوم یک فراخوانی جدا بفرست. "
                       "نام معصوم را در query ننویس؛ برای آن speaker را بده.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "یک مفهوم در یک عبارت، مثل «پاداش صبر بر مصیبت»، «حال کودکانی که پیش از بلوغ می‌میرند» یا «فضل طلب العلم»."},
            "speaker": _SPEAKER_PARAM,
            "limit": {"type": "integer", "description": "تعداد نتیجه، ۱ تا ۱۵. پیش‌فرض ۸."},
        }, "required": ["query"]},
    },
    {
        "name": "bm25_search",
        "description": "جستجوی واژه‌ای رتبه‌دار (BM25). هر واژه جدا امتیاز می‌گیرد: واژه کم‌یاب‌تر وزن بیشتری دارد و حدیثی که واژه‌های بیشتری از query را دارد بالاتر می‌آید؛ "
                       "لازم نیست همه واژه‌ها باشند. پس این‌جا فهرست چند واژه درست است: واژه‌های کلیدی، هم‌ریشه‌ها و مترادف‌ها، "
                       "به‌ویژه واژه‌های عربی که در متن روایت می‌آیند، مثل «الموت القبر البرزخ». واژه‌های عام و نام معصوم فایده ندارند.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "چند واژه کلیدی، فارسی و عربی."},
            "speaker": _SPEAKER_PARAM,
            "limit": {"type": "integer", "description": "تعداد نتیجه، ۱ تا ۱۵. پیش‌فرض ۸."},
        }, "required": ["query"]},
    },
    {
        "name": "keyword_search",
        "description": "جستجوی دقیق واژه. احادیثی که همه واژه‌های terms در متن عربی یا ترجمه فارسی‌شان هست. "
                       "حرکات و شکل همزه و ی/ک مهم نیست. عبارت داخل \" \" باید عیناً پشت هم بیاید. "
                       "برای عبارت عربی شناخته‌شده، نام شخص یا اصطلاح فنی مناسب است. خروجی تعداد کل یافته‌ها را هم می‌دهد.",
        "parameters": {"type": "object", "properties": {
            "terms": {"type": "string", "description": "مثل «\"العلم نور\"» یا «همسایه حق»."},
            "speaker": _SPEAKER_PARAM,
            "limit": {"type": "integer", "description": "تعداد نتیجه، ۱ تا ۱۵. پیش‌فرض ۸."},
        }, "required": ["terms"]},
    },
    {
        "name": "get_hadith",
        "description": "متن کامل یک حدیث با شماره‌اش. وقتی متن در نتیجه‌ها کوتاه شده و تمامش لازم است.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "شماره حدیث."}}, "required": ["id"]},
    },
    {
        "name": "similar_hadith",
        "description": "احادیث هم‌معنای یک حدیث مشخص. برای یافتن نقل‌های دیگر و شواهد هم‌سو.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "شماره حدیث مبنا."},
            "limit": {"type": "integer", "description": "۱ تا ۱۰. پیش‌فرض ۵."},
        }, "required": ["id"]},
    },
]}]


def _brief(card, fa_len=700, ar_len=350):
    return {"id": card["id"], "speaker": card["speaker_label"], "source": card["source"],
            "fa": card["fa"][:fa_len], "ar": card["ar"][:ar_len]}


def _limit(args, default, top):
    try:
        return max(1, min(top, int(args.get("limit") or default)))
    except (TypeError, ValueError):
        return default


def clean_args(args):
    """Models keep writing the speaker into the query ("... پیامبر"). The filter
    already covers that and the name only blurs the meaning, so it moves out."""
    args = dict(args or {})
    if args.get("query"):
        found, rest = detect_speaker(str(args["query"]))
        if found:
            args["query"] = rest
            if args.get("speaker") in (None, "", "all"):
                args["speaker"] = found
    return args


def run_tool(name, args):
    """Run one tool call. Returns (result for the model, cards for the reader)."""
    index = get_index()
    speaker = args.get("speaker") or args.get("imam") or None
    if speaker == "all":
        speaker = None
    if name.endswith("search_hadith"):
        query = str(args.get("query") or args.get("q") or "")
        try:
            cards = index.semantic(query, speaker, _limit(args, 8, 15))
        except Exception:  # embedding API down: words are better than nothing
            cards = index.lexical(query, speaker, _limit(args, 8, 15))
        return {"results": [_brief(c) for c in cards]}, cards
    if name.endswith("bm25_search"):
        cards = index.lexical(str(args.get("query") or args.get("terms") or ""), speaker, _limit(args, 8, 15))
        return {"results": [_brief(c) for c in cards]}, cards
    if name.endswith("keyword_search"):
        cards, total = index.keyword(str(args.get("terms") or args.get("query") or ""), speaker, _limit(args, 8, 15))
        return {"total_matches": total, "results": [_brief(c) for c in cards]}, cards
    if name.endswith("get_hadith"):
        i = int(args.get("id", -1))
        if not 0 <= i < len(index):
            return {"error": "چنین شماره‌ای نیست"}, []
        card = index.card(i)
        return _brief(card, 6000, 4000), [card]
    if name.endswith("similar_hadith"):
        i = int(args.get("id", -1))
        if not 0 <= i < len(index):
            return {"error": "چنین شماره‌ای نیست"}, []
        cards = index.similar(i, None, _limit(args, 5, 10))
        return {"results": [_brief(c) for c in cards]}, cards
    return {"error": f"ابزار ناشناخته: {name}"}, []


class Interrupted(Exception):
    """The connection dropped after part of the answer was already shown."""


def _stream(contents, force_tools=False):
    """One streamed model round. Yields ("text", str) and finally ("parts", list)."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "tools": TOOLS,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048,
                             "thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    if force_tools:  # the first turn must search, never answer from the seed alone
        body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
    error = None
    for model in MODELS:
        for _ in range(2):
            try:
                r = requests.post(f"{BASE}/v1beta/models/{model}:streamGenerateContent?alt=sse",
                                  headers={"x-goog-api-key": next_key()}, json=body, stream=True, timeout=60)
            except requests.RequestException as e:
                error = e
                continue
            if r.status_code != 200:
                error = RuntimeError(f"{model} {r.status_code}: {r.text[:200]}")
                if r.status_code == 400 and "thinking" in r.text.lower():
                    body["generationConfig"].pop("thinkingConfig", None)
                continue
            parts, wrote = [], False
            try:
                # Split raw bytes: decoded text would also break at U+0085, and
                # the UTF-8 of Persian letters like «م» contains that byte.
                for raw in r.iter_lines():
                    if not raw.startswith(b"data:"):
                        continue
                    chunk = json.loads(raw[5:].decode("utf-8"))
                    for cand in chunk.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            parts.append(part)
                            if part.get("text") and not part.get("thought"):
                                wrote = True
                                yield "text", part["text"]
            except (requests.RequestException, ValueError) as e:
                if not wrote:  # nothing reached the reader yet: try again cleanly
                    error = e
                    continue
                raise Interrupted() from e
            yield "parts", parts
            return
    raise error or RuntimeError("no model answered")


def _merge_parts(parts):
    """Join streamed text fragments back into whole parts for the history."""
    merged = []
    for p in parts:
        if merged and "text" in p and "text" in merged[-1] and set(p) <= {"text", "thoughtSignature"} \
                and set(merged[-1]) <= {"text", "thoughtSignature"} and "thoughtSignature" not in merged[-1]:
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + p["text"],
                          **({"thoughtSignature": p["thoughtSignature"]} if "thoughtSignature" in p else {})}
        else:
            merged.append(dict(p))
    return merged


def research(question, speaker=None, seed_query=None):
    """Yield events: step, step_done, cards, delta, done, error.

    Every tool call is announced as a step before it runs and closed with
    step_done and its hit count, so the page can draw a live search trace.
    """
    index = get_index()
    seed_query = seed_query or question
    yield {"type": "step", "id": 0, "tool": "search_hadith", "args": {"query": seed_query, "speaker": speaker}}
    try:
        seed = index.semantic(seed_query, speaker, SEED_HITS)
    except Exception:
        seed = index.lexical(seed_query, speaker, SEED_HITS)
    yield {"type": "step_done", "id": 0, "count": len(seed), "ids": [c["id"] for c in seed]}
    yield {"type": "cards", "cards": seed}
    note = f" (فقط احادیث {SPEAKER_LABELS[speaker]})" if speaker and speaker in SPEAKER_LABELS else ""
    seed_text = json.dumps([_brief(c) for c in seed], ensure_ascii=False)
    contents = [{"role": "user", "parts": [{"text":
        f"پرسش: {question}{note}\n\nنتیجه‌های اولیه جستجوی معنایی برای همین پرسش:\n{seed_text}"}]}]
    step = 0

    for _round in range(MAX_ROUNDS):
        parts = []
        try:
            for kind, value in _stream(contents, force_tools=_round == 0):
                if kind == "text":
                    yield {"type": "delta", "text": value}
                else:
                    parts = value
        except Interrupted:
            yield {"type": "done", "partial": True}
            return
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            yield {"type": "done"}
            return
        contents.append({"role": "model", "parts": _merge_parts(parts)})

        # Cleaned copies: the model's own calls stay untouched in the history,
        # where their thought signatures are checked.
        args_list = [clean_args(c.get("args")) for c in calls]
        ids = []
        for call, args in zip(calls, args_list):
            step += 1
            ids.append(step)
            yield {"type": "step", "id": step, "tool": call["name"], "args": args}
        # Independent searches run side by side; the model asked for them in one turn.
        with ThreadPoolExecutor(min(6, len(calls))) as pool:
            outcomes = list(pool.map(lambda ca: run_tool(ca[0]["name"], ca[1]), zip(calls, args_list)))
        replies = []
        for call, sid, (result, cards) in zip(calls, ids, outcomes):
            yield {"type": "step_done", "id": sid, "count": len(cards), "ids": [c["id"] for c in cards]}
            if cards:
                yield {"type": "cards", "cards": cards}
            replies.append({"functionResponse": {"name": call["name"], "response": result,
                                                 **({"id": call["id"]} if "id" in call else {})}})
        contents.append({"role": "user", "parts": replies})

    contents.append({"role": "user", "parts": [{"text": "دیگر ابزار صدا نزن و با همین یافته‌ها جواب بده."}]})
    try:
        for kind, value in _stream(contents):
            if kind == "text":
                yield {"type": "delta", "text": value}
    except Interrupted:
        yield {"type": "done", "partial": True}
        return
    yield {"type": "done"}

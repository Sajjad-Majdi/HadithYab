"""Server-rendered pages: home, one text, the Quran, Nahj al-Balagha, hadiths by speaker, about.

Every page carries its own title, description, canonical link and JSON-LD, and
its main text sits in the HTML itself, so search engines and AI readers that
never run the page script still see it.
"""
import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates

from . import browse, seo
from .core.index import get_index
from .core.text import detect_speaker

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
log = logging.getLogger("hadithyab")
router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(WEB, "templates"))


def setup(version):
    templates.env.globals.update(fa=browse.fa_num, site=seo.SITE, version=version,
                                 google_verify=os.environ.get("GOOGLE_SITE_VERIFICATION", ""))


def render(request, template, meta, jsonld, status_code=200, **context):
    """A page with its own title, description, canonical link and JSON-LD."""
    path = meta.pop("path", request.url.path)
    meta.setdefault("canonical", seo.SITE + path)
    meta.setdefault("description", seo.DESCRIPTION)
    context.setdefault("trail", [])
    return templates.TemplateResponse(request, template, {"seo": meta, "jsonld": jsonld, **context},
                                      status_code=status_code)


def not_found(request):
    return render(request, "browse.html", {"title": "صفحه پیدا نشد | حدیث‌یاب", "h1": "این صفحه پیدا نشد",
                                           "robots": "noindex"}, [], status_code=404,
                  trail=[("پیدا نشد", None)], intro="نشانی را دوباره ببینید، یا از صفحه اصلی جستجو کنید.")


def _clip(text, size):
    text = " ".join(text.split())
    return text if len(text) <= size else text[:size].rsplit(" ", 1)[0] + "…"


def _cards(ids):
    index = get_index()
    return [index.card(i) for i in ids]


def _nahj_trail(doc):
    one, plural, _ = browse.NAHJ_SECTIONS[doc["section"]]
    return [("نهج‌البلاغه", "/nahj"), (plural, f"/nahj/{doc['section']}"),
            (f"{one} {browse.fa_num(doc['number'])}", f"/nahj/{doc['section']}/{doc['number']}")]


# ---- home and one text ----------------------------------------------------------

@router.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home(request: Request, q: str = Query("", max_length=400), m: str = "", s: str = ""):
    q = " ".join(q.split())
    if not q:
        return render(request, "index.html", {"title": "حدیث‌یاب | جستجوی معنایی قرآن، نهج‌البلاغه و احادیث",
                                              "og_title": "حدیث‌یاب", "path": "/"},
                      seo.website_ld(), hadith=None, q="", results=None)
    # Readers without script (AI assistants, link previews) get the top matches
    # in the HTML; the page script then runs its own search over them.
    index = get_index()
    scope = s if s in browse.SCOPES else None
    if m == "keyword":
        results, _ = index.keyword(q, scope, 10)
    else:
        try:
            found, rest = detect_speaker(q)
            results = index.semantic(rest if found else q, found or scope, 10)
        except Exception:
            log.exception("page search failed, falling back to BM25")
            results = index.lexical(q, scope, 10)
    meta = {"title": f"{_clip(q, 60)} | جستجو در حدیث‌یاب", "path": "/", "robots": "noindex, follow",
            "description": f"آیات و احادیث درباره «{_clip(q, 80)}» با متن عربی، ترجمه فارسی و منبع."}
    return render(request, "index.html", meta, [], hadith=None, q=q, results=results)


@router.get("/h/{doc_id}", response_class=HTMLResponse)
def text_page(request: Request, doc_id: int):
    index = get_index()
    if not 0 <= doc_id < len(index):
        raise HTTPException(404, "چنین حدیثی نیست")
    card, doc = index.card(doc_id), index.docs[doc_id]
    name = browse.heading(card)
    if card["kind"] == "quran":
        trail = [("قرآن کریم", "/quran"), (f"سوره {browse.surah(doc['surah'])['name']}", f"/quran/{doc['surah']}")]
    elif card["kind"] == "nahj":
        trail = _nahj_trail(doc)
        if doc.get("title"):
            name = f"{name}: {doc['title']}"
    else:
        trail = [("احادیث", "/hadith"), (f"احادیث {browse.speaker_label(card['speaker'])}" if card["speaker"] != "other"
                                         else "دیگر احادیث", f"/hadith/{card['speaker']}")]
    gist = browse.gist(card, 64)
    url = f"{seo.SITE}/h/{doc_id}"
    lead = f"{card['fa']} ({card['source']})" if card["source"] else card["fa"]
    meta = {"title": f"{name}: «{gist}» | حدیث‌یاب", "og_title": f"{name} | حدیث‌یاب", "og_type": "article",
            "h1": f"{name}: {gist}", "description": _clip(lead, 160)}
    try:
        similar = index.similar(doc_id, None, 6)
    except Exception:
        log.exception("similar failed")
        similar = []
    jsonld = [seo.text_ld(card, url, name), seo.breadcrumbs_ld(trail + [(f"#{browse.fa_num(doc_id)}", None)])]
    return render(request, "index.html", meta, jsonld, hadith=card, similar=similar, q="", results=None)


# ---- the Quran -------------------------------------------------------------------

@router.get("/quran", response_class=HTMLResponse)
def quran_index(request: Request):
    items = [{"href": f"/quran/{s['n']}", "label": f"{browse.fa_num(s['n'])}. سوره {s['name']}",
              "note": f"{browse.fa_num(len(s['ids']))} آیه"} for s in browse.tree()["surahs"]]
    meta = {"title": "قرآن کریم با ترجمه فارسی مکارم شیرازی | ۱۱۴ سوره | حدیث‌یاب", "h1": "قرآن کریم با ترجمه فارسی",
            "description": "متن کامل قرآن کریم، ۱۱۴ سوره و ۶۲۳۶ آیه، با متن عربی تنزیل و ترجمه فارسی آیت‌الله مکارم شیرازی. هر آیه پیوند ثابت دارد."}
    trail = [("قرآن کریم", None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail, groups=[{"links": items}],
                  intro="سوره را انتخاب کنید تا همه آیه‌هایش را با ترجمه فارسی بخوانید. برای یافتن آیه با موضوع، در صفحه اصلی جستجو کنید.")


@router.get("/quran/{n}", response_class=HTMLResponse)
def quran_surah(request: Request, n: int):
    s = browse.surah(n)
    if not s:
        raise HTTPException(404, "چنین سوره‌ای نیست")
    cards = _cards(s["ids"])
    name = f"سوره {s['name']}"
    first = cards[1] if n == 1 and len(cards) > 1 else cards[0]
    meta = {"title": f"{name} با ترجمه فارسی | متن عربی و ترجمه مکارم شیرازی | حدیث‌یاب",
            "h1": f"{name} با ترجمه فارسی",
            "description": _clip(f"متن عربی و ترجمه فارسی {name}، {browse.fa_num(len(cards))} آیه. {first['fa']}", 160)}

    def nav(k):
        t = browse.surah(k)
        return {"href": f"/quran/{k}", "label": f"سوره {t['name']}"} if t else None

    trail = [("قرآن کریم", "/quran"), (name, None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail, cards=cards,
                  prev=nav(n - 1), next=nav(n + 1),
                  intro=f"{name}، سوره {browse.fa_num(n)} قرآن کریم، {browse.fa_num(len(cards))} آیه دارد. "
                        "متن عربی از تنزیل و ترجمه فارسی از آیت‌الله مکارم شیرازی است.")


# ---- Nahj al-Balagha ---------------------------------------------------------------

@router.get("/nahj", response_class=HTMLResponse)
def nahj_index(request: Request):
    tree = browse.tree()["nahj"]
    items = [{"href": f"/nahj/{k}", "label": f"{of} نهج‌البلاغه", "note": f"{browse.fa_num(len(tree[k]))} {one}"}
             for k, (one, _, of) in browse.NAHJ_SECTIONS.items()]
    meta = {"title": "نهج‌البلاغه با ترجمه فارسی | خطبه‌ها، نامه‌ها و حکمت‌ها | حدیث‌یاب", "h1": "نهج‌البلاغه با ترجمه فارسی",
            "description": "متن عربی و ترجمه فارسی نهج‌البلاغه امام علی (ع): خطبه‌ها، نامه‌ها و حکمت‌ها با ترجمه محمد دشتی و پیوند ثابت برای هر بخش."}
    trail = [("نهج‌البلاغه", None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail,
                  groups=[{"links": items, "wide": True}],
                  intro="نهج‌البلاغه گزیده سخنان امام علی (ع) است که سید رضی گرد آورده است: خطبه‌ها، نامه‌ها و سخنان کوتاه "
                        "(حکمت‌ها). ترجمه فارسی از محمد دشتی است.")


@router.get("/nahj/{section}", response_class=HTMLResponse)
def nahj_section(request: Request, section: str):
    if section not in browse.NAHJ_SECTIONS:
        raise HTTPException(404, "چنین بخشی نیست")
    one, plural, of = browse.NAHJ_SECTIONS[section]
    index = get_index()
    items = [{"href": f"/nahj/{section}/{it['n']}", "label": f"{one} {browse.fa_num(it['n'])}",
              "note": it["title"] or browse.gist(index.card(it["ids"][0]), 48)} for it in browse.tree()["nahj"][section]]
    meta = {"title": f"{of} نهج‌البلاغه با ترجمه فارسی | حدیث‌یاب", "h1": f"{of} نهج‌البلاغه",
            "description": f"فهرست {of} نهج‌البلاغه امام علی (ع) با متن عربی و ترجمه فارسی محمد دشتی؛ {browse.fa_num(len(items))} {one}."}
    trail = [("نهج‌البلاغه", "/nahj"), (plural, None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail,
                  groups=[{"links": items, "wide": True}])


@router.get("/nahj/{section}/{n}", response_class=HTMLResponse)
def nahj_page(request: Request, section: str, n: int):
    item = browse.nahj_item(section, n) if section in browse.NAHJ_SECTIONS else None
    if not item:
        raise HTTPException(404, "چنین بخشی نیست")
    one, plural, _ = browse.NAHJ_SECTIONS[section]
    cards = _cards(item["ids"])
    name = f"{one} {browse.fa_num(n)} نهج‌البلاغه"
    head = f"{name}: {item['title']}" if item["title"] else name
    meta = {"title": f"{head} | متن و ترجمه | حدیث‌یاب", "h1": head,
            "description": _clip(f"متن عربی و ترجمه فارسی {name}. {browse.gist(cards[0], 200)}", 160)}
    items = browse.tree()["nahj"][section]
    at = items.index(item)

    def nav(j):
        return {"href": f"/nahj/{section}/{items[j]['n']}", "label": f"{one} {browse.fa_num(items[j]['n'])}"} \
            if 0 <= j < len(items) else None

    trail = [("نهج‌البلاغه", "/nahj"), (plural, f"/nahj/{section}"), (f"{one} {browse.fa_num(n)}", None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail, cards=cards,
                  prev=nav(at - 1), next=nav(at + 1))


# ---- hadiths by speaker ------------------------------------------------------------

def _speaker_name(key):
    return browse.speaker_label(key) if key != "other" else "دیگر بزرگان"


@router.get("/hadith", response_class=HTMLResponse)
def hadith_index(request: Request):
    tree = browse.tree()["speakers"]
    items = [{"href": f"/hadith/{k}", "label": f"احادیث {_speaker_name(k)}", "note": f"{browse.fa_num(len(tree[k]))} حدیث"}
             for k in browse.SPEAKER_KEYS if tree.get(k)]
    total = sum(len(v) for v in tree.values())
    meta = {"title": "احادیث پیامبر و اهل‌بیت با ترجمه فارسی و منبع | حدیث‌یاب", "h1": "احادیث معصومین",
            "description": f"{browse.fa_num(total)} حدیث از پیامبر اکرم، حضرت زهرا و دوازده امام با متن عربی، ترجمه فارسی و نام منبع، دسته‌بندی‌شده بر پایه گوینده."}
    trail = [("احادیث", None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail, groups=[{"links": items}],
                  intro="احادیث را بر پایه گوینده مرور کنید، یا در صفحه اصلی موضوع را بنویسید تا حدیث‌های هم‌معنا را بیابید. "
                        "منبع هر حدیث کنار آن آمده است؛ حدیث‌یاب سند را بررسی نمی‌کند.")


@router.get("/hadith/{speaker}", response_class=HTMLResponse)
def hadith_speaker(request: Request, speaker: str, page: int = Query(1, ge=1)):
    ids = browse.tree()["speakers"].get(speaker) if speaker in browse.SPEAKER_KEYS else None
    pages = browse.speaker_pages(speaker)
    if not ids or page > pages:
        raise HTTPException(404, "چنین صفحه‌ای نیست")
    label = _speaker_name(speaker)
    suffix = f"، صفحه {browse.fa_num(page)}" if page > 1 else ""
    meta = {"title": f"احادیث {label} با ترجمه فارسی و منبع{suffix} | حدیث‌یاب", "h1": f"احادیث {label}{suffix}",
            "path": f"/hadith/{speaker}" + (f"?page={page}" if page > 1 else ""),
            "description": f"{browse.fa_num(len(ids))} حدیث از {label} با متن عربی، ترجمه فارسی و منبع هر حدیث{suffix}."}
    trail = [("احادیث", "/hadith"), (f"احادیث {label}", None)]
    return render(request, "browse.html", meta, [seo.breadcrumbs_ld(trail)], trail=trail,
                  cards=_cards(browse.speaker_slice(speaker, page)), page=page, pages=pages, base=f"/hadith/{speaker}",
                  intro=f"{browse.fa_num(len(ids))} حدیث از {label}، هر کدام با متن عربی، ترجمه فارسی و منبع." if page == 1 else None)


# ---- about, robots, llms.txt, sitemaps ------------------------------------------------

@router.get("/about", response_class=HTMLResponse)
def about(request: Request):
    meta = {"title": "درباره حدیث‌یاب و پرسش‌های رایج | حدیث‌یاب", "h1": "درباره حدیث‌یاب",
            "description": "حدیث‌یاب چیست، چه متن‌هایی دارد، جستجوی معنایی و هوشواره‌ای چطور کار می‌کند و متن‌ها و ترجمه‌ها از کجا آمده‌اند."}
    return render(request, "about.html", meta, [seo.faq_ld(), seo.breadcrumbs_ld([("درباره", None)])],
                  faq=seo.FAQ, counts=get_index().book_counts)


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return seo.ROBOTS


@router.get("/llms.txt", response_class=PlainTextResponse)
def llms():
    return seo.llms_txt(get_index().book_counts)


def _xml(body):
    return Response(body, media_type="application/xml", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/sitemap.xml")
def sitemap():
    return _xml(seo.sitemap_index(len(get_index())))


@router.get("/sitemap-pages.xml")
def sitemap_pages():
    tree = browse.tree()
    paths = ["/", "/about", "/quran", "/nahj", "/hadith"] + [f"/quran/{s['n']}" for s in tree["surahs"]]
    for sec, items in tree["nahj"].items():
        paths += [f"/nahj/{sec}"] + [f"/nahj/{sec}/{it['n']}" for it in items]
    for k in browse.SPEAKER_KEYS:
        if tree["speakers"].get(k):
            paths += [f"/hadith/{k}"] + [f"/hadith/{k}?page={p}" for p in range(2, browse.speaker_pages(k) + 1)]
    return _xml(seo.sitemap_pages(paths))


@router.get("/sitemap-texts-{chunk}.xml")
def sitemap_texts(chunk: int):
    total = len(get_index())
    if not 1 <= chunk <= -(-total // seo.SITEMAP_CHUNK):
        raise HTTPException(404, "چنین نقشه‌ای نیست")
    return _xml(seo.sitemap_texts(chunk, total))

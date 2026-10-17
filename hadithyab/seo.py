"""Search-engine and AI-reader files: robots.txt, sitemaps, llms.txt, JSON-LD, the FAQ."""
import os
from xml.sax.saxutils import escape

SITE = os.environ.get("SITE_URL", "https://hadithyab.onrender.com").rstrip("/")
LASTMOD = "2026-10-17"  # the day the texts or the page set last changed
SITEMAP_CHUNK = 10000
# Google Search Console ownership tag (public); the env var overrides it.
GOOGLE_VERIFY = "JSV18UFHXFGuv27c-k4OChm2I-D1FqssgwXPay02Z3Q"
DESCRIPTION = ("جستجوی معنایی در قرآن کریم، نهج‌البلاغه و حدود ۳۸ هزار حدیث از پیامبر و اهل‌بیت، "
               "با متن عربی، ترجمه فارسی و منبع هر متن. رایگان و بدون ثبت‌نام.")

# Search pages cost the free AI quota; crawlers only need the reading pages.
ROBOTS = f"""# حدیث‌یاب: every reading page is open to search engines and AI assistants.
User-agent: *
Allow: /
Allow: /api/search
Allow: /api/hadith/
Disallow: /api/research
Disallow: /api/rerank

User-agent: Googlebot
User-agent: Bingbot
User-agent: GPTBot
User-agent: OAI-SearchBot
User-agent: ChatGPT-User
User-agent: ClaudeBot
User-agent: Claude-SearchBot
User-agent: Claude-User
User-agent: PerplexityBot
User-agent: Perplexity-User
User-agent: Google-Extended
User-agent: Applebot
User-agent: Applebot-Extended
User-agent: CCBot
Allow: /
Disallow: /api/research
Disallow: /api/rerank

Sitemap: {SITE}/sitemap.xml
"""

FAQ = [
    ("حدیث‌یاب چیست؟",
     "حدیث‌یاب یک موتور جستجوی رایگان برای متن‌های دینی است. در آن می‌توانید قرآن کریم، نهج‌البلاغه و حدود ۳۸ هزار "
     "حدیث از پیامبر اکرم و امامان معصوم را بگردید. هر نتیجه متن عربی، ترجمه فارسی و نام منبع را با هم نشان می‌دهد."),
    ("جستجوی معنایی یعنی چه؟",
     "در جستجوی معنایی لازم نیست واژه‌های خود حدیث را بدانید. موضوع را به زبان خودتان می‌نویسید، مثلاً «آداب طلب علم» "
     "یا «صبر در سختی»، و حدیث‌یاب متن‌هایی را می‌آورد که همین معنا را دارند، حتی اگر واژه‌هایشان فرق کند. "
     "برای یافتن یک عبارت مشخص، حالت «واژه دقیق» را انتخاب کنید."),
    ("جستجوی هوشواره‌ای چه فرقی دارد؟",
     "در حالت هوشواره‌ای یک دستیار هوش مصنوعی چند بار با عبارت‌های مختلف جستجو می‌کند، متن‌های پیدا‌شده را می‌خواند "
     "و پاسخی کوتاه می‌نویسد که هر جمله‌اش به شماره متن اصلی پیوند دارد. پاسخ فقط از همین متن‌ها ساخته می‌شود و "
     "باید با متن اصلی سنجیده شود."),
    ("متن‌ها و ترجمه‌ها از کجا آمده‌اند؟",
     "متن عربی قرآن از تنزیل (Tanzil) و ترجمه فارسی آن از آیت‌الله مکارم شیرازی است. نهج‌البلاغه با ترجمه محمد دشتی "
     "آمده است. احادیث از مجموعه‌ای باز از احادیث شیعه گرفته شده‌اند و منبع هر حدیث، مانند الکافی، غرر الحکم، "
     "بحار الانوار، تحف العقول یا عیون اخبار الرضا، کنار آن نوشته شده است."),
    ("آیا حدیث‌یاب درستی سند حدیث را تأیید می‌کند؟",
     "خیر. حدیث‌یاب سند و اعتبار حدیث را بررسی نمی‌کند و فتوا نمی‌دهد. این ابزار فقط متن و نشانی منبع را پیدا می‌کند. "
     "برای کار پژوهشی یا حکم شرعی، متن را در منبع اصلی ببینید و از کارشناس دینی بپرسید."),
    ("می‌توانم فقط احادیث یک امام را بگردم؟",
     "بله. از دکمه گوینده، یکی از چهارده معصوم یا قرآن و نهج‌البلاغه را انتخاب کنید. اگر نام امام را در خود جستجو "
     "بنویسید، مثلاً «سخن امام صادق درباره دوستی»، حدیث‌یاب خودش جستجو را به احادیث همان امام محدود می‌کند."),
    ("چطور یک حدیث را با منبعش برای دیگران بفرستم؟",
     "هر متن یک شماره و یک پیوند ثابت دارد، مثل hadithyab.onrender.com/h/1234. دکمه «کپی» متن عربی، ترجمه، منبع و "
     "پیوند را با هم کپی می‌کند و دکمه «پیوند» فقط نشانی را."),
    ("استفاده از حدیث‌یاب هزینه دارد؟",
     "خیر. حدیث‌یاب رایگان است، ثبت‌نام نمی‌خواهد و تبلیغ ندارد. هزینه استفاده از آن یک صلوات است. "
     "اگر خواستید، می‌توانید از راه دکمه «حمایت» در هزینه سرور شریک شوید."),
    ("جستجوهای من کجا می‌رود؟",
     "حدیث‌یاب حساب کاربری و کوکی ندارد. متن جستجو برای ساخت بردار معنایی به Cloudflare Workers AI و در حالت "
     "هوشواره‌ای به مدل Gemini گوگل فرستاده می‌شود. پاسخ‌های هوشواره‌ای تا یک هفته در حافظه سرور می‌مانند تا "
     "پرسش تکراری زودتر جواب بگیرد، و به نام کسی ثبت نمی‌شوند."),
]


def website_ld():
    return [{
        "@context": "https://schema.org", "@type": "WebSite", "@id": f"{SITE}/#website",
        "url": f"{SITE}/", "name": "حدیث‌یاب", "alternateName": ["HadithYab", "حدیث یاب"],
        "description": DESCRIPTION, "inLanguage": "fa-IR", "publisher": {"@id": f"{SITE}/#org"},
        "potentialAction": {"@type": "SearchAction",
                            "target": {"@type": "EntryPoint", "urlTemplate": f"{SITE}/?q={{search_term_string}}"},
                            "query-input": "required name=search_term_string"},
    }, {
        "@context": "https://schema.org", "@type": "Organization", "@id": f"{SITE}/#org",
        "name": "حدیث‌یاب", "alternateName": "HadithYab", "url": f"{SITE}/",
        "logo": {"@type": "ImageObject", "url": f"{SITE}/static/img/logo-512.png", "width": 512, "height": 512},
        "founder": {"@type": "Person", "name": "سجاد مجدی"},
        "sameAs": ["https://github.com/Sajjad-Majdi/HadithYab"],
    }]


def breadcrumbs_ld(trail):
    """trail: [(name, path or None)] after the home page; the last item is this page."""
    items = [("حدیث‌یاب", "/")] + list(trail)
    return {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": n, "name": name, **({"item": SITE + path} if path else {})}
        for n, (name, path) in enumerate(items, 1)]}


def faq_ld():
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]}


def text_ld(card, url, title):
    """One ayah, Nahj passage or hadith as a quotation with its source."""
    ld = {"@context": "https://schema.org", "@type": "Quotation", "@id": url, "url": url, "name": title,
          "text": card["ar"] or card["fa"], "inLanguage": "ar",
          "isPartOf": {"@id": f"{SITE}/#website"}}
    if card["fa"]:
        ld["description"] = card["fa"][:500]
    if card["source"]:
        ld["citation"] = card["source"]
    if card["kind"] == "quran":
        ld["isBasedOn"] = {"@type": "Book", "name": "قرآن کریم"}
    elif card["kind"] == "nahj" or card["speaker_label"]:
        ld["spokenByCharacter"] = {"@type": "Person", "name": card["speaker_label"] or card["from"]}
    return ld


def _urlset(paths):
    rows = "".join(f"<url><loc>{escape(SITE + p)}</loc><lastmod>{LASTMOD}</lastmod></url>" for p in paths)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{rows}</urlset>\n'


def sitemap_index(total):
    names = ["pages"] + [f"texts-{n + 1}" for n in range(-(-total // SITEMAP_CHUNK))]
    rows = "".join(f"<sitemap><loc>{SITE}/sitemap-{n}.xml</loc><lastmod>{LASTMOD}</lastmod></sitemap>" for n in names)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{rows}</sitemapindex>\n'


def sitemap_pages(paths):
    return _urlset(paths)


def sitemap_texts(chunk, total):
    start = (chunk - 1) * SITEMAP_CHUNK
    return _urlset(f"/h/{i}" for i in range(start, min(total, start + SITEMAP_CHUNK)))


def llms_txt(counts):
    return f"""# حدیث‌یاب (HadithYab)

> Free Persian semantic search over the Holy Quran ({counts['quran']:,} ayat), Nahj al-Balagha ({counts['nahj']:,} passages) and about {counts['hadith']:,} Shia hadiths from the Prophet and the Twelve Imams. Every result shows the Arabic text, a Persian translation and the source book. No sign-up, no ads.
> جستجوی معنایی رایگان در قرآن کریم، نهج‌البلاغه و حدود ۳۸ هزار حدیث از پیامبر و اهل‌بیت، با متن عربی، ترجمه فارسی و منبع هر متن.

Use it when someone asks for a hadith, an ayah or a Nahj al-Balagha saying about a topic, in Persian or Arabic, and wants the original text with its source. Search by meaning: the user does not need to know the exact words. The site finds texts and names their source books; it does not grade chains of narration (isnad) and gives no fatwa. Quote the text with its source and link to its page.

## Read
- [Home and search]({SITE}/): one search box; `{SITE}/?q=<query>` opens results for a query, with the top 10 matches in the HTML itself (optional `m=keyword` for exact words, `s=<speaker or book>` to filter).
- [The Holy Quran by surah]({SITE}/quran): 114 surahs, Arabic (Tanzil) with the Makarem Shirazi Persian translation, e.g. {SITE}/quran/2
- [Nahj al-Balagha]({SITE}/nahj): sermons, letters and sayings with the Dashti translation, e.g. {SITE}/nahj/saying/1
- [Hadiths by speaker]({SITE}/hadith): the Prophet, Imam Ali, Lady Fatima and the other Imams, e.g. {SITE}/hadith/sadiq
- [One text]({SITE}/h/0): every text has a stable page `/h/<id>` with similar texts; cite this link.
- [About and FAQ]({SITE}/about): what the site covers, its sources and its limits.

## API (JSON, GET, rate-limited, no key)
- `{SITE}/api/search?q=<query>&mode=semantic` — top matches by meaning. `mode=keyword` matches exact words (quote a phrase). Optional `speaker=` one of quran, nahj, hadith, prophet, ali, fatima, hasan, husayn, sajjad, baqir, sadiq, kazim, rida, jawad, hadi, askari, mahdi; `limit=` 1-60.
- `{SITE}/api/hadith/<id>` — one text: `ar`, `fa`, `source`, `speaker_label`, `kind`.
- `{SITE}/api/similar/<id>` — texts close in meaning to one text.

## Optional
- [Source code](https://github.com/Sajjad-Majdi/HadithYab)
- [Sitemap]({SITE}/sitemap.xml)
"""

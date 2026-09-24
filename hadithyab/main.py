"""حدیث‌یاب web server (FastAPI): the page, a JSON API and a streamed research mode."""
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core.index import get_index
from .guard import Guard
from .core.text import BOOKS, SPEAKER_LABELS, SPEAKERS, detect_speaker
from .research.agent import research
from .research.rerank import rerank

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VERSION = "2.0.0"
SPEAKER_ORDER = [k for k, _, _ in SPEAKERS] + ["other"]

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hadithyab")


@asynccontextmanager
async def lifespan(_app):
    # Load the index in the background: the port opens at once (Render waits
    # for it), and a search that arrives early simply waits for the load.
    threading.Thread(target=get_index, daemon=True).start()
    yield


app = FastAPI(title="حدیث‌یاب", version=VERSION, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(WEB, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB, "templates"))


def _inline_script_hash():
    """CSP hash of the page's one inline script, read from the template itself."""
    with open(os.path.join(WEB, "templates", "index.html"), encoding="utf-8") as f:
        body = re.search(r"<script>(.*?)</script>", f.read(), re.S).group(1)
    return base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()


app.add_middleware(Guard, theme_hash=_inline_script_hash())

Q = Query(..., min_length=1, max_length=400, description="عبارت جستجو")
Speaker = Optional[Literal[tuple(SPEAKER_ORDER + list(BOOKS))]]


def _doc(doc_id):
    index = get_index()
    if not 0 <= doc_id < len(index):
        raise HTTPException(404, "چنین حدیثی نیست")
    return index


@app.exception_handler(HTTPException)
async def http_error(_request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def bad_input(_request, _exc):
    # Plain message only: never echo the submitted input back.
    return JSONResponse({"error": "ورودی نامعتبر است."}, status_code=422)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "texts": len(get_index()), "version": VERSION}


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"version": VERSION, "hadith": None})


@app.get("/h/{doc_id}", response_class=HTMLResponse)
def hadith_page(request: Request, doc_id: int):
    card = _doc(doc_id).card(doc_id)
    return templates.TemplateResponse(request, "index.html", {"version": VERSION, "hadith": card})


@app.get("/api/meta")
def meta():
    index = get_index()
    return {"total": len(index),
            "books": [{"key": k, "label": v, "count": index.book_counts.get(k, 0)} for k, v in BOOKS.items()],
            "speakers": [
        {"key": k, "label": SPEAKER_LABELS[k], "count": index.counts.get(k, 0)} for k in SPEAKER_ORDER]}


@app.get("/api/search")
def search(q: str = Q, mode: Literal["semantic", "keyword"] = "semantic",
           speaker: Speaker = None, limit: int = Query(20, ge=1, le=60)):
    index = get_index()
    t0 = time.perf_counter()
    detected, fallback = None, False
    if mode == "keyword":
        results, total = index.keyword(q, speaker, limit)
    else:
        # "احادیث امام علی درباره مرگ" searches "مرگ" among Imam Ali's hadiths.
        if speaker is None:
            detected, q = detect_speaker(q)
            speaker = detected
        try:
            results, total = index.semantic(q, speaker, limit), None
        except Exception:
            # The embedding API is down or out of quota: rank by words instead.
            log.exception("semantic search failed, falling back to BM25")
            results, total, fallback = index.lexical(q, speaker, limit), None, True
    return {"results": results, "total": total, "speaker": detected, "fallback": fallback,
            "query": q, "ms": round((time.perf_counter() - t0) * 1000)}


@lru_cache(maxsize=1024)
def _rerank_ids(q, ids):
    index = get_index()
    return tuple(c["id"] for c in rerank(q, [index.card(i) for i in ids]))


@app.post("/api/rerank")
def rerank_results(q: str = Body(..., max_length=400), ids: List[int] = Body(..., max_length=40)):
    """The same hits, reordered by an LLM that reads them against the query."""
    index = get_index()
    ids = tuple(i for i in ids if 0 <= i < len(index))
    t0 = time.perf_counter()
    order = _rerank_ids(q.strip(), ids)
    return {"ids": list(order), "ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/api/similar/{doc_id}")
def similar(doc_id: int, speaker: Speaker = None, limit: int = Query(8, ge=1, le=30)):
    return {"results": _doc(doc_id).similar(doc_id, speaker, limit)}


@app.get("/api/hadith/{doc_id}")
def hadith(doc_id: int):
    return _doc(doc_id).card(doc_id)


# Finished smart-search answers, replayed for anyone who asks the same thing.
# In memory only: Render's free disk does not outlive a restart anyway.
_answers = OrderedDict()
_ANSWER_TTL = 7 * 24 * 3600
_ANSWER_MAX = 400


def _sse(event):
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _cached_answer(key):
    hit = _answers.get(key)
    if hit and time.time() - hit[0] < _ANSWER_TTL:
        _answers.move_to_end(key)
        return hit[1]
    _answers.pop(key, None)
    return None


@app.get("/api/research")
def research_stream(q: str = Q, speaker: Speaker = None):
    q = " ".join(q.split())
    # The first search uses the question without the speaker's name in it.
    found, seed_query = detect_speaker(q)
    speaker = speaker or found
    key = (q, speaker)

    def events():
        cached = _cached_answer(key)
        if cached:
            for event in cached:
                yield _sse(event)
            return
        log_ = []
        try:
            for event in research(q, speaker, seed_query):
                if event["type"] == "done":
                    event = {**event, "cached": False}
                log_.append(event)
                yield _sse(event)
        except Exception:
            log.exception("research failed")
            yield _sse({"type": "error", "text": "پژوهش ناتمام ماند. دوباره تلاش کنید."})
            return
        last = log_[-1] if log_ else {}
        if last.get("type") == "done" and not last.get("partial"):
            log_[-1] = {**last, "cached": True}
            _answers[key] = (time.time(), log_)
            while len(_answers) > _ANSWER_MAX:
                _answers.popitem(last=False)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

"""Reorder the top semantic hits with a fast LLM.

On our benchmark this lifted EmbeddingGemma from MRR 0.63 to 0.85: the
embedding finds the right hadith somewhere in the top 20-30, and the model,
reading the texts, puts it first. The page shows the embedding order at
once and swaps in this order when it arrives.
"""
import logging
import re

from .llm import complete

log = logging.getLogger("hadithyab")

PROMPT = """پرسش کاربر: «{q}»
این‌ها احادیث نامزدند، هر کدام با شماره:
{cands}
شماره‌ها را از مرتبط‌ترین به کم‌ارتباط‌ترین با پرسش مرتب کن. فقط شماره‌ها را با ویرگول جدا کن، مثل: 4,1,7"""


def rerank(query, cards, keep=300):
    """Return `cards` in the model's order; on any failure, the order given."""
    if len(cards) < 2:
        return cards
    listing = "\n".join(f"{n + 1}. {c['fa'][:keep]} | {c['ar'][:keep // 2]}" for n, c in enumerate(cards))
    try:
        reply = complete(PROMPT.format(q=query, cands=listing), temperature=0, max_tokens=200)
    except Exception:
        log.exception("rerank failed")
        return cards
    order = []
    for m in re.findall(r"\d+", reply):
        n = int(m) - 1
        if 0 <= n < len(cards) and n not in order:
            order.append(n)
    order += [n for n in range(len(cards)) if n not in order]
    return [cards[n] for n in order]

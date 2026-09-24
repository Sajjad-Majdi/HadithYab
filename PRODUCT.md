# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Seminary students (طلبه) and researchers of Shia hadith. They arrive with a topic or a half-remembered phrase and need the exact hadith, its speaker and its source, to cite it in study, writing or teaching. They read Arabic and Persian; the interface is Persian and right-to-left.

## Product Purpose

حدیث‌یاب finds hadiths by meaning. A user writes a natural Persian or Arabic sentence and gets the hadiths that say that thing, even when the words differ. Success is the right hadith in the first few results, fast, with a source the user can cite.

## Positioning

Semantic search over ~38,000 Shia hadiths (Arabic text with Persian translation) from the IslamShia/shia-hadith collection, plus a research mode where an agent searches repeatedly and writes a cited answer. Keyword sites need the exact words; this needs only the meaning.

## Operating Context

Used at a desk while studying or writing, and on phones. Results are copied into notes, papers and messages, so copy-with-source and shareable links matter. Free to use; the footer's price is "one salawat".

## Capabilities and Constraints

- Semantic search (gemini-embedding-2, 768-dim, in-memory index), exact keyword search, filter by the fourteen infallibles, similar hadiths, retellings grouped under one result.
- Research mode: Gemini Flash Lite agent with search tools, answers in Persian with [#id] citations.
- Hosted on Render's free tier (512 MB RAM, sleeps when idle). Must stay fast.
- The tool does not grade chains of narration (سند) and gives no fatwa.

## Brand Commitments

Name: حدیث‌یاب. Footer line «هزینه استفاده: یک صلوات» stays. Owner preferences: no pure white surfaces in light mode, dark mode tinted and not near-black, every label with an explicit colour, logo in the header.

## Evidence on Hand

The corpus itself (data/hadiths.json). A small retrieval benchmark (data/eval_set.json): gemini-embedding-2 put the right hadith in the top 10 for 80 of 80 test queries. No testimonials or usage numbers exist; do not invent them.

## Product Principles

1. The hadith text is the hero; chrome stays out of its way.
2. Every result is citable as it stands: speaker, source, copy.
3. Never fabricate a hadith, source or number.
4. Speed is a feature.

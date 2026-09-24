// حدیث‌یاب front end: search, research stream, and the hadith cards.
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const fa = (n) => String(n).replace(/\d/g, (d) => FA_DIGITS[d]);
const PAGE = 20;

const state = { q: "", mode: "semantic", speaker: "all", limit: PAGE, speakers: [], labels: {} };
let researchSource = null;
let searchToken = 0;

const els = {
  form: $("#search"), q: $("#q"), speaker: $("#speaker"), clear: $("#clear"), status: $("#status"),
  results: $("#results"), more: $("#more"), answer: $("#answer"), steps: $("#steps"), text: $("#answer-text"),
  tpl: $("#hadith-tpl"),
};

// ---- theme ----------------------------------------------------------------

$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* private mode */ }
});

// ---- donate ------------------------------------------------------------------

$("#copy-card").addEventListener("click", async (e) => {
  if (await copyText("6219861967576876")) { e.target.textContent = "کپی شد"; toast("شماره کارت کپی شد. خدا خیرتان دهد"); }
  setTimeout(() => { e.target.textContent = "کپی شماره کارت"; }, 2000);
});

// ---- small helpers ----------------------------------------------------------

function toast(text) {
  let t = $(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.append(t); }
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 3200);
}

function setStatus(text, error = false) {
  els.status.textContent = text || "";
  els.status.classList.toggle("error", !!error);
}

async function getJSON(url) {
  const r = await fetch(url);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || "خطا در ارتباط با سرور");
  return body;
}

function hadithUrl(id) { return `${location.origin}/h/${id}`; }

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall through */ }
  const ta = Object.assign(document.createElement("textarea"), { value: text });
  document.body.append(ta); ta.select();
  const ok = document.execCommand("copy"); ta.remove();
  return ok;
}

// Highlight query words inside a text node tree (keyword mode only).
const FOLD = { "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و" };
const foldChar = (c) => FOLD[c] ?? c;
const HARAKAT = /[ً-ٰٟـ]/;

function highlight(el, terms) {
  const text = el.textContent;
  // Build a folded copy that remembers where each character came from.
  let folded = "", map = [];
  for (let i = 0; i < text.length; i++) {
    if (HARAKAT.test(text[i])) continue;
    folded += foldChar(text[i]); map.push(i);
  }
  const ranges = [];
  for (const t of terms) {
    let from = 0, at;
    while (t && (at = folded.indexOf(t, from)) !== -1) {
      ranges.push([map[at], map[at + t.length - 1] + 1]); from = at + t.length;
    }
  }
  if (!ranges.length) return;
  ranges.sort((a, b) => a[0] - b[0]);
  el.textContent = "";
  let pos = 0;
  for (const [s, e] of ranges) {
    if (s < pos) continue;
    el.append(text.slice(pos, s));
    const m = document.createElement("mark"); m.textContent = text.slice(s, e); el.append(m);
    pos = e;
  }
  el.append(text.slice(pos));
}

function queryTerms(q) {
  const out = [];
  q.split('"').forEach((part, i) => {
    const f = [...part].filter((c) => !HARAKAT.test(c)).map(foldChar).join("").trim();
    if (!f) return;
    if (i % 2) out.push(f); else out.push(...f.split(/\s+/).filter((w) => w.length > 1));
  });
  return out;
}

// ---- hadith card ------------------------------------------------------------

function renderHadith(h, opts = {}) {
  const node = els.tpl.content.firstElementChild.cloneNode(true);
  node.id = opts.nested ? "" : `h-${h.id}`;
  node.dataset.id = h.id;
  $(".who", node).textContent = h.speaker_label || h.from || "";
  const num = $(".num", node);
  num.textContent = `#${fa(h.id)}`;
  num.href = `/h/${h.id}`;
  $(".src", node).textContent = h.source ? `منبع: ${h.source}` : "";
  const ar = $(".ar", node), faEl = $(".fa", node);
  ar.textContent = h.ar || "";
  faEl.textContent = h.fa || "";
  if (!h.ar) ar.hidden = true;
  if (opts.terms?.length) { highlight(ar, opts.terms); highlight(faEl, opts.terms); }

  if ((h.ar || "").length + (h.fa || "").length > 1400 && !opts.full) {
    node.classList.add("clamp");
    const btn = Object.assign(document.createElement("button"), { type: "button", className: "expand", textContent: "ادامه متن" });
    btn.addEventListener("click", () => { node.classList.remove("clamp"); btn.remove(); });
    faEl.after(btn);
  }

  const variants = h.variants || [];
  const vBtn = $('[data-act="variants"]', node);
  if (variants.length) { vBtn.hidden = false; $("span", vBtn).textContent = `${fa(variants.length)} نقل دیگر`; }
  if (opts.nested) $('[data-act="similar"]', node).hidden = true;

  node.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-act]");
    if (btn) act(btn.dataset.act, h, node, btn);
  });
  return node;
}

async function act(kind, h, node, btn) {
  if (kind === "copy") {
    const who = h.speaker_label || h.from || "";
    const text = [h.ar, h.fa, `(${[who, h.source].filter(Boolean).join("، ")})`, hadithUrl(h.id)].filter(Boolean).join("\n");
    if (await copyText(text)) { toast("متن حدیث با منبع کپی شد"); flashDone(btn); }
    return;
  }
  if (kind === "share") {
    const url = hadithUrl(h.id);
    if (navigator.share && matchMedia("(pointer: coarse)").matches) {
      navigator.share({ title: "حدیث‌یاب", text: h.fa.slice(0, 120), url }).catch(() => {});
    } else if (await copyText(url)) { toast("پیوند حدیث کپی شد"); flashDone(btn); }
    return;
  }
  const box = $(".nested", node);
  const open = btn.getAttribute("aria-expanded") === "true";
  node.querySelectorAll(':scope > .acts [aria-expanded]').forEach((b) => b.setAttribute("aria-expanded", "false"));
  if (open) { box.hidden = true; box.textContent = ""; return; }
  btn.setAttribute("aria-expanded", "true");
  box.hidden = false;
  box.innerHTML = '<div class="skel" style="height:90px"></div>';
  try {
    let list, head;
    if (kind === "similar") {
      list = (await getJSON(`/api/similar/${h.id}?limit=6`)).results;
      head = "احادیث هم‌معنا";
    } else {
      list = await Promise.all(h.variants.map((id) => getJSON(`/api/hadith/${id}`)));
      head = "نقل‌های دیگر همین حدیث";
    }
    box.textContent = "";
    box.append(Object.assign(document.createElement("div"), { className: "nested-head", textContent: head }));
    list.forEach((x) => box.append(renderHadith(x, { nested: true })));
  } catch (err) {
    box.textContent = err.message;
  }
}

function flashDone(btn) {
  btn.classList.add("done");
  setTimeout(() => btn.classList.remove("done"), 1200);
}

// ---- speakers ---------------------------------------------------------------

const MENU = $("#speaker-menu"), SPEAKER_BTN = $("#speaker"), SPEAKER_LABEL = $("#speaker-label");
const TICK = '<svg class="tick" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg>';

function option(key, label, count, wide) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "who-opt" + (wide ? " wide" : "");
  b.setAttribute("role", "option");
  b.dataset.key = key;
  b.setAttribute("aria-selected", String(state.speaker === key));
  b.innerHTML = TICK;
  b.append(label, Object.assign(document.createElement("small"), { textContent: fa(count) }));
  return b;
}

function renderSpeakers() {
  const list = state.speakers.filter((sp) => sp.count);
  if (!list.some((sp) => sp.key === state.speaker)) state.speaker = "all";
  const total = list.reduce((a, sp) => a + sp.count, 0);
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.append(option("all", "همه معصومین", total, true));
  grid.append(Object.assign(document.createElement("div"), { className: "group", textContent: "چهارده معصوم" }));
  list.filter((sp) => sp.key !== "other").forEach((sp) => grid.append(option(sp.key, sp.label, sp.count)));
  const other = list.find((sp) => sp.key === "other");
  if (other) grid.append(option("other", "دیگر گویندگان", other.count, true));
  MENU.replaceChildren(grid);
  const current = list.find((sp) => sp.key === state.speaker);
  SPEAKER_LABEL.textContent = current ? current.label : "همه معصومین";
  SPEAKER_BTN.classList.toggle("on", state.speaker !== "all");
}

function chooseSpeaker(key) {
  MENU.hidePopover();
  if (key === state.speaker) return;
  state.speaker = key;
  state.autoSpeaker = false;
  state.limit = PAGE;
  renderSpeakers();
  if (state.q) run(true);
}

MENU.addEventListener("click", (e) => {
  const opt = e.target.closest(".who-opt");
  if (opt) chooseSpeaker(opt.dataset.key);
});

// Anchor the menu under its button; the popover itself lives in the top layer.
MENU.addEventListener("toggle", (e) => {
  const open = e.newState === "open";
  SPEAKER_BTN.setAttribute("aria-expanded", String(open));
  if (!open) return;
  const r = SPEAKER_BTN.getBoundingClientRect();
  const w = MENU.offsetWidth, h = MENU.offsetHeight;
  const below = r.bottom + 6 + h <= innerHeight - 8;
  MENU.style.top = `${below ? r.bottom + 6 : Math.max(8, r.top - 6 - h)}px`;
  MENU.style.left = `${Math.min(Math.max(8, r.right - w), innerWidth - w - 8)}px`;
  (MENU.querySelector('[aria-selected="true"]') || MENU.querySelector(".who-opt"))?.focus();
});

MENU.addEventListener("keydown", (e) => {
  const opts = [...MENU.querySelectorAll(".who-opt")];
  const i = opts.indexOf(document.activeElement);
  const step = { ArrowDown: 2, ArrowUp: -2, ArrowLeft: 1, ArrowRight: -1, Home: -99, End: 99 }[e.key];
  if (step === undefined) return;
  e.preventDefault();
  const one = matchMedia("(max-width: 420px)").matches;
  const move = one && Math.abs(step) === 2 ? step / 2 : step;
  opts[Math.min(opts.length - 1, Math.max(0, (i < 0 ? 0 : i) + move))].focus();
});

// ---- search -----------------------------------------------------------------

function syncUrl(push) {
  const p = new URLSearchParams();
  if (state.q) p.set("q", state.q);
  if (state.mode !== "semantic") p.set("m", state.mode);
  if (state.speaker !== "all") p.set("s", state.speaker);
  const url = p.toString() ? `/?${p}` : "/";
  if (url !== location.pathname + location.search) history[push ? "pushState" : "replaceState"](null, "", url);
}

const PLACEHOLDER = {
  semantic: "مثلاً: پاداش صبر در برابر مصیبت",
  keyword: 'مثلاً: "حسن الخلق" یا طلب العلم',
  research: "مثلاً: آداب طلب علم از نگاه اهل‌بیت چیست؟",
};
function setPlaceholder() { els.q.placeholder = PLACEHOLDER[state.mode]; }

function readUrl() {
  const p = new URLSearchParams(location.search);
  state.q = p.get("q") || "";
  state.mode = ["semantic", "keyword", "research"].includes(p.get("m")) ? p.get("m") : "semantic";
  state.speaker = p.get("s") || "all";
  els.q.value = state.q;
  els.form.querySelector(`input[name=mode][value=${state.mode}]`).checked = true;
  setPlaceholder();
}

function skeleton(n = 3) {
  els.results.textContent = "";
  for (let i = 0; i < n; i++) els.results.append(Object.assign(document.createElement("div"), { className: "skel" }));
}

function setView() {
  const reading = !!$("#initial-hadith") && !state.q;
  document.body.className = state.q ? "results" : reading ? "reading" : "home";
  els.clear.hidden = !els.q.value;
}

function stopResearch() {
  if (researchSource) { researchSource.close(); researchSource = null; }
}

const RERANK_POOL = 30;

async function run(push = false, append = false) {
  stopResearch();
  syncUrl(push);
  setView();
  els.answer.hidden = true;
  document.querySelectorAll(".cited-head").forEach((h) => h.remove());
  if (!state.q) { els.results.textContent = ""; setStatus(""); els.more.hidden = true; document.title = "حدیث‌یاب"; return; }
  document.title = `${state.q} | حدیث‌یاب`;
  if (state.mode === "research") return runResearch();

  const token = ++searchToken;
  const semantic = state.mode === "semantic";
  // The first page of a semantic search fetches a wider pool for the reranker.
  const fetchLimit = semantic && !append ? Math.max(state.limit, RERANK_POOL) : state.limit;
  if (!append) { setStatus("در حال جستجو…"); skeleton(); }
  els.more.hidden = true;
  const p = new URLSearchParams({ q: state.q, mode: state.mode, limit: fetchLimit });
  if (state.speaker !== "all") p.set("speaker", state.speaker);
  try {
    const data = await getJSON(`/api/search?${p}`);
    if (token !== searchToken) return;
    if (data.speaker && state.speaker === "all") {
      state.speaker = data.speaker;
      state.autoSpeaker = true;  // named in the query; the next query starts from "all" again
      renderSpeakers();
      syncUrl(false);
    }
    const terms = state.mode === "keyword" || data.fallback ? queryTerms(data.query || state.q) : [];
    if (!data.results.length) {
      els.results.textContent = "";
      setStatus(state.mode === "keyword"
        ? "هیچ حدیثی همه این واژه‌ها را ندارد. جستجوی «معنایی» را امتحان کنید."
        : "نتیجه‌ای پیدا نشد. عبارت را ساده‌تر بنویسید.");
      return;
    }
    const shown = new Set(append ? [...els.results.children].map((n) => +n.dataset.id) : []);
    if (!append) els.results.textContent = "";
    const fresh = data.results.filter((h) => !shown.has(h.id));
    const visible = append ? fresh : fresh.slice(0, state.limit);
    visible.forEach((h) => els.results.append(renderHadith(h, { terms })));

    const count = data.total != null ? `${fa(data.total)} حدیث یافت شد` : `${fa(els.results.children.length)} نتیجه نزدیک به معنا`;
    const note = data.fallback ? " · جستجوی معنایی در دسترس نبود؛ مرتب‌شده بر اساس واژه‌ها" : "";
    setStatus(`${count} · ${fa(data.ms)} میلی‌ثانیه${note}`);
    const canMore = data.total != null ? data.total > els.results.children.length : data.results.length >= fetchLimit;
    els.more.hidden = !(canMore && state.limit < 60);

    if (semantic && !append && !data.fallback && data.results.length > 2) {
      rerankPool(token, data.query || state.q, data.results, count);
    }
  } catch (err) {
    if (token !== searchToken) return;
    if (!append) els.results.textContent = "";
    setStatus(err.message, true);
  }
}

// The embedding order shows at once; the LLM's reading of the texts
// replaces it when it lands (usually within a second or two).
async function rerankPool(token, query, pool, count) {
  els.results.classList.add("sorting");
  setStatus(`${count} · در حال مرتب‌سازی دقیق…`);
  try {
    const r = await fetch("/api/rerank", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: query, ids: pool.map((h) => h.id) }),
    });
    const data = await r.json();
    if (token !== searchToken || !r.ok) return;
    const byId = new Map(pool.map((h) => [h.id, h]));
    const nodes = new Map([...els.results.children].map((n) => [+n.dataset.id, n]));
    els.results.textContent = "";
    data.ids.slice(0, state.limit).forEach((id) => els.results.append(nodes.get(id) || renderHadith(byId.get(id))));
    setStatus(`${count} · مرتب‌شده با خواندن متن احادیث`);
  } catch (e) {
    if (token === searchToken) setStatus(count);
  } finally {
    if (token === searchToken) els.results.classList.remove("sorting");
  }
}

// ---- research ---------------------------------------------------------------

const ICON = {
  search_hadith: '<svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></svg>',
  bm25_search: '<svg viewBox="0 0 24 24"><path d="M4 19 9 5l5 14M6 14h6M15 11h5M15 15h5M15 19h5"/></svg>',
  keyword_search: '<svg viewBox="0 0 24 24"><path d="M7 7H4v5h3l-2 5M17 7h-3v5h3l-2 5"/></svg>',
  get_hadith: '<svg viewBox="0 0 24 24"><path d="M5 4h10l4 4v12H5z"/><path d="M9 12h6M9 16h6"/></svg>',
  similar_hadith: '<svg viewBox="0 0 24 24"><circle cx="8" cy="12" r="4"/><circle cx="16" cy="12" r="4"/></svg>',
};
const TOOL_LABEL = {
  search_hadith: (a) => ["معنایی", a.query + (a.speaker && a.speaker !== "all" && state.labels[a.speaker] ? ` · ${state.labels[a.speaker]}` : "")],
  bm25_search: (a) => ["واژه‌های کلیدی", a.query],
  keyword_search: (a) => ["عبارت دقیق", a.terms],
  get_hadith: (a) => ["خواندن حدیث", `#${fa(a.id)}`],
  similar_hadith: (a) => ["هم‌معنای", `#${fa(a.id)}`],
};

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// A small, safe markdown subset: headings, lists, bold, citations.
function renderAnswer(md) {
  const lines = escapeHtml(md).split(/\n/);
  let html = "", list = null;
  const inline = (t) => t
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[#\s*(\d+)\]/g, (_, id) => `<a class="cite" href="#h-${id}" data-cite="${id}">${fa(id)}</a>`)
    .replace(/«([؀-ۿ\sً-ٟ،]{4,})»/g, (m, t) => /[ً-ْ]/.test(t) ? `«<span class="arq" lang="ar">${t}</span>»` : m);
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    let m;
    if (!line) { close(); continue; }
    if ((m = line.match(/^#{1,4}\s+(.*)/))) { close(); html += `<h3>${inline(m[1])}</h3>`; continue; }
    if ((m = line.match(/^[-*•]\s+(.*)/))) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; continue; }
    if ((m = line.match(/^[\d۰-۹]+[.)]\s+(.*)/))) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; continue; }
    close(); html += `<p>${inline(line)}</p>`;
  }
  close();
  return html;
}

// "۳ جستجوی معنایی، ۱ جستجوی واژه‌ای" — the trace's one-line summary.
const KIND_NAME = {
  search_hadith: "جستجوی معنایی", bm25_search: "جستجوی واژه‌ای", keyword_search: "جستجوی عبارت",
  get_hadith: "خواندن متن کامل", similar_hadith: "یافتن هم‌معنا",
};
function summarize(kinds) {
  return Object.entries(kinds).map(([k, n]) => `${fa(n)} ${KIND_NAME[k] || k}`).join("، ") || "جستجو";
}

function runResearch() {
  els.answer.hidden = false;
  els.steps.textContent = "";
  els.text.textContent = "";
  els.results.textContent = "";
  els.more.hidden = true;
  setStatus("");
  const trace = $("#trace"), title = $("#trace-title");
  trace.open = true;
  trace.classList.add("live");
  title.textContent = "در حال جستجو…";
  const seen = new Set();
  const rows = new Map();
  let answer = "";
  const kinds = {};
  const head = Object.assign(document.createElement("h2"), { className: "cited-head", textContent: "احادیثی که بررسی شد" });

  const key = `${state.q.split(/\s+/).join(" ")}|${state.speaker}`;
  const log = [];
  const t0 = performance.now();
  let src = null;

  const finish = (text, error, cached) => {
    if (src) src.close();
    researchSource = null;
    trace.classList.remove("live");
    els.text.classList.remove("caret");
    rows.forEach((li) => li.querySelector(".spin")?.remove());
    const when = cached ? "از جستجوهای قبلی" : `${fa(((performance.now() - t0) / 1000).toFixed(1))} ثانیه`;
    title.textContent = `${summarize(kinds)}، ${fa(seen.size)} حدیث خواند · ${when}`;
    if (answer) trace.open = false;
    if (text) setStatus(text, error);
  };

  const handle = (ev, replay) => {
    if (!replay) log.push(ev);
    if (ev.type === "step") {
      kinds[ev.tool] = (kinds[ev.tool] || 0) + 1;
      const [kind, what] = (TOOL_LABEL[ev.tool] || (() => [ev.tool, ""]))(ev.args || {});
      const li = document.createElement("li");
      li.className = "step";
      li.innerHTML = ICON[ev.tool] || ICON.search_hadith;
      const span = document.createElement("span");
      span.className = "what";
      const b = document.createElement("b"); b.textContent = kind;
      span.append(b, " ", what || "");
      li.append(span, Object.assign(document.createElement("span"), { className: "spin" }));
      els.steps.append(li);
      rows.set(ev.id, li);
    } else if (ev.type === "step_done") {
      const li = rows.get(ev.id);
      if (li) {
        li.querySelector(".spin")?.remove();
        const fresh = (ev.ids || []).filter((id) => !seen.has(id)).length;
        const text = !ev.count ? "بی‌نتیجه" : ev.id === 0 ? `${fa(ev.count)} حدیث` : fresh ? `${fa(fresh)} تازه از ${fa(ev.count)}` : "چیز تازه‌ای نبود";
        li.append(Object.assign(document.createElement("span"), { className: "n", textContent: text }));
      }
    } else if (ev.type === "cards") {
      if (!head.isConnected) els.results.before(head);
      for (const h of ev.cards) {
        if (seen.has(h.id)) continue;
        seen.add(h.id);
        els.results.append(renderHadith(h));
      }
    } else if (ev.type === "delta") {
      if (!answer) title.textContent = "نوشتن پاسخ…";
      answer += ev.text;
      els.text.innerHTML = renderAnswer(answer);
      els.text.classList.add("caret");
    } else if (ev.type === "done") {
      finish(ev.partial ? "ارتباط وسط پاسخ قطع شد؛ پاسخ ناتمام است." : "", !!ev.partial, replay || ev.cached);
      if (!replay && !ev.partial) answerCache.put(key, log);
    } else if (ev.type === "error") {
      finish(ev.text, true);
    }
  };

  // A refresh or the back button replays the stored answer: no request at all.
  const stored = answerCache.get(key);
  if (stored) {
    stored.forEach((ev) => handle(ev, true));
    return;
  }
  const p = new URLSearchParams({ q: state.q });
  if (state.speaker !== "all") p.set("speaker", state.speaker);
  src = new EventSource(`/api/research?${p}`);
  researchSource = src;
  src.onmessage = (e) => handle(JSON.parse(e.data), false);
  src.onerror = () => { if (researchSource) finish("ارتباط با سرور قطع شد. دوباره تلاش کنید.", true); };
}

// Finished smart-search answers kept in this browser for a day, newest 15.
const answerCache = {
  NAME: "hadithyab.answers",
  DAY: 24 * 3600 * 1000,
  all() {
    try { return JSON.parse(localStorage.getItem(this.NAME)) || {}; } catch (e) { return {}; }
  },
  get(key) {
    const hit = this.all()[key];
    return hit && Date.now() - hit.t < this.DAY ? hit.events : null;
  },
  put(key, events) {
    const all = this.all();
    all[key] = { t: Date.now(), events };
    const keep = Object.entries(all).sort((a, b) => b[1].t - a[1].t).slice(0, 15);
    try { localStorage.setItem(this.NAME, JSON.stringify(Object.fromEntries(keep))); } catch (e) { /* full or private */ }
  },
};

els.text.addEventListener("click", (e) => {
  const a = e.target.closest("a.cite");
  if (!a) return;
  e.preventDefault();
  const card = document.getElementById(`h-${a.dataset.cite}`);
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1600);
  } else {
    getJSON(`/api/hadith/${a.dataset.cite}`).then((h) => {
      const node = renderHadith(h); els.results.prepend(node);
      node.scrollIntoView({ behavior: "smooth" });
    }).catch((err) => toast(err.message));
  }
});

// ---- wiring -----------------------------------------------------------------

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  state.q = els.q.value.trim();
  state.mode = els.form.querySelector("input[name=mode]:checked").value;
  if (state.autoSpeaker) { state.speaker = "all"; state.autoSpeaker = false; renderSpeakers(); }
  state.limit = PAGE;
  run(true);
});

els.form.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
  state.mode = r.value;
  state.limit = PAGE;
  setPlaceholder();
  if (state.q) run(true);
}));

els.more.addEventListener("click", () => { state.limit = Math.min(60, state.limit + PAGE); run(false, true); });

els.q.addEventListener("input", () => { els.clear.hidden = !els.q.value; });

els.clear.addEventListener("click", () => {
  els.q.value = "";
  state.q = "";
  state.limit = PAGE;
  $("#initial-hadith")?.remove();
  run(true);
  els.q.focus();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== els.q) { e.preventDefault(); els.q.focus(); els.q.select(); }
  if (e.key === "Escape" && document.activeElement === els.q && els.q.value) els.clear.click();
});

window.addEventListener("popstate", () => { readUrl(); renderSpeakers(); run(); });

(async function init() {
  const initial = $("#initial-hadith");
  if (initial) {
    const h = JSON.parse(initial.textContent);
    const node = renderHadith(h, { full: true });
    els.results.append(node);
    $('[data-act="similar"]', node).click();
  } else {
    readUrl();
  }
  try {
    const meta = await getJSON("/api/meta");
    state.speakers = meta.speakers;
    state.labels = Object.fromEntries(meta.speakers.map((x) => [x.key, x.label]));
    renderSpeakers();
  } catch (e) { /* chips are optional */ }
  setView();
  if (state.q) run();
  else if (!initial) els.q.focus({ preventScroll: true });
})();

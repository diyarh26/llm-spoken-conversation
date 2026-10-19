/* Conversation Explorer: renders one topic's 12 generated calls (4 architectures x 3 prompts)
   next to the human call's turn rhythm. Data comes from explorer/build.py. */
"use strict";

const ARCHS = ["C1", "C2", "C3", "C4"];
const PROMPTS = ["P0", "P1", "P2"];
const ENDINGS = {
  single_pass: "written in a single pass",
  mutual_closing: "ended with a mutual goodbye",
  closing: "ended after a goodbye",
  turn_cap: "cut off at the 40-turn cap",
  empty_turn: "a speaker fell silent",
};
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (sel) => document.querySelector(sel);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const svg = (tag, attrs = {}) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};
const words = (text) => text.split(/\s+/).filter(Boolean).length;
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);

const state = { index: null, id: null, prompt: "P1", tab: "C1", cache: new Map() };

/* ---------- theme ---------- */
function initTheme() {
  try {
    const saved = localStorage.getItem("explorer-theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch { /* storage unavailable: follow the OS setting */ }
  $("#theme").addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("explorer-theme", next); } catch { /* ignore */ }
  });
}

/* ---------- routing: #<conversation>/<prompt> ---------- */
function readHash() {
  const [id, prompt] = location.hash.slice(1).split("/");
  const known = state.index.topics.some((t) => String(t.id) === id);
  state.id = known ? Number(id) : state.index.featured;
  state.prompt = PROMPTS.includes(prompt) ? prompt : "P1";
}
function writeHash() {
  history.replaceState(null, "", `#${state.id}/${state.prompt}`);
}

/* ---------- data ---------- */
async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}
async function conversation(id) {
  if (!state.cache.has(id)) state.cache.set(id, getJSON(`data/${id}.json`));
  return state.cache.get(id);
}

/* ---------- controls ---------- */
function buildControls() {
  const select = $("#topic");
  for (const t of state.index.topics) {
    select.append(el("option", { value: t.id, textContent: `${t.topic}  ·  #${t.id}` }));
  }
  select.addEventListener("change", () => { state.id = Number(select.value); render(); });

  const seg = $("#prompt");
  for (const p of PROMPTS) {
    const b = el("button", { type: "button", textContent: p });
    b.setAttribute("role", "radio");
    b.dataset.prompt = p;
    b.addEventListener("click", () => { state.prompt = p; render(); });
    seg.append(b);
  }
  seg.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    state.prompt = PROMPTS[(PROMPTS.indexOf(state.prompt) + step + PROMPTS.length) % PROMPTS.length];
    render().then(() => seg.querySelector('[aria-checked="true"]').focus());
  });

  $("#shuffle").addEventListener("click", () => {
    const others = state.index.topics.filter((t) => t.id !== state.id);
    state.id = others[Math.floor(Math.random() * others.length)].id;
    render();
  });

  const tabs = $("#tabs");
  for (const a of ARCHS) {
    const b = el("button", { type: "button", textContent: `${a} · ${state.index.architectures[a].name}` });
    b.setAttribute("role", "tab");
    b.dataset.arch = a;
    b.style.setProperty("--arch", `var(--${a.toLowerCase()})`);
    b.addEventListener("click", () => selectTab(a));
    tabs.append(b);
  }
}

function selectTab(arch) {
  state.tab = arch;
  for (const b of document.querySelectorAll("#tabs button")) {
    b.setAttribute("aria-selected", String(b.dataset.arch === arch));
  }
  for (const c of document.querySelectorAll(".col")) c.classList.toggle("active", c.dataset.arch === arch);
}

/* ---------- rendering ---------- */
async function render() {
  writeHash();
  $("#topic").value = String(state.id);
  for (const b of document.querySelectorAll("#prompt button")) {
    b.setAttribute("aria-checked", String(b.dataset.prompt === state.prompt));
    b.tabIndex = b.dataset.prompt === state.prompt ? 0 : -1;
  }
  $("#prompt-note").textContent = state.index.prompts[state.prompt];

  const conv = await conversation(state.id);
  document.title = `${conv.topic} · Conversation Explorer`;
  renderContext(conv);
  renderRhythm(conv);
  renderColumns(conv);
  selectTab(state.tab);
}

function renderContext(conv) {
  $("#instruction").textContent = conv.instruction;
  const box = $("#personas");
  box.replaceChildren();
  for (const s of ["A", "B"]) {
    const p = conv.personas[s];
    box.append(el("div", { className: "persona" },
      el("span", { className: "badge", textContent: s }),
      el("div", {},
        el("strong", { textContent: `Speaker ${s}: ${p.who}` }),
        el("span", { textContent: `${p.card}.` }))));
  }
  box.append(el("p", { className: "persona-note", textContent: "Persona cards are shown to the model in P1 and P2 only; P0 uses the bare demographics, as in the original study." }));
}

function rhythmRow({ label, sub, color, rhythm, scale, archKey }) {
  const total = rhythm.reduce((n, r) => n + r[1], 0);
  const turns = rhythm.length;
  const listening = rhythm.filter((r) => r[2]).length;
  const plot = svg("svg", { class: "rsvg", viewBox: `0 0 1000 34`, preserveAspectRatio: "none", role: "img" });
  plot.setAttribute("aria-label", `${label}: ${turns} turns, ${total} words, ${listening} listener turns`);
  plot.append(svg("line", { x1: 0, x2: 1000, y1: 17, y2: 17, stroke: "var(--line-soft)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }));
  let pos = 0;
  rhythm.forEach(([speaker, n, isListener], i) => {
    const x = (pos / scale) * 1000;
    const w = (n / scale) * 1000;
    pos += n;
    const tip = `Turn ${i + 1} · speaker ${speaker} · ${n} word${n === 1 ? "" : "s"}${isListener ? " (listener turn)" : ""}`;
    let mark;
    if (isListener) {
      mark = svg("circle", { cx: x + w / 2, cy: 17, r: 4.5, fill: "var(--ink)", stroke: "var(--card)", "stroke-width": 1.5 });
      mark.setAttribute("vector-effect", "non-scaling-stroke");
    } else {
      const fill = speaker === "A" ? color : `color-mix(in srgb, ${color} var(--mix-b), var(--card))`;
      mark = svg("rect", { x: x + 0.6, y: 5, width: Math.max(w - 1.2, 1.4), height: 24, rx: 2.5, style: `fill:${fill}` });
    }
    mark.dataset.tip = tip;
    if (archKey) {
      mark.dataset.target = `t-${archKey}-${i}`;
      mark.dataset.arch = archKey;
    }
    plot.append(mark);
  });
  const stat = el("div", { className: "rstat" },
    el("b", { textContent: (total / Math.max(turns, 1)).toFixed(0) }), " words/turn · ",
    el("b", { textContent: String(turns) }), " turns");
  if (!archKey) stat.append(el("br"), el("b", { textContent: String(listening) }), " listener turns");
  return el("div", { className: "rrow" },
    el("div", { className: "rlabel" },
      el("span", { className: "swatch", style: `background:${color}` }),
      el("div", {}, label, el("small", { textContent: sub }))),
    plot, stat);
}

function renderRhythm(conv) {
  const rows = [{ label: "Humans", sub: "real call (Switchboard)", color: "var(--human)", rhythm: conv.human.rhythm }];
  for (const a of ARCHS) {
    const c = conv.conditions[`${a}-${state.prompt}`];
    rows.push({
      label: `${a} · ${state.index.architectures[a].name}`,
      sub: state.index.architectures[a].desc,
      color: `var(--${a.toLowerCase()})`,
      rhythm: c.rhythm.map(([n, l], i) => [c.turns[i][0], n, l]),
      archKey: a,
    });
  }
  const scale = Math.max(...rows.map((r) => r.rhythm.reduce((n, t) => n + t[1], 0)));
  $("#rhythm").replaceChildren(...rows.map((r) => rhythmRow({ ...r, scale })));
}

function renderColumns(conv) {
  const cols = ARCHS.map((a) => {
    const c = conv.conditions[`${a}-${state.prompt}`];
    const meta = state.index.architectures[a];
    const wpt = mean(c.turns.map(([, t]) => words(t)));
    const corpus = state.index.summary[`${a}-${state.prompt}`];
    const col = el("article", { className: "col" });
    col.dataset.arch = a;
    col.style.setProperty("--arch", `var(--${a.toLowerCase()})`);
    const stats = el("dl", { className: "stats" },
      el("div", {}, el("dt", { textContent: "turns" }), el("dd", { textContent: String(c.turns.length) })),
      el("div", {}, el("dt", { textContent: "words/turn" }), el("dd", { textContent: wpt.toFixed(0) })));
    col.append(el("header", { className: "col-head" },
      el("h3", {}, el("span", { className: "id", textContent: a }), meta.name),
      el("p", { textContent: meta.desc }),
      stats,
      el("p", { className: "avg", textContent: `Across all 50 calls in ${a}-${state.prompt}: ${corpus.words_per_turn} words/turn` }),
      el("span", { className: "ending", textContent: ENDINGS[c.ended_by] ?? c.ended_by })));
    const list = el("ol", { className: "turns" });
    c.turns.forEach(([speaker, text], i) => {
      const li = el("li", { className: `turn ${speaker === "A" ? "a" : "b"}`, id: `t-${a}-${i}` },
        el("span", { className: "meta", textContent: `${speaker} · ${words(text)} words` }),
        el("p", { textContent: text }));
      list.append(li);
    });
    col.append(list);
    return col;
  });
  $("#columns").replaceChildren(...cols);
}

/* ---------- interaction: tooltip + jump to turn ---------- */
function initInteraction() {
  const tip = $("#tooltip");
  const rhythm = $("#rhythm");
  rhythm.addEventListener("pointermove", (e) => {
    const t = e.target.closest("[data-tip]");
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t.dataset.tip;
    tip.style.left = `${e.clientX}px`;
    tip.style.top = `${e.clientY}px`;
    tip.hidden = false;
  });
  rhythm.addEventListener("pointerleave", () => { tip.hidden = true; });
  rhythm.addEventListener("click", (e) => {
    const t = e.target.closest("[data-target]");
    if (!t) return;
    selectTab(t.dataset.arch);
    const turn = document.getElementById(t.dataset.target);
    if (!turn) return;
    turn.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    turn.classList.remove("flash");
    void turn.offsetWidth; // restart the animation
    turn.classList.add("flash");
  });
}

/* ---------- boot ---------- */
(async function main() {
  initTheme();
  try {
    state.index = await getJSON("data/index.json");
  } catch (err) {
    document.querySelector("main").prepend(el("p", { className: "card", textContent:
      "Could not load the data bundle. Run `python explorer/build.py`, then serve this folder " +
      "over HTTP (for example `python -m http.server --directory explorer`)." }));
    throw err;
  }
  readHash();
  buildControls();
  initInteraction();
  window.addEventListener("hashchange", () => { readHash(); render(); });
  await render();
})();

'use strict';
/*
 * Движок для уроков: JS-копия src/Value.php, Neuron/Layer/MLP (строка в строку по смыслу),
 * отрисовка графа Value в SVG, пошаговый backward, подсветка PHP-кода и маленькие графики.
 * В уроках считаем прямо в браузере, чтобы ползунки отзывались мгновенно.
 */

// ---------------- Value ----------------
class V {
  constructor(data, prev = [], op = '', label = '') {
    this.data = data;
    this.grad = 0;
    this.prev = prev;
    this.op = op;
    this.label = label;
    this._backward = () => {};
    this.id = ++V.count;
  }
  static wrap(x) { return x instanceof V ? x : new V(x, [], '', String(x)); }

  add(other) {
    other = V.wrap(other);
    const out = new V(this.data + other.data, [this, other], '+');
    out._backward = () => {
      V.acc(this, out.grad);
      V.acc(other, out.grad);
    };
    return out;
  }
  mul(other) {
    other = V.wrap(other);
    const out = new V(this.data * other.data, [this, other], '*');
    out._backward = () => {
      V.acc(this, other.data * out.grad);
      V.acc(other, this.data * out.grad);
    };
    return out;
  }
  sub(other) { return this.add(V.wrap(other).mul(-1)); }
  tanh() {
    const t = Math.tanh(this.data);
    const out = new V(t, [this], 'tanh');
    out._backward = () => { V.acc(this, (1 - t * t) * out.grad); };
    return out;
  }

  /** топологический порядок: каждый узел после всех своих родителей */
  topo() {
    const topo = [], seen = new Set();
    const build = (v) => {
      if (seen.has(v)) return;
      seen.add(v);
      for (const c of v.prev) build(c);
      topo.push(v);
    };
    build(this);
    return topo;
  }
  backward() {
    const topo = this.topo();
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}
V.count = 0;
V.accumulate = true; // урок 6: переключатель «+=» / «=», чтобы увидеть, что ломается
V.acc = (node, g) => { node.grad = V.accumulate ? node.grad + g : g; };

// ---------------- сеть (как src/Neuron.php, Layer.php, MLP.php) ----------------
function rng(seed) { // mulberry32 — воспроизводимые «случайные» веса
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
class Neuron {
  constructor(nin, rand, name = '') {
    this.w = [];
    for (let i = 0; i < nin; i++) this.w.push(new V(rand() * 2 - 1, [], '', `${name}w${i + 1}`));
    this.b = new V(0, [], '', `${name}b`);
  }
  call(x) {
    let act = this.b;
    for (let i = 0; i < this.w.length; i++) act = act.add(this.w[i].mul(x[i]));
    return act.tanh();
  }
  parameters() { return [...this.w, this.b]; }
}
class Layer {
  constructor(nin, nout, rand, name = '') {
    this.neurons = [];
    for (let i = 0; i < nout; i++) this.neurons.push(new Neuron(nin, rand, `${name}n${i + 1}.`));
  }
  call(x) { return this.neurons.map((n) => n.call(x)); }
  parameters() { const p = []; for (const n of this.neurons) p.push(...n.parameters()); return p; }
}
class MLP {
  constructor(nin, nouts, rand) {
    this.sizes = [nin, ...nouts];
    this.layers = [];
    for (let i = 0; i < nouts.length; i++) this.layers.push(new Layer(this.sizes[i], this.sizes[i + 1], rand, `L${i + 1}.`));
  }
  call(x) { for (const l of this.layers) x = l.call(x); return x; }
  parameters() { const p = []; for (const l of this.layers) p.push(...l.parameters()); return p; }
}

// ---------------- мелочи ----------------
const $ = (id) => document.getElementById(id);
const f3 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
const f2 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
const fs = (v, d = 3) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

/** Ползунок: slider('a', -5, 5, 2, 0.1, onChange) → HTML уже в контейнере */
function sliders(container, defs, onChange) {
  const el = typeof container === 'string' ? $(container) : container;
  el.classList.add('sliders');
  el.innerHTML = defs.map((d) => `<label class="slider"><b>${d.label ?? d.name}</b>
    <input type="range" min="${d.min}" max="${d.max}" step="${d.step ?? 0.1}" value="${d.value}" data-k="${d.name}">
    <output>${f2(d.value)}</output></label>`).join('');
  const vals = Object.fromEntries(defs.map((d) => [d.name, d.value]));
  el.querySelectorAll('input').forEach((inp) => inp.addEventListener('input', () => {
    vals[inp.dataset.k] = parseFloat(inp.value);
    inp.nextElementSibling.textContent = f2(vals[inp.dataset.k]);
    onChange(vals);
  }));
  return vals;
}

// ---------------- подсказка ----------------
let tipEl;
function tip(ev, html) {
  if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'tip'; document.body.appendChild(tipEl); }
  if (html === null) { tipEl.hidden = true; return; }
  tipEl.innerHTML = html; tipEl.hidden = false;
  tipEl.style.left = Math.min(ev.clientX + 14, innerWidth - tipEl.offsetWidth - 8) + 'px';
  tipEl.style.top = (ev.clientY + 14) + 'px';
}
function bindTips(root, selector, getHtml) {
  root.querySelectorAll(selector).forEach((el) => {
    el.addEventListener('mousemove', (ev) => { const h = getHtml(el); if (h) tip(ev, h); });
    el.addEventListener('mouseleave', (ev) => tip(ev, null));
  });
}

// ---------------- граф Value ----------------
/**
 * Рисует граф как draw_dot у Карпатого: прямоугольник = Value (имя, data, grad),
 * кружок = операция, которая этот Value породила.
 * opts: { active: V, recv: Set<V>, sel: V, showGrad: bool, onClick(v), explain(v) → html, flow: bool }
 */
function drawGraph(svg, root, opts = {}) {
  const nodes = root.topo();
  const CW = 116, BW = 140, BH = 58, RH = 72;
  const col = new Map(), row = new Map();

  // колонка: листья — 0, остальные — на 2 правее самого правого родителя
  for (const v of nodes) col.set(v, v.prev.length ? Math.max(...v.prev.map((c) => col.get(c))) + 2 : 0);
  // листья подтягиваем вправо, поближе к узлу, который их использует
  for (const v of nodes) {
    if (v.prev.length) continue;
    const users = nodes.filter((u) => u.prev.includes(v));
    if (users.length) col.set(v, Math.min(...users.map((u) => col.get(u))) - 2);
  }
  // строка: листья по порядку обхода, узел — посередине между своими входами
  let slot = 0;
  const place = (v) => {
    if (row.has(v)) return row.get(v);
    if (!v.prev.length) { row.set(v, slot++); return row.get(v); }
    const ys = v.prev.map(place);
    row.set(v, ys.reduce((a, b) => a + b, 0) / ys.length);
    return row.get(v);
  };
  place(root);
  // в одной колонке узлы не должны налезать друг на друга: раздвигаем вниз
  const byCol = new Map();
  for (const v of nodes) { const c = col.get(v); if (!byCol.has(c)) byCol.set(c, []); byCol.get(c).push(v); }
  for (const list of byCol.values()) {
    list.sort((a, b) => row.get(a) - row.get(b));
    for (let i = 1; i < list.length; i++) {
      if (row.get(list[i]) < row.get(list[i - 1]) + 1) row.set(list[i], row.get(list[i - 1]) + 1);
    }
  }

  const cx = (v) => col.get(v) * CW + BW / 2 + 10;
  const cy = (v) => row.get(v) * RH + BH / 2 + 12;
  const W = Math.max(...nodes.map((v) => cx(v))) + BW / 2 + 14;
  const H = Math.max(...nodes.map((v) => cy(v))) + BH / 2 + 14;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.classList.add('vg');

  const OPS = { '+': '+', '*': '×', tanh: 'tanh' };
  let edges = '', ops = '', boxes = '';
  for (const v of nodes) {
    const x = cx(v), y = cy(v);
    if (v.prev.length) {
      const ox = x - CW, oy = y; // кружок операции — в колонке слева от узла
      const isActive = opts.active === v;
      ops += `<g class="opc${isActive ? ' active' : ''}"><circle cx="${ox}" cy="${oy}" r="${v.op === 'tanh' ? 22 : 16}"/><text x="${ox}" y="${oy}">${OPS[v.op] ?? v.op}</text></g>`;
      edges += `<path class="edge${isActive && opts.flow !== false ? ' flow' : ''}" marker-end="url(#ar)" d="M${ox + (v.op === 'tanh' ? 22 : 16)},${oy} L${x - BW / 2 - 2},${y}"/>`;
      for (const c of v.prev) {
        const x1 = cx(c) + BW / 2, y1 = cy(c), x2 = ox - (v.op === 'tanh' ? 22 : 16);
        const mx = (x1 + x2) / 2;
        // длинная связь через несколько колонок огибает промежуточные узлы снизу
        const d = x2 - x1 > CW * 3
          ? `M${x1},${y1} C${x1 + 60},${y1 + RH * 0.9} ${x2 - 60},${oy + RH * 0.9} ${x2},${oy}`
          : `M${x1},${y1} C${mx},${y1} ${mx},${oy} ${x2},${oy}`;
        edges += `<path class="edge${isActive && opts.flow !== false ? ' flow' : ''}" d="${d}"/>`;
      }
    }
    const cls = ['vbox', v.prev.length ? '' : 'leaf', opts.active === v ? 'active' : '', opts.recv?.has(v) ? 'recv' : '', opts.sel === v ? 'sel' : ''].join(' ');
    const name = v.label || (v.prev.length ? '' : String(v.data));
    boxes += `<g class="${cls}" data-id="${v.id}">
      <rect class="body" x="${x - BW / 2}" y="${y - BH / 2}" width="${BW}" height="${BH}" rx="8"/>
      <text class="name" x="${x - BW / 2 + 10}" y="${y - 16}">${esc(name)}</text>
      <text x="${x - BW / 2 + 10}" y="${y + 2}"><tspan class="dim">data</tspan> ${f3(v.data)}</text>
      ${opts.showGrad === false ? '' : `<text class="grad" x="${x - BW / 2 + 10}" y="${y + 19}"><tspan class="dim">grad</tspan> ${f3(v.grad)}</text>`}
    </g>`;
  }
  svg.innerHTML = `<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L8,4 L0,8 z" fill="${cssVar('--ink-3')}"/></marker></defs>${edges}${ops}${boxes}`;

  const byId = new Map(nodes.map((v) => [String(v.id), v]));
  svg.querySelectorAll('.vbox').forEach((g) => {
    const v = byId.get(g.dataset.id);
    if (opts.onClick) g.addEventListener('click', () => opts.onClick(v));
    if (opts.explain) {
      g.addEventListener('mousemove', (ev) => tip(ev, opts.explain(v)));
      g.addEventListener('mouseleave', (ev) => tip(ev, null));
    }
  });
}

// ---------------- пошаговый backward ----------------
/**
 * Шаг 0 — только forward, все grad = 0.
 * Шаг 1 — grad корня = 1.
 * Шаг k ≥ 2 — выполнен _backward у (k−1)-го узла в обратном топологическом порядке.
 */
class BackwardStepper {
  constructor(root) {
    this.root = root;
    this.order = root.topo().reverse().filter((v) => v.prev.length);
    this.k = 0;
  }
  get total() { return this.order.length + 1; }
  go(k) {
    this.k = Math.max(0, Math.min(this.total, k));
    for (const v of this.root.topo()) v.grad = 0;
    this.info = null;
    if (this.k >= 1) this.root.grad = 1;
    for (let i = 0; i < this.k - 1; i++) {
      const node = this.order[i];
      const before = node.prev.map((c) => c.grad);
      node._backward();
      if (i === this.k - 2) this.info = { node, before, after: node.prev.map((c) => c.grad) };
    }
    return this;
  }
  /** Человеческое объяснение последнего шага */
  explain(name = (v) => v.label || String(v.data)) {
    if (this.k === 0) return 'Forward посчитан: у каждого узла есть <b>data</b>, все <b>grad = 0</b>. Нажмите «шаг ▶».';
    if (this.k === 1) return `Старт backward: <code>${name(this.root)}.grad = 1</code>. Производная величины по самой себе равна 1.`;
    const { node: out, before, after } = this.info;
    const [a, b] = out.prev;
    const g = f3(out.grad);
    const delta = (i) => (before[i] !== 0 && V.accumulate ? ` (было ${f3(before[i])}, стало ${f3(after[i])}: <b>+=</b> прибавил)` : '');
    if (out.op === '+') {
      return `Узел <code>+</code> → <b>${name(out)}</b> (grad ${g}). У сложения локальная производная 1, поэтому grad проходит к обоим слагаемым как есть:
        <code>${name(a)}.grad += ${g}</code>${delta(0)}, <code>${name(b)}.grad += ${g}</code>${delta(1)}.`;
    }
    if (out.op === '*') {
      return `Узел <code>×</code> → <b>${name(out)}</b> (grad ${g}). Каждый множитель получает grad выхода, умноженный на ДРУГОЙ множитель:
        <code>${name(a)}.grad += ${f3(b.data)}·${g} = ${f3(after[0] - (V.accumulate ? before[0] : 0))}</code>${delta(0)},
        <code>${name(b)}.grad += ${f3(a.data)}·${g} = ${f3(after[1] - (V.accumulate ? before[1] : 0))}</code>${delta(1)}.`;
    }
    if (out.op === 'tanh') {
      return `Узел <code>tanh</code> → <b>${name(out)}</b> = ${f3(out.data)} (grad ${g}). Производная tanh = 1 − tanh²:
        <code>${name(a)}.grad += (1 − ${f3(out.data)}²)·${g} = ${f3(after[0] - before[0])}</code>.`;
    }
    return '';
  }
}

/** Кнопки ◀ шаг ▶ / всё / сброс для BackwardStepper */
function stepperControls(el, onStep) {
  el.classList.add('stepper');
  el.innerHTML = `<button data-a="reset">⟲ сброс</button><button data-a="prev">◀ шаг</button>
    <button data-a="next" class="primary">шаг ▶</button><button data-a="all">весь backward()</button><span class="count"></span>`;
  el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => onStep(b.dataset.a)));
  return { count: el.querySelector('.count') };
}

// ---------------- PHP-код с подсветкой строк ----------------
function renderCode(el, code, hl = []) {
  const lines = code.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
  const inHl = (n) => hl.some(([a, b]) => n >= a && n <= (b ?? a));
  el.classList.add('code');
  el.innerHTML = lines.map((ln, i) => {
    const cut = ln.indexOf('//');
    const codePart = cut === -1 ? ln : ln.slice(0, cut), comment = cut === -1 ? '' : ln.slice(cut);
    let h = esc(codePart).replace(/\b(public|function|return|foreach|for|new|class|as|private|array|float|int|if)\b/g, '<span class="k">$1</span>');
    if (comment) h += `<span class="c">${esc(comment)}</span>`;
    return `<span class="ln${inHl(i + 1) ? ' hl' : ''}" data-n="${i + 1}">${h || ' '}</span>`;
  }).join('');
}

// ---------------- маленький график ----------------
/** Оси и сетка. Возвращает масштабы X, Y и готовый HTML-каркас. */
function axes({ W = 460, H = 260, x: [x0, x1], y: [y0, y1], xTicks = 5, yTicks = 4, xLabel = '', yLabel = '', pad = { l: 44, r: 12, t: 12, b: 34 } }) {
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (y1 - v) / (y1 - y0) * (H - pad.t - pad.b);
  let g = '';
  for (let i = 0; i <= yTicks; i++) {
    const v = y0 + (y1 - y0) * i / yTicks;
    g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${cssVar('--grid')}"/>
      <text x="${pad.l - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="central" font-size="10" fill="${cssVar('--ink-3')}">${+v.toFixed(2)}</text>`;
  }
  for (let i = 0; i <= xTicks; i++) {
    const v = x0 + (x1 - x0) * i / xTicks;
    g += `<text x="${X(v)}" y="${H - pad.b + 14}" text-anchor="middle" font-size="10" fill="${cssVar('--ink-3')}">${+v.toFixed(2)}</text>`;
  }
  if (x0 < 0 && x1 > 0) g += `<line x1="${X(0)}" x2="${X(0)}" y1="${pad.t}" y2="${H - pad.b}" stroke="${cssVar('--ink-3')}" stroke-width="0.8"/>`;
  if (y0 < 0 && y1 > 0) g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${cssVar('--ink-3')}" stroke-width="0.8"/>`;
  g += `<text x="${W - pad.r}" y="${H - 4}" text-anchor="end" font-size="11" fill="${cssVar('--ink-2')}">${xLabel}</text>
    <text x="${pad.l}" y="${pad.t - 2}" font-size="11" fill="${cssVar('--ink-2')}">${yLabel}</text>`;
  return { X, Y, W, H, pad, html: g };
}
function pathOf(points, X, Y) { return points.map(([x, y], i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(y).toFixed(1)).join(''); }

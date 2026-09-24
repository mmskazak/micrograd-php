'use strict';

// ---------- состояние ----------
const S = {
  sizes: [], segments: {}, digits: {}, params: [], history: [], epoch: 0,
  x: [], target: null,          // target = null → «авто»
  fwd: null,                    // ответ ?action=forward для текущего входа
  mode: 'signal',
  sel: { l: 0, n: 0 },          // нейрон под лупой (индекс слоя модели, индекс нейрона)
  paramsView: null,             // веса для показа во время анимации шага
  gradsView: null,              // градиенты для показа во время анимации шага
  busy: false,
};

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (v, d = 3) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(action, body) {
  const res = await fetch('api.php?action=' + action, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const json = await res.json();
  if (json.error) throw new Error(json.error);
  return json;
}

// ---------- цвета ----------
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
/** Расходящаяся шкала: −1 красный … 0 серый … +1 синий */
function diverge(v, zeroVar = '--zero') {
  const t = Math.max(-1, Math.min(1, v));
  return mix(cssVar(zeroVar), cssVar(t >= 0 ? '--pos' : '--neg'), Math.abs(t));
}

// ---------- имена ----------
const lastLayer = () => S.sizes.length - 2;
function neuronName(l, n) {
  return l === lastLayer() ? `выход «${n}»` : `слой ${l + 1}, нейрон ${n + 1}`;
}
function inputName(l, i) {
  return l === 0 ? `x${i + 1}` : `h${l}.${i + 1}`;
}

// ---------- 1. шаблон индекса ----------
function renderTemplate() {
  const svg = $('template');
  const k = 100; // масштаб: шаблон 1×2 → 100×200
  let html = '';
  for (const [num, s] of Object.entries(S.segments)) {
    const [x1, y1] = s.from.map((v) => v * k), [x2, y2] = s.to.map((v) => v * k);
    const on = S.x[num - 1] ? 'on' : '';
    html += `<g data-seg="${num}"><title>${num}: ${s.name}</title>
      <line class="seg-hit" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>
      <line class="seg-line ${on}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/></g>`;
  }
  // номера палочек: смещены от середины наружу, чтобы не лежать на линии
  const off = { 1: [0, -16], 2: [-20, 0], 3: [20, 0], 4: [14, 10], 5: [0, -12], 6: [-20, 0], 7: [20, 0], 8: [14, 10], 9: [0, 16] };
  for (const [num, s] of Object.entries(S.segments)) {
    const mx = (s.from[0] + s.to[0]) / 2 * k + off[num][0], my = (s.from[1] + s.to[1]) / 2 * k + off[num][1];
    html += `<g class="seg-num"><circle cx="${mx}" cy="${my}" r="8"/><text x="${mx}" y="${my}">${num}</text></g>`;
  }
  svg.innerHTML = html;
  svg.querySelectorAll('[data-seg]').forEach((g) => g.addEventListener('click', () => toggleSegment(+g.dataset.seg - 1)));

  $('vec-raw').textContent = '[' + S.x.join(', ') + ']';
  $('vec-enc').textContent = '[' + S.x.map((v) => (v ? '+1' : '−1')).join(', ') + ']';
  const m = S.fwd ? S.fwd.matched : null;
  document.querySelectorAll('#digit-buttons button').forEach((b) => b.classList.toggle('on', +b.dataset.d === m));
}

function toggleSegment(i) {
  if (S.busy) return;
  S.x[i] = S.x[i] ? 0 : 1;
  S.target = null;
  refresh();
}

// ---------- 2. сеть ----------
const NET = { W: 930, top: 44, bottom: 486, left: 60, right: 590, r: 14 };
function nodePos(c, i) {
  const n = S.sizes[c];
  const x = NET.left + c * (NET.right - NET.left) / (S.sizes.length - 1);
  const step = Math.min(44, (NET.bottom - NET.top) / (n - 1));
  const mid = (NET.top + NET.bottom) / 2;
  return [x, mid + (i - (n - 1) / 2) * step];
}

/** Значение на связи вход i → нейрон n слоя l в текущем режиме */
function edgeValue(l, n, i) {
  if (S.mode === 'signal') return S.fwd.neurons[l][n].muls[i].data;
  if (S.mode === 'weights') return (S.paramsView || S.params)[l][n].w[i];
  return (S.gradsView || S.fwd.grads)[l][n].w[i];
}
function nodeValue(c, i) {
  if (c === 0) return S.fwd.encoded[i];
  return S.fwd.neurons[c - 1][i].out.data;
}

const MODE_HINT = {
  signal: 'Сигнал w·x: сколько каждая связь добавляет в сумму нейрона для текущего входа. Это и есть forward.',
  weights: 'Веса w: то, чему сеть научилась. Именно эти числа хранятся в SQLite.',
  grads: 'Градиент ∂loss/∂w для текущего примера и выбранного правильного ответа. Update сдвинет вес против градиента: синий уменьшится, красный вырастет.',
};

function renderNet() {
  const svg = $('net');
  const L = S.sizes.length;
  const labels = ['вход: палочки', ...S.sizes.slice(1, -1).map((_, i) => `скрытый слой ${i + 1}`), 'цифры'];
  let html = '';
  for (let c = 0; c < L; c++) {
    html += `<g class="layer" data-c="${c}">`;
    const [lx] = nodePos(c, 0);
    html += `<text class="col-label" x="${lx}" y="16">${labels[c]} (${S.sizes[c]})</text>`;
    if (c > 0) {
      const l = c - 1;
      let max = 1e-9;
      for (let n = 0; n < S.sizes[c]; n++) for (let i = 0; i < S.sizes[c - 1]; i++) max = Math.max(max, Math.abs(edgeValue(l, n, i)));
      for (let n = 0; n < S.sizes[c]; n++) {
        const [x2, y2] = nodePos(c, n);
        const isSel = S.sel.l === l && S.sel.n === n;
        for (let i = 0; i < S.sizes[c - 1]; i++) {
          const [x1, y1] = nodePos(c - 1, i);
          const t = edgeValue(l, n, i) / max;
          const w = 0.4 + 3.6 * Math.abs(t);
          const op = 0.12 + 0.88 * Math.abs(t);
          const mx = (x1 + x2) / 2;
          html += `<path class="edge${isSel ? ' sel' : ''}" data-l="${l}" data-n="${n}" d="M${x1 + NET.r},${y1} C${mx},${y1} ${mx},${y2} ${x2 - NET.r},${y2}"
            stroke="${diverge(t)}" stroke-width="${w.toFixed(2)}" opacity="${op.toFixed(2)}"/>`;
        }
      }
    }
    for (let i = 0; i < S.sizes[c]; i++) {
      const [x, y] = nodePos(c, i);
      const v = nodeValue(c, i);
      const sel = c > 0 && S.sel.l === c - 1 && S.sel.n === i ? ' sel' : '';
      const dark = Math.abs(v) > 0.55;
      html += `<g class="node${sel}" data-c="${c}" data-i="${i}">
        <circle cx="${x}" cy="${y}" r="${NET.r}" fill="${diverge(v, '--node-zero')}"/>
        <text class="val" x="${x}" y="${y}" style="${dark ? 'fill:#fff' : ''}">${c === 0 ? (v > 0 ? '+1' : '−1') : v.toFixed(2).replace('-', '−')}</text>`;
      if (c === 0) html += `<text x="${x - 30}" y="${y}">x${i + 1}</text>`;
      if (c === L - 1) html += `<text x="${x + 28}" y="${y}" style="font-size:14px;font-weight:700">${i}</text>`;
      html += '</g>';
    }
    html += '</g>';
  }
  html += lossColumn();
  svg.setAttribute('viewBox', `0 0 ${NET.W} 500`);
  svg.innerHTML = html;
  // выбранный нейрон рисуем поверх остальных связей
  svg.querySelectorAll('.edge.sel').forEach((e) => e.parentNode.insertBefore(e, e.parentNode.querySelector('.node')));

  svg.querySelectorAll('.node').forEach((g) => {
    const c = +g.dataset.c, i = +g.dataset.i;
    g.addEventListener('click', () => {
      if (c === 0) return toggleSegment(i);
      S.sel = { l: c - 1, n: i };
      renderNet(); renderNeuron();
    });
    g.addEventListener('mouseenter', (ev) => {
      if (c > 0) {
        svg.classList.add('dim-edges');
        svg.querySelectorAll(`.edge[data-l="${c - 1}"][data-n="${i}"]`).forEach((e) => e.classList.add('sel'));
      }
      const t = c === 0
        ? `Палочка ${i + 1} (${S.segments[i + 1].name}): ${S.x[i] ? 'есть → +1' : 'нет → −1'}. Клик — переключить.`
        : `${neuronName(c - 1, i)}: out = ${fmt(nodeValue(c, i))}, b = ${fmt((S.paramsView || S.params)[c - 1][i].b)}. Клик — под лупу.`;
      showTip(ev, t);
    });
    g.addEventListener('mouseleave', () => {
      svg.classList.remove('dim-edges');
      svg.querySelectorAll('.edge.sel').forEach((e) => {
        const l = +e.dataset.l, n = +e.dataset.n;
        if (!(S.sel.l === l && S.sel.n === n)) e.classList.remove('sel');
      });
      hideTip();
    });
    g.addEventListener('mousemove', moveTip);
  });
  svg.querySelectorAll('.err').forEach((g) => {
    const k = +g.dataset.k, o = S.fwd.outputs[k], t = k === S.fwd.target ? 1 : -1;
    g.addEventListener('mousemove', (ev) => showTip(ev, `Нейрон «${k}»: выход ${fmt(o)}, цель ${t > 0 ? '+1' : '−1'}. (${fmt(o)} − (${t > 0 ? '+1' : '−1'}))² = ${((o - t) ** 2).toFixed(4)}`));
    g.addEventListener('mouseleave', hideTip);
  });
  svg.querySelectorAll('.sum-node').forEach((g) => {
    g.addEventListener('mousemove', (ev) => showTip(ev, `loss этого примера = сумма 10 квадратов ошибок слева = ${S.fwd.loss.toFixed(4)}. При обучении такие суммы считаются для всех 10 цифр и усредняются.`));
    g.addEventListener('mouseleave', hideTip);
  });
  $('mode-hint').textContent = MODE_HINT[S.mode];
}

/**
 * Правая часть схемы: как 10 выходов превращаются в loss.
 * (выход − цель)² у каждого нейрона-цифры → Σ = loss примера.
 * Во время «1 шаг с разбором» ещё и loss эпохи = ¹⁄₁₀ Σ по 10 цифрам.
 */
function lossColumn() {
  const L = S.sizes.length, f = S.fwd;
  const [ox] = nodePos(L - 1, 0);
  const bx = ox + 52, bw = 58;                 // квадраты ошибок
  const sx = bx + bw + 64, sy = (NET.top + NET.bottom) / 2; // узел Σ
  const ink = cssVar('--ink'), ink2 = cssVar('--ink-2'), ink3 = cssVar('--ink-3');
  let h = `<g class="layer loss-col" data-c="${L}">`;
  h += `<text class="col-label" x="${bx + bw / 2}" y="16">ошибка²</text>`;

  f.outputs.forEach((o, k) => {
    const [, y] = nodePos(L - 1, k);
    const t = k === f.target ? 1 : -1, e = (o - t) ** 2;
    const mag = Math.min(1, Math.sqrt(e) / 2);   // |выход − цель| / 2 ∈ [0, 1]
    const goal = k === f.target;
    h += `<g class="err" data-k="${k}">
      <line x1="${ox + 38}" y1="${y}" x2="${bx}" y2="${y}" stroke="${ink3}" stroke-width="1"/>
      <rect x="${bx}" y="${y - 9}" width="${bw}" height="18" rx="4" fill="${cssVar('--node-zero')}" stroke="${goal ? cssVar('--hi') : 'none'}" stroke-width="2"/>
      <rect x="${bx}" y="${y - 9}" width="${(bw * mag).toFixed(1)}" height="18" rx="4" fill="${cssVar('--neg')}" opacity="0.55"/>
      <text x="${bx + bw / 2}" y="${y}" text-anchor="middle" style="font-size:9.5px">${e < 0.001 ? e.toExponential(0) : e.toFixed(3)}</text>
      <path d="M${bx + bw},${y} C${bx + bw + 34},${y} ${sx - 40},${sy} ${sx - 22},${sy}" fill="none" stroke="${ink3}"
        stroke-width="${(0.6 + 3 * mag).toFixed(2)}" opacity="${(0.35 + 0.65 * mag).toFixed(2)}"/></g>`;
  });

  h += `<g class="sum-node">
    <circle cx="${sx}" cy="${sy}" r="22" fill="${cssVar('--surface')}" stroke="${ink}" stroke-width="1.5"/>
    <text x="${sx}" y="${sy - 6}" text-anchor="middle" style="font-size:15px">Σ</text>
    <text x="${sx}" y="${sy + 9}" text-anchor="middle" style="font-size:9.5px">${f.loss.toFixed(3)}</text>
    <text x="${sx}" y="${sy + 36}" text-anchor="middle" style="font-size:10.5px;font-weight:700">loss примера</text>
    <text x="${sx}" y="${sy + 50}" text-anchor="middle" style="font-size:9.5px;fill:${ink2}">цифра ${f.target}</text></g>`;

  if (STEP.active && STEP.i >= 1) {
    // loss эпохи: 10 сумм (по одной на цифру) → среднее
    const per = STEP.r.perDigit, max = Math.max(...per, 1e-9);
    const ex = sx + 100, ey = sy, rowH = 15, top0 = sy - 60 - 10 * rowH;
    h += `<text class="col-label" x="${ex}" y="16">loss эпохи</text>`;
    per.forEach((v, d) => {
      const y = top0 + d * rowH, cur = f.matched === d && f.target === d;
      h += `<text x="${ex - 30}" y="${y}" style="font-size:9.5px;${cur ? 'font-weight:700' : ''}">${d}</text>
        <rect x="${ex - 22}" y="${y - 5}" width="${(46 * v / max).toFixed(1)}" height="10" rx="2" fill="${cssVar('--neg')}" opacity="${cur ? 0.9 : 0.45}"><title>цифра ${d}: Σ (выход − цель)² = ${v.toFixed(4)}</title></rect>
        <line x1="${ex + 28}" y1="${y}" x2="${ex}" y2="${ey - 26}" stroke="${ink3}" stroke-width="0.6" opacity="0.6"/>`;
      if (cur) h += `<path d="M${sx + 16},${sy - 16} C${sx + 30},${y + 20} ${ex - 50},${y} ${ex - 36},${y}" fill="none" stroke="${cssVar('--hi')}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    });
    const back = STEP.i === 3;
    h += `<circle cx="${ex}" cy="${ey}" r="25" fill="${cssVar('--surface')}" stroke="${back ? cssVar('--hi') : ink}" stroke-width="${back ? 3 : 2}"/>
      <text x="${ex}" y="${ey - 6}" text-anchor="middle" style="font-size:10px;font-weight:700">loss</text>
      <text x="${ex}" y="${ey + 8}" text-anchor="middle" style="font-size:9.5px">${STEP.r.log[0].loss.toFixed(4)}</text>
      <text x="${ex}" y="${ey + 38}" text-anchor="middle" style="font-size:10px">= ¹⁄₁₀ · Σ</text>
      <text x="${ex}" y="${ey + 51}" text-anchor="middle" style="font-size:10px">по 10 цифрам</text>` +
      (back ? `<text x="${ex}" y="${ey + 68}" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:${cssVar('--hi')}">grad = 1 → старт</text>` : '');
  }
  return h + '</g>';
}

// ---------- 3. выход ----------
function renderOutputs() {
  const f = S.fwd;
  $('answer').textContent = f.prediction;
  let html = '';
  f.outputs.forEach((v, d) => {
    const left = v >= 0 ? 50 : 50 + v * 50, width = Math.abs(v) * 50;
    html += `<div class="bar-row${d === f.prediction ? ' win' : ''}${d === f.target ? ' goal' : ''}" title="нейрон «${d}»: tanh = ${fmt(v)}${d === f.target ? ' · цель +1' : ' · цель −1'}">
      <span class="d">${d}</span>
      <span class="track"><span class="fill" style="left:${left}%;width:${width}%;background:${cssVar(v >= 0 ? '--pos' : '--neg')}"></span></span>
      <span class="v">${fmt(v, 2)}</span></div>`;
  });
  $('bars').innerHTML = html;
  $('loss-one').textContent = f.loss.toFixed(4);

  const sel = $('target');
  const auto = f.matched !== null ? `авто: ${f.matched} (совпадает с шаблоном)` : `авто: ${f.prediction} (ответ сети)`;
  sel.innerHTML = `<option value="">${auto}</option>` +
    Object.keys(S.digits).map((d) => `<option value="${d}"${S.target === +d ? ' selected' : ''}>${d}</option>`).join('');
}

// ---------- 4. нейрон под лупой: граф Value ----------
function renderNeuron() {
  const { l, n } = S.sel;
  const t = S.fwd.neurons[l][n];
  const k = t.w.length;
  $('neuron-name').textContent = neuronName(l, n);

  const num = (v) => (v < 0 ? `(−${Math.abs(v).toFixed(2)})` : v.toFixed(2));
  const terms = t.w.map((w, i) => `${num(w.data)}·${num(t.x[i].data)}`).join(' + ');
  const pre = t.sums[k - 1].data;
  $('neuron-formula').textContent = `out = tanh(b + Σ wᵢ·xᵢ) = tanh(${t.b.data.toFixed(2).replace('-', '−')} + ${terms}) = tanh(${fmt(pre, 3)}) = ${fmt(t.out.data)}`;

  const rowH = 52, top = 64;
  const C = { leaf: 8, leafW: 210, mul: 262, opW: 150, sum: 462, tanh: 664, tanhW: 160 };
  const cy = (i) => top + i * rowH + 22;
  const H = top + k * rowH + 20, W = C.tanh + C.tanhW + 10;
  const svg = $('graph');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.minWidth = '640px';
  svg.style.width = '100%';

  const esc = (s) => s.replace(/"/g, '&quot;');
  const g = (v) => `∇ ${fmt(v.grad, 4)}`;
  const bar = (x, y, w, grad) => {
    // полоска под узлом: цвет — знак градиента, длина — величина (относительно самого большого в графе)
    const len = Math.min(1, Math.abs(grad) / gMax) * (w - 8);
    return `<rect class="bar" x="${x + 4}" y="${y}" width="${len.toFixed(1)}" height="3" rx="1.5" fill="${cssVar(grad >= 0 ? '--pos' : '--neg')}"/>`;
  };
  const all = [...t.w, ...t.x, ...t.muls, ...t.sums, t.b, t.out];
  const gMax = Math.max(1e-9, ...all.map((v) => Math.abs(v.grad)));

  // leaf: одна строка «метка = data │ ∇ grad»
  const leaf = (x, y, w, label, v, tip) => `<g class="box" data-tip="${esc(tip)}">
      <rect x="${x}" y="${y}" width="${w}" height="21" rx="5"/>
      <text class="lbl" x="${x + 7}" y="${y + 10}">${label}</text>
      <text x="${x + 50}" y="${y + 10}">${fmt(v.data)}</text>
      <text class="g" x="${x + 118}" y="${y + 10}">${g(v)}</text>
      ${bar(x, y + 17, w, v.grad)}</g>`;
  // op: символ операции + две строки data/grad
  const op = (x, y, w, sym, v, tip) => `<g class="box op" data-tip="${esc(tip)}">
      <rect x="${x}" y="${y}" width="${w}" height="38" rx="6"/>
      <text class="opchar" x="${x + 18}" y="${y + 19}">${sym}</text>
      <text x="${x + 38}" y="${y + 12}">${fmt(v.data)}</text>
      <text class="g" x="${x + 38}" y="${y + 27}">${g(v)}</text>
      ${bar(x, y + 34, w, v.grad)}</g>`;
  const link = (x1, y1, x2, y2) => {
    const mx = (x1 + x2) / 2;
    return `<path class="link" marker-end="url(#arr)" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"/>`;
  };

  let links = '', boxes = '';
  // bias — стартовое значение суммы
  const bY = 8;
  boxes += op(C.sum, bY, C.opW, 'b', t.b,
    `b — стартовое значение суммы. ∇b = ∇ первого «+» = ${fmt(t.sums[0].grad, 4)} (у сложения градиент проходит без изменений).`);
  links += `<path class="link" marker-end="url(#arr)" d="M${C.sum + C.opW / 2},${bY + 38} L${C.sum + C.opW / 2},${cy(0) - 21}"/>`;

  for (let i = 0; i < k; i++) {
    const y = cy(i), w = t.w[i], x = t.x[i], m = t.muls[i], s = t.sums[i];
    const xn = inputName(l, i);
    boxes += leaf(C.leaf, y - 23, C.leafW, `w${i + 1}`, w,
      `∇w${i + 1} = x${i + 1} · ∇× = ${fmt(x.data)} · ${fmt(m.grad, 4)} = ${fmt(w.grad, 4)}. Update: w −= lr·∇w.`);
    boxes += leaf(C.leaf, y + 2, C.leafW, xn, x,
      l === 0
        ? `${xn} — вход (палочка ${i + 1}). Его ∇ считается, но менять вход нечего: учатся только веса.`
        : `${xn} — выход нейрона предыдущего слоя. Его ∇ — сумма вкладов от ВСЕХ нейронов этого слоя, которые его читают (grad += …). Отсюда градиент идёт дальше назад.`);
    boxes += op(C.mul, y - 19, C.opW, '×', m,
      `×: ${fmt(w.data)} · ${fmt(x.data)} = ${fmt(m.data)}. ∇× = ∇ своего «+» = ${fmt(s.grad, 4)}. Дальше делится между множителями: ∇w = x·∇×, ∇x = w·∇×.`);
    boxes += op(C.sum, y - 19, C.opW, '+', s,
      i === k - 1
        ? `Итоговая сумма ${fmt(s.data)}. ∇ = (1 − tanh²) · ∇out = (1 − ${t.out.data.toFixed(3)}²) · ${fmt(t.out.grad, 4)} = ${fmt(s.grad, 4)}.`
        : `Накопленная сумма после ${i + 1}-го слагаемого. У «+» градиент проходит к обоим слагаемым без изменений.`);
    links += link(C.leaf + C.leafW, y - 12, C.mul, y - 4);
    links += link(C.leaf + C.leafW, y + 12, C.mul, y + 4);
    links += link(C.mul + C.opW, y, C.sum, y);
    if (i > 0) links += `<path class="link" marker-end="url(#arr)" d="M${C.sum + C.opW / 2},${cy(i - 1) + 19} L${C.sum + C.opW / 2},${y - 21}"/>`;
  }
  const yOut = cy(k - 1);
  boxes += op(C.tanh, yOut - 19, C.tanhW, 'th', t.out,
    `out = tanh(${fmt(pre)}) = ${fmt(t.out.data)}. ∇out пришёл «сверху»: ${l === lastLayer() ? 'прямо из loss: 2·(out − цель)' : 'сумма вкладов от всех нейронов следующего слоя'} = ${fmt(t.out.grad, 4)}.`);
  links += link(C.sum + C.opW, yOut, C.tanh, yOut);

  svg.innerHTML = `<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="${cssVar('--ink-3')}"/></marker></defs>${links}${boxes}
    <text class="lbl" x="${C.tanh}" y="${yOut + 34}">→ ${l === lastLayer() ? 'в loss' : 'в следующий слой'}</text>`;

  svg.querySelectorAll('.box').forEach((b) => {
    b.addEventListener('mouseenter', (ev) => showTip(ev, b.dataset.tip));
    b.addEventListener('mousemove', moveTip);
    b.addEventListener('mouseleave', hideTip);
  });
}

// ---------- 5. обучение: график loss ----------
function renderChart() {
  const svg = $('chart');
  const h = S.history;
  const W = 420, H = 200, P = { l: 46, r: 12, t: 10, b: 24 };
  if (!h.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${cssVar('--ink-3')}" font-size="12">Сеть ещё не обучали: веса случайные</text>`;
    return;
  }
  const losses = h.map((r) => r.loss);
  const lo = Math.floor(Math.log10(Math.min(...losses))), hi = Math.ceil(Math.log10(Math.max(...losses)));
  const top = hi === lo ? hi + 1 : hi;
  const xMax = Math.max(2, h[h.length - 1].epoch);
  const X = (e) => P.l + (e - 1) / (xMax - 1) * (W - P.l - P.r);
  const Y = (v) => P.t + (top - Math.log10(v)) / (top - lo) * (H - P.t - P.b);

  let grid = '', axis = '';
  for (let p = lo; p <= top; p++) {
    grid += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(10 ** p)}" y2="${Y(10 ** p)}"/>`;
    axis += `<text x="${P.l - 6}" y="${Y(10 ** p) + 3}" text-anchor="end">${10 ** p >= 1 ? 10 ** p : (10 ** p).toFixed(-p)}</text>`;
  }
  const ticks = 5;
  for (let j = 0; j <= ticks; j++) {
    const e = Math.round(1 + (xMax - 1) * j / ticks);
    axis += `<text x="${X(e)}" y="${H - 6}" text-anchor="middle">${e}</text>`;
  }
  // прореживаем точки, чтобы путь не был огромным
  const stride = Math.max(1, Math.floor(h.length / 400));
  let d = '';
  for (let j = 0; j < h.length; j += stride) d += (d ? 'L' : 'M') + X(h[j].epoch).toFixed(1) + ',' + Y(h[j].loss).toFixed(1);
  const last = h[h.length - 1];
  d += 'L' + X(last.epoch).toFixed(1) + ',' + Y(last.loss).toFixed(1);

  svg.innerHTML = `<g class="grid">${grid}</g><g class="axis">${axis}</g>
    <path class="series" d="${d}"/>
    <line class="cross" id="cross" y1="${P.t}" y2="${H - P.b}" visibility="hidden"/>
    <circle class="dot" id="cross-dot" r="4.5" visibility="hidden"/>
    <rect id="chart-hit" x="${P.l}" y="0" width="${W - P.l - P.r}" height="${H}" fill="transparent"/>`;

  const tip = $('chart-tip');
  const hit = $('chart-hit');
  hit.addEventListener('mousemove', (ev) => {
    const r = svg.getBoundingClientRect();
    const sx = (ev.clientX - r.left) / r.width * W;
    const e = Math.round(1 + (sx - P.l) / (W - P.l - P.r) * (xMax - 1));
    const row = h[Math.max(0, Math.min(h.length - 1, e - h[0].epoch))];
    const cx = X(row.epoch), cyv = Y(row.loss);
    $('cross').setAttribute('x1', cx); $('cross').setAttribute('x2', cx); $('cross').setAttribute('visibility', 'visible');
    const dot = $('cross-dot'); dot.setAttribute('cx', cx); dot.setAttribute('cy', cyv); dot.setAttribute('visibility', 'visible');
    tip.hidden = false;
    tip.innerHTML = `эпоха ${row.epoch}<br>loss ${row.loss.toFixed(4)}<br>угадано ${Math.round(row.accuracy * 10)}/10`;
    const px = cx / W * r.width, py = cyv / H * r.height;
    tip.style.left = Math.min(px + 12, r.width - 120) + 'px';
    tip.style.top = Math.max(0, py - 60) + 'px';
  });
  hit.addEventListener('mouseleave', () => {
    tip.hidden = true;
    $('cross').setAttribute('visibility', 'hidden'); $('cross-dot').setAttribute('visibility', 'hidden');
  });
}

function renderStats() {
  const last = S.history[S.history.length - 1];
  $('st-epoch').textContent = S.epoch;
  $('st-loss').textContent = last ? last.loss.toFixed(4) : '–';
  $('st-acc').textContent = last ? `${Math.round(last.accuracy * 10)}/10` : '–';
}

// ---------- подсказки ----------
function showTip(ev, text) { const t = $('tip'); t.textContent = text; t.hidden = false; moveTip(ev); }
function moveTip(ev) {
  const t = $('tip');
  const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  t.style.left = x + 'px'; t.style.top = (ev.clientY + 14) + 'px';
}
function hideTip() { $('tip').hidden = true; }

// ---------- обновление ----------
async function refresh() {
  S.fwd = await api('forward', { x: S.x, target: S.target });
  renderTemplate(); renderNet(); renderOutputs(); renderNeuron();
}

function setBusy(b) {
  S.busy = b;
  ['btn-step', 'btn-50', 'btn-200', 'btn-reset'].forEach((id) => { $(id).disabled = b; });
}

function setMode(m) {
  S.mode = m;
  document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  renderNet();
}

async function trainEpochs(total) {
  setBusy(true);
  try {
    const lr = parseFloat($('lr').value) || S.lr;
    for (let done = 0; done < total; done += 50) {
      const r = await api('train', { epochs: Math.min(50, total - done), lr });
      S.history.push(...r.log);
      S.params = r.params; S.epoch = r.epoch;
      renderChart(); renderStats();
    }
    await refresh();
  } finally { setBusy(false); }
}

/**
 * Одна эпоха, разобранная по стадиям прямо на схеме сети.
 * Эпоха считается на сервере сразу (один запрос), а стадии пользователь листает сам:
 * кнопки «◀ назад / дальше ▶» над схемой, стрелки ← → на клавиатуре, клик по стадии, Esc — закончить.
 */
const STEP = { active: false, i: 0, r: null, lr: 0, prevMode: 'signal', token: 0, applied: false };
const PHASE_KEYS = ['forward', 'loss', 'zero', 'backward', 'update'];
const PHASE_NAMES = ['forward', 'loss', 'zero_grad', 'backward', 'update'];

async function trainStepAnimated() {
  setBusy(true);
  try {
    STEP.lr = parseFloat($('lr').value) || S.lr;
    STEP.r = await api('train', { epochs: 1, lr: STEP.lr });
  } catch (e) {
    setBusy(false);
    throw e;
  }
  STEP.active = true;
  STEP.applied = false;
  STEP.prevMode = S.mode;
  goPhase(0);
}

function phaseText(i) {
  const r = STEP.r;
  if (i === 0) return 'Все 10 цифр по очереди проходят через сеть слева направо. На схеме показан путь текущего входа: каждый нейрон считает tanh(b + Σ w·x) и передаёт результат следующему слою. Связи раскрашены по сигналу w·x.';
  if (i === 1) return `Справа на схеме: у каждого из 10 выходов считается (выход − цель)², где цель +1 у правильной цифры и −1 у остальных. Эти 10 квадратов складываются в <b>Σ = loss примера</b>. Так делается для каждой из 10 цифр (полоски справа), и loss эпохи — их среднее: ¹⁄₁₀ · Σ = <b>${r.log[0].loss.toFixed(4)}</b>. Это один узел <code>Value</code> — корень графа.`;
  if (i === 2) return 'Перед backward все grad обнуляются, иначе к ним прибавились бы градиенты прошлой эпохи (backward делает grad += …). Схема в режиме градиентов, и все связи серые: градиентов пока нет.';
  if (i === 3) return '<code>loss.backward()</code>: в корне grad = 1, дальше по графу справа налево. Каждый узел раздаёт свой grad родителям по цепному правилу. Толстые связи — веса, которые сильнее всего влияют на loss (синий: вес надо уменьшить, красный: увеличить).';
  let maxD = 0, where = '';
  r.params.forEach((layer, l) => layer.forEach((nr, n) => nr.w.forEach((w, i2) => {
    const dlt = Math.abs(w - r.before[l][n].w[i2]);
    if (dlt > maxD) { maxD = dlt; where = `${inputName(l, i2)} → ${neuronName(l, n)}`; }
  })));
  return `Каждый вес сдвигается против своего градиента: <code>w −= ${STEP.lr} · ∇w</code>. Сильнее всего изменился вес ${where}: на ${maxD.toFixed(4)}. Схема показывает, как веса переходят из старых в новые. Новые веса уже сохранены в SQLite.`;
}

async function goPhase(i) {
  if (!STEP.active) return;
  STEP.i = Math.max(0, Math.min(PHASE_KEYS.length - 1, i));
  const token = ++STEP.token;               // быстрый клик «дальше» отменяет недоигранную анимацию
  const alive = () => token === STEP.token && STEP.active;
  const r = STEP.r, net = $('net');
  const cols = [...Array(S.sizes.length).keys()];

  // шапка стадий
  document.querySelectorAll('#phases li').forEach((li, k) => {
    li.classList.toggle('on', k === STEP.i);
    li.classList.toggle('done', k < STEP.i);
  });
  const text = phaseText(STEP.i);
  $('phase-text').innerHTML = text;
  const last = STEP.i === PHASE_KEYS.length - 1;
  const banner = $('net-phase');
  banner.hidden = false;
  banner.innerHTML = `<div class="np-head"><b>${STEP.i + 1}/5 · ${PHASE_NAMES[STEP.i]}</b>
      <span class="np-btns">
        <button data-go="prev"${STEP.i === 0 ? ' disabled' : ''}>◀ назад</button>
        <button data-go="next" class="primary">${last ? 'готово ✓' : 'дальше ▶'}</button>
        <button data-go="end" title="Esc">завершить</button>
      </span></div><div>${text}</div>
      <div class="np-keys">клавиши: ← → , Esc — завершить</div>`;
  banner.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => stepNav(b.dataset.go)));

  // состояние схемы для стадии (каждая стадия выставляет всё с нуля, поэтому «назад» работает)
  $('out-card').classList.toggle('flash', STEP.i === 1);
  net.querySelectorAll('.layer').forEach((g) => g.classList.remove('lit'));
  net.classList.remove('phase');
  S.paramsView = STEP.i === 4 ? r.params : r.before;
  S.gradsView = STEP.i === 2 ? r.grads.map((layer) => layer.map((nr) => ({ w: nr.w.map(() => 0), b: 0 }))) : r.grads;
  setMode(STEP.i <= 1 ? 'signal' : STEP.i <= 3 ? 'grads' : 'weights');

  const light = async (order) => {
    net.classList.add('phase');
    for (const c of order) {
      if (!alive()) return;
      $('net').querySelector(`.layer[data-c="${c}"]`)?.classList.add('lit');
      await sleep(450);
    }
    if (alive()) net.classList.remove('phase');
  };

  if (STEP.i === 1) {
    net.classList.add('phase');
    [S.sizes.length - 1, S.sizes.length].forEach((c) => net.querySelector(`.layer[data-c="${c}"]`)?.classList.add('lit'));
  }
  if (STEP.i === 0) await light(cols);
  if (STEP.i === 3) await light([S.sizes.length, ...[...cols].reverse()]);
  if (STEP.i === 4) {
    if (!STEP.applied) {
      STEP.applied = true;
      S.params = r.params; S.epoch = r.epoch; S.history.push(...r.log);
      renderChart(); renderStats();
    }
    const frames = 30;
    for (let f = 1; f <= frames && alive(); f++) {
      const t = f / frames;
      S.paramsView = r.before.map((layer, l) => layer.map((nr, n) => ({
        w: nr.w.map((w, k) => w + (r.params[l][n].w[k] - w) * t),
        b: nr.b + (r.params[l][n].b - nr.b) * t,
      })));
      renderNet();
      await sleep(40);
    }
  }
}

function stepNav(dir) {
  if (!STEP.active) return;
  if (dir === 'prev') goPhase(STEP.i - 1);
  else if (dir === 'next' && STEP.i < PHASE_KEYS.length - 1) goPhase(STEP.i + 1);
  else finishStep();
}

async function finishStep() {
  if (!STEP.active) return;
  const r = STEP.r;
  if (!STEP.applied) { // закончили раньше пятой стадии — эпоха всё равно уже посчитана на сервере
    S.params = r.params; S.epoch = r.epoch; S.history.push(...r.log);
    renderChart(); renderStats();
  }
  STEP.active = false; STEP.token++;
  $('net-phase').hidden = true;
  $('out-card').classList.remove('flash');
  document.querySelectorAll('#phases li').forEach((li) => { li.classList.remove('on'); li.classList.add('done'); });
  $('phase-text').innerHTML = `Эпоха ${r.epoch} готова: loss до шага был ${r.log[0].loss.toFixed(4)}. Нажмите «1 шаг с разбором» ещё раз, чтобы разобрать следующую.`;
  S.paramsView = null; S.gradsView = null;
  const net = $('net');
  net.classList.remove('phase');
  net.querySelectorAll('.layer').forEach((g) => g.classList.remove('lit'));
  S.mode = STEP.prevMode;
  await refresh();
  setMode(STEP.prevMode);
  setBusy(false);
}

document.addEventListener('keydown', (e) => {
  if (!STEP.active || e.target.closest('input, select, textarea')) return;
  if (e.key === 'ArrowRight') { e.preventDefault(); stepNav('next'); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepNav('prev'); }
  if (e.key === 'Escape') finishStep();
});

// ---------- старт ----------
async function init() {
  const st = await api('state');
  Object.assign(S, st);
  S.x = [...st.digits[0]];
  S.sel = { l: st.sizes.length - 2, n: 0 };
  $('arch').textContent = st.sizes.join(' → ');
  $('lr').value = st.lr;

  $('digit-buttons').innerHTML = Object.keys(st.digits).map((d) => `<button data-d="${d}">${d}</button>`).join('');
  document.querySelectorAll('#digit-buttons button').forEach((b) => b.addEventListener('click', () => {
    if (S.busy) return;
    S.x = [...S.digits[b.dataset.d]];
    S.target = null;
    S.sel = { l: S.sizes.length - 2, n: +b.dataset.d };
    refresh();
  }));
  document.querySelectorAll('#mode button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('target').addEventListener('change', (e) => { S.target = e.target.value === '' ? null : +e.target.value; refresh(); });
  $('btn-step').addEventListener('click', trainStepAnimated);
  document.querySelectorAll('#phases li').forEach((li, k) => li.addEventListener('click', () => { if (STEP.active) goPhase(k); }));
  $('btn-50').addEventListener('click', () => trainEpochs(50));
  $('btn-200').addEventListener('click', () => trainEpochs(200));
  $('btn-reset').addEventListener('click', async () => {
    setBusy(true);
    try {
      const r = await api('reset', {});
      S.params = r.params; S.epoch = 0; S.history = [];
      renderChart(); renderStats();
      document.querySelectorAll('#phases li').forEach((li) => li.classList.remove('on', 'done'));
      $('phase-text').textContent = 'Веса снова случайные. Попробуйте «1 шаг с разбором» или «+200 эпох».';
      await refresh();
    } finally { setBusy(false); }
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderNet(); renderOutputs(); renderNeuron(); renderChart(); });

  renderChart(); renderStats();
  await refresh();
}

init().catch((e) => {
  document.querySelector('main').insertAdjacentHTML('afterbegin', `<div class="card" style="color:var(--neg)">Ошибка: ${e.message}</div>`);
});

'use strict';
// Общая навигация: лаборатория + уроки. Подключается на всех страницах.
(function () {
  const inLearn = location.pathname.includes('/learn/');
  const root = inLearn ? '../' : '';
  const pages = [
    ['index.html', 'Лаборатория', ''],
    ['learn/value.html', 'Value', '1'],
    ['learn/neuron.html', 'Нейрон', '2'],
    ['learn/layer.html', 'Слой', '3'],
    ['learn/mlp.html', 'MLP', '4'],
    ['learn/loss.html', 'Loss', '5'],
    ['learn/backward.html', 'Градиенты', '6'],
    ['learn/update.html', 'Обновление', '7'],
    ['learn/parameters.html', 'parameters()', '8'],
  ];
  const file = location.pathname.split('/').pop() || 'index.html';
  const current = (inLearn ? 'learn/' : '') + file;
  const isHere = (href) => href === current;
  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.innerHTML = pages.map(([href, title, n]) =>
    `<a href="${root}${href}"${isHere(href) ? ' class="here" aria-current="page"' : ''}>${n ? `<span class="n">${n}</span>` : ''}${title}</a>`
  ).join('');
  document.body.prepend(nav);

  // «назад / дальше» внизу урока
  const idx = pages.findIndex(([href]) => isHere(href));
  const slot = document.getElementById('pager');
  if (slot && idx > 0) {
    const prev = pages[idx - 1], next = pages[idx + 1];
    slot.innerHTML =
      `<a href="${root}${prev[0]}">← ${prev[2] ? prev[2] + '. ' : ''}${prev[1]}</a>` +
      (next ? `<a class="next" href="${root}${next[0]}">${next[2]}. ${next[1]} →</a>` : `<a class="next" href="${root}index.html">В лабораторию →</a>`);
  }
})();

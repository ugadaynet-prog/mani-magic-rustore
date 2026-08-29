'use strict';
// Определение нативного плагина NailSegmentation для Capacitor.
// Без этого файла window.Capacitor.Plugins.NailSegmentation будет undefined.
// Файл загружается ДО tryon.js в index.html.

(() => {
  if (!window.Capacitor || !window.Capacitor.Plugins) return;

  const { registerPlugin } = window.Capacitor;
  if (typeof registerPlugin !== 'function') return;

  window.Capacitor.Plugins.NailSegmentation = registerPlugin('NailSegmentation', {
    web: {
      segment: () => Promise.reject(new Error('Нативное распознавание недоступно в браузере')),
    },
  });
})();

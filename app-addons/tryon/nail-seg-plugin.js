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

  // Сохранение примерки в галерею. Плагин в приложении есть давно
  // (TryOnMediaPlugin.kt, MediaStore), но со стороны страницы его никто не
  // объявлял, и кнопка «Сохранить» падала на браузерное скачивание — файл
  // уезжал в каталог загрузок WebView, где владелец его не нашёл.
  window.Capacitor.Plugins.TryOnMedia = registerPlugin('TryOnMedia', {
    web: {
      saveImage: () => Promise.reject(new Error('Галерея доступна только в приложении')),
    },
  });
})();

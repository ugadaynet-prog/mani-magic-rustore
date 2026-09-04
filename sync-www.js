// Копирует веб-приложение (../app) в www/ для сборки Capacitor. Источник правды —
// ../app: сюда ничего руками не редактируем, только синхронизируем перед сборкой.
// Запуск: node sync-www.js

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'app');
const DEST = path.join(__dirname, 'www');
// Нативные экраны и тяжёлые локальные ресурсы живут в этом репозитории, а не
// на сайте. После синхронизации добавляем их поверх общей веб-части.
const ADDONS = path.join(__dirname, 'app-addons');

// Локальные dev-файлы, ненужные внутри упакованного приложения.
// .git — важно: app/ сам по себе отдельный git-репозиторий (боевой сайт на GitHub
// Pages), его историю никак нельзя утащить внутрь Android-сборки.
const EXCLUDE = new Set(['Open-MANI-Magic-RU.bat', 'server.js', 'server.log', 'sw.js', '.git', '.gitignore']);

fs.rmSync(DEST, { recursive: true, force: true });
fs.mkdirSync(DEST, { recursive: true });

function copyDir(src, dest) {
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (EXCLUDE.has(entry.name)) continue;
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      fs.mkdirSync(d, { recursive: true });
      copyDir(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}
copyDir(SRC, DEST);
if (fs.existsSync(ADDONS)) copyDir(ADDONS, DEST);

// ONNX Runtime Web больше не нужен: распознавание ногтей теперь выполняет
// нативный Capacitor-плагин NailSegmentation (Kotlin + onnxruntime-android),
// который загружает модель из android/app/src/main/assets/models/nail-unet.onnx
// и возвращает маску в JavaScript. WebView не трогает ни WASM, ни .mjs.

// Кладём модель туда, где её ждёт плагин. Раньше этого шага не было: файла в
// assets/models не оказывалось ни в репозитории, ни в сборке, `assets.open`
// бросал исключение, и примерка отвечала «Ошибка сегментации» — то есть не
// работала вообще. Источник правды один, app-addons/tryon/nail-unet.onnx:
// оттуда же модель попадает и в www для веб-прототипа.
const MODEL_SRC = path.join(ADDONS, 'tryon', 'nail-unet.onnx');
const MODEL_DEST_DIR = path.join(__dirname, 'android', 'app', 'src', 'main', 'assets', 'models');
if (!fs.existsSync(MODEL_SRC)) {
  throw new Error('Нет модели ' + MODEL_SRC + ' — примерка без неё не заработает');
}
fs.mkdirSync(MODEL_DEST_DIR, { recursive: true });
fs.copyFileSync(MODEL_SRC, path.join(MODEL_DEST_DIR, 'nail-unet.onnx'));
console.log('Модель скопирована в android assets/models: ' +
  (fs.statSync(MODEL_SRC).size / 1e6).toFixed(1) + ' МБ');

// Вход в примерку есть только в Android-сборке: сайт остаётся лёгким и не
// скачивает ONNX Runtime с моделью. Добавляем пункт первым в меню «Ещё».
const indexPath = path.join(DEST, 'index.html');
let index = fs.readFileSync(indexPath, 'utf8');
const moreTitle = '<h3 class="more-title">Ещё</h3>';
if (!index.includes(moreTitle)) throw new Error('Не найдено меню «Ещё» в index.html');
index = index.replace(
  moreTitle,
  moreTitle + '\n\n      <a id="tryOnItem" class="more-item tryon-menu-item" href="tryon/index.html">✦ Примерить цвет на фото</a>'
);
fs.writeFileSync(indexPath, index, 'utf8');

// Нативная сборка всегда указывает на боевой сервер — это не веб-сайт, где
// ?server= может подставить пользователь; тут адрес фиксированный.
const scriptPath = path.join(DEST, 'script.js');
let script = fs.readFileSync(scriptPath, 'utf8');
script = script.replace(
  "const DEFAULT_SERVER_URL = '';",
  "const DEFAULT_SERVER_URL = 'https://api.mani-magic.ru';"
);
fs.writeFileSync(scriptPath, script, 'utf8');

console.log('www/ синхронизирован из ../app + app-addons (примерка включена, DEFAULT_SERVER_URL → api.mani-magic.ru)');

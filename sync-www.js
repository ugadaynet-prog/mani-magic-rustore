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

// ONNX Runtime приходит из npm: в git не храним вторую копию 11-МБ WASM.
// Имена фиксированы версией в package-lock.json.
const ortDist = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
const tryOnDest = path.join(DEST, 'tryon');
// Runtime динамически подгружает также JSEP-модуль. Копируем полный набор,
// иначе Android WebView получает «Failed to fetch dynamically imported module».
for (const name of [
  'ort.min.js',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm'
]) {
  const from = path.join(ortDist, name);
  if (!fs.existsSync(from)) throw new Error(`Не найден ${from}; выполните npm ci`);
  fs.copyFileSync(from, path.join(tryOnDest, name));
}

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

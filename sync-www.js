// Копирует веб-приложение (../app) в www/ для сборки Capacitor. Источник правды —
// ../app: сюда ничего руками не редактируем, только синхронизируем перед сборкой.
// Запуск: node sync-www.js

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'app');
const DEST = path.join(__dirname, 'www');

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

// Нативная сборка всегда указывает на боевой сервер — это не веб-сайт, где
// ?server= может подставить пользователь; тут адрес фиксированный.
const scriptPath = path.join(DEST, 'script.js');
let script = fs.readFileSync(scriptPath, 'utf8');
script = script.replace(
  "const DEFAULT_SERVER_URL = '';",
  "const DEFAULT_SERVER_URL = 'https://api.mani-magic.ru';"
);
fs.writeFileSync(scriptPath, script, 'utf8');

console.log('www/ синхронизирован из ../app (DEFAULT_SERVER_URL → api.mani-magic.ru)');

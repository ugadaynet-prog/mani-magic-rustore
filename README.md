# MANI Magic — нативное приложение для RuStore

Android-обёртка (Capacitor) вокруг веб-приложения `../app`, с нативным биллингом
RuStore Pay SDK — TWA/PWA-обёртки RuStore не пропускает, поэтому здесь настоящий
Android-проект. Собирается в облаке через GitHub Actions — локально Android
Studio/JDK не нужны.

## Статус

Код написан и синтаксически проверен, но **ни разу не собирался** (нет Android SDK
в рабочей среде). Почти наверняка первая сборка в CI покажет 1-2 ошибки компиляции
Kotlin из-за неточных имён в RuStore Pay SDK (см. `ПРОВЕРИТЬ` в коде — сборка их
покажет сразу и однозначно, это не скрытый риск).

## Что где

- `www/` — веб-приложение, копия `../app` (генерируется `node sync-www.js`, в git
  не попадает — не редактировать руками, менять только `../app`).
- `android/` — нативный проект. Ключевые файлы:
  - `app/src/main/java/app/manimagic/rustore/RuStoreBillingPlugin.kt` — мост
    JS ↔ RuStore Pay SDK (методы `purchaseProduct`, `getProducts`, `getPurchases`).
  - `app/src/main/java/app/manimagic/rustore/MainActivity.kt` — регистрирует плагин,
    обрабатывает возврат из банковского приложения при оплате.
  - `app/src/main/AndroidManifest.xml` — `console_app_id_value` (id из консоли
    RuStore, сейчас плейсхолдер) и схема для колбэка оплаты.
- `.github/workflows/generate-keystore.yml` — запустить один раз, создаёт ключ
  подписи приложения.
- `.github/workflows/build-rustore.yml` — собирает подписанный `.aab` при каждом
  пуше (или вручную).
- Сервер (`../server/src/rustore.js`, `../server/src/routes/rustore.js`) — проверяет
  покупку через Public API RuStore перед выдачей подписки (клиенту не верим на слово,
  тот же принцип, что и с вебхуком ЮKassa).

## Что нужно сделать (по шагам)

### 1. Завести GitHub-репозиторий
Нужен репозиторий, чтобы заработали GitHub Actions (сборка идёт у них в облаке).
Можно новый — под тем же аккаунтом, что и остальной проект.

### 2. Один раз — сгенерировать ключ подписи
В репозитории: **Actions → Generate signing keystore → Run workflow**. Через
минуту скачать артефакт `rustore-signing-keystore` — внутри `.keystore`,
`.keystore.base64` и `password.txt`. **Сохранить всё это НАВСЕГДА** в надёжном
месте (пароль-менеджер/облако) — без этого файла нельзя будет выпускать
обновления приложения, только новое (для пользователей — как переустановка).

### 3. Добавить секреты репозитория
**Settings → Secrets and variables → Actions → New repository secret:**

| Имя секрета | Значение |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | содержимое файла `.keystore.base64` из шага 2 |
| `ANDROID_KEYSTORE_PASSWORD` | пароль из `password.txt` |
| `ANDROID_KEY_PASSWORD` | тот же пароль (что и storePassword) |
| `ANDROID_KEY_ALIAS` | `manimagic` |

### 4. Зарегистрировать приложение в консоли RuStore
[RuStore Console](https://console.rustore.ru) → добавить приложение →
package name **`app.manimagic.rustore`**. Там же:
- Настроить **Монетизацию** → добавить товары с id, **точно совпадающими** с
  тарифами: `full_month`, `full_year`, `pro_month`, `pro_year` (цены — как на
  сайте: 199/990/399/2490 ₽).
- Получить **Console App ID** — вписать в
  `android/app/src/main/res/values/strings.xml` вместо
  `REPLACE_WITH_RUSTORE_CONSOLE_APP_ID`.
- В разделе API-доступа — получить `serviceToken`, передать мне (я впишу в
  `.env` на сервере как `RUSTORE_SERVICE_TOKEN` + `RUSTORE_CONSOLE_APP_ID`).
- **Монетизация → Серверные уведомления** — указать URL нашего сервера,
  сохранить AES-ключ (показывается один раз) — тоже передать мне.

### 5. Запустить сборку
**Actions → Build signed AAB for RuStore → Run workflow** (или просто запушить
изменения в ветку `main`). Через несколько минут скачать артефакт
`mani-magic-rustore-aab` — это файл для загрузки в консоль RuStore.

Если сборка упадёт на Kotlin-ошибке — это, скорее всего, одно из мест с пометкой
«ПРОВЕРИТЬ» в коде (точное имя поля/метода RuStore Pay SDK). Пришлите текст
ошибки — поправим конкретную строку по официальному примеру RuStore
(`rustore-dev/rustore-example-java-billing`).

### 6. Загрузить в RuStore и пройти модерацию
В консоли RuStore → загрузить `.aab` → отправить на модерацию. По опыту похожих
приложений — будьте готовы к 2-3 попыткам (модератор может попросить убрать
следы «браузерности» — стандартный процесс, не повод для паники).

### 7. Проверить на реальном устройстве
Здесь нужен ваш телефон — установить тестовую сборку и пройти покупку целиком
(RuStore поддерживает тестовые покупки без реальных денег до публикации).

## Локальная разработка (без сборки)

```bash
node sync-www.js      # обновить www/ из ../app
npx cap sync android   # синхронизировать с нативным проектом
```

Собрать `.aab`/`.apk` локально без Android Studio нельзя — для этого и
существует `build-rustore.yml`.

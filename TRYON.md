# Примерка маникюра

Экран `app-addons/tryon/` доступен в Android-приложении через **Ещё → Примерить цвет на фото**.

## Что делает

- выбирает фото из камеры или галереи;
- запускает `nail-unet.onnx` локально через ONNX Runtime Web;
- строит маску ногтей без отправки фото на сервер;
- меняет цвет с сохранением светотени;
- регулирует интенсивность и порог маски;
- показывает исходник по удержанию кнопки;
- сохраняет JPEG в `Pictures/MANI Magic` через `TryOnMediaPlugin`;
- отправляет JPEG через системное меню Android.

## Сборка

```bash
npm ci
node sync-www.js
npx cap sync android
cd android
./gradlew assembleDebug
```

Либо запустить GitHub Actions workflow **Build try-on test APK**. Артефакт называется `mani-magic-tryon-debug-apk`.

## Контроль модели

SHA-256 `app-addons/tryon/nail-unet.onnx`:

`ba41b982e219700ec863e2efeead7886e791d6230e022579b0ca3e4d18797d7a`

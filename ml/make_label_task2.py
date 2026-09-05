"""Второй круг разметки: кадры, на которых модель сомневается.

Первые 40 кадров были экзаменом, эти — обучающие. Значит и отбирать их надо
иначе: не «поровну и представительно», а туда, где разметка научит большему.

Кадр, который модель и так уверенно разбирает, в обучении почти бесполезен —
она уже знает ответ. Полезен тот, где она колеблется: много пикселей с
вероятностью около половины, найдено подозрительно мало областей, или
области не похожи на ногти по размеру. Такие кадры и отбираем.

Сначала, впрочем, проверяем, что на кадре вообще есть кисть. Без этой
проверки отбор по неуверенности вырождается: первый заход притащил логотипы,
портреты лиц, скриншоты приложения и карты таро — модель «сомневается» там
именно потому, что ногтей нет вовсе, а размечать такое бессмысленно. Руку
ищет MediaPipe Hand Landmarker (Apache-2.0); от него нужен один ответ —
есть кисть или нет, точность точек тут ни на что не влияет.

Экзамен при этом остаётся неприкосновенным: всё, что похоже на любой из 40
размеченных кадров ближе HAMMING_GOLD, выбрасывается — иначе обучение
подсмотрело бы ответы. Между собой отобранные тоже разводятся: почти
одинаковые кадры второй раз ничего не добавляют.

    python make_label_task2.py --count 150

Пишет labels2/{photos,task.json}. Размечать тем же инструментом:
    python label_tool.py --work labels2
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

import exam

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.abspath(os.path.join(HERE, '..', '..'))
GOLD_PHOTOS = os.path.join(HERE, 'labels', 'photos')
OUT = os.path.join(HERE, 'labels2')

SOURCES = ['Новые дизайны', 'дизайны', 'Новая папка', 'ВК-49-фото-новые',
           'Ии персонажи']


def hands_model():
    """Путь к hand_landmarker.task, гарантированно без кириллицы.

    Загрузчик MediaPipe написан на C++ и не открывает файлы, в пути к которым
    есть не-ASCII. Проект живёт в «КОЛОДА ПРИЛ», поэтому модель при
    необходимости копируется во временную папку с латинским путём.
    """
    src = os.path.join(HERE, 'hand_landmarker.task')
    if src.isascii():
        return src
    import shutil
    import tempfile
    dst = os.path.join(tempfile.gettempdir(), 'mm-hand_landmarker.task')
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copyfile(src, dst)
    return dst

EXT = {'.jpg', '.jpeg', '.png', '.webp'}
WORK_SIDE = 1024
# Ближе этого к любому эталонному кадру — не берём: экзамен должен остаться
# независимым от обучения.
HAMMING_GOLD = 20
# Ближе этого друг к другу — почти дубликаты, второй ничего не добавит.
HAMMING_SELF = 14
# Какую долю кадра должна занимать кисть. Просто «кисть найдена» оказалось
# мало: в отбор лезли салонные сцены, где рука в углу и размером с ноготь,
# и рекламные коллажи из шести портретов, где детектор цеплялся за жест.
# Размечать там нечего — ногти в несколько пикселей.
HAND_MIN_FRAC = 0.06


def ahash(path_or_img, n=12):
    im = (Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img)
    a = np.asarray(im.convert('L').resize((n, n), Image.BILINEAR), np.float32)
    return (a > a.mean()).ravel()


def make_hand_detector():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hands_model()),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.3))


def hand_fraction(det, im):
    """Какую долю кадра занимает самая крупная кисть; 0 — кисти нет.

    Напрямую detect() тут не годится: MediaPipe обучен на снимках, где рука
    видна целиком, и на крупном плане кисти, обрезанной рамкой, не находит
    ничего — проверено на кадре 01, где рука занимает половину снимка.
    label_deck.detect_hands перебирает поля и повороты и берёт вариант, где
    кистей больше; та же функция используется и в разметке колоды.
    """
    from label_deck import detect_hands
    rgb = np.asarray(im.convert('RGB'))
    hands, to_orig, _, _ = detect_hands(det, rgb)
    if not hands:
        return 0.0
    h, w = rgb.shape[:2]
    best = 0.0
    for lm in hands:
        pts = [to_orig(p.x, p.y) for p in lm]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        best = max(best, (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h))
    return float(best)


def uncertainty(sess, im, size):
    """Насколько модель не уверена в этом кадре — чем больше, тем полезнее.

    Складываем три признака, каждый в долях от единицы:
      край   доля пикселей с вероятностью в середине (0.35..0.65) — модель
             буквально не может решить, ноготь это или нет;
      мало   нашла меньше пяти областей: на снимке руки их обычно пять;
      разнос области сильно разного размера — верный признак, что часть
             ногтя отрезана или прихвачен посторонний предмет.
    """
    import cv2
    x = np.asarray(exam.letterbox(im, size), np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    p = 1 / (1 + np.exp(-sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]))
    edge = float(((p > 0.35) & (p < 0.65)).mean())

    m = exam.unletterbox(p > 0.5, im.width, im.height).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    areas = sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                   if stats[i, cv2.CC_STAT_AREA] >= 60)
    few = max(0, 5 - len(areas)) / 5
    spread = 0.0
    if len(areas) >= 2:
        spread = min(1.0, (areas[-1] / max(1, np.median(areas)) - 1) / 4)
    return 12 * edge + 0.5 * few + 0.3 * spread, len(areas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--model', default=os.path.join('..', 'app-addons', 'tryon',
                                                    'nail-unet.onnx'))
    args = ap.parse_args()

    gold = [ahash(os.path.join(GOLD_PHOTOS, f)) for f in os.listdir(GOLD_PHOTOS)]
    print(f'эталонных кадров, от которых держимся подальше: {len(gold)}')

    cands = []
    for src in SOURCES:
        d = os.path.join(DECK, src)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in EXT:
                    cands.append(os.path.join(root, f))
    print(f'кандидатов в папках: {len(cands)}')

    det = make_hand_detector()
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    size = sess.get_inputs()[0].shape[2]
    if not isinstance(size, int):
        size = 384

    scored, no_hand = [], 0
    for i, p in enumerate(cands):
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        k = WORK_SIDE / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        h = ahash(im)
        if min(int((h != g).sum()) for g in gold) < HAMMING_GOLD:
            continue
        frac = hand_fraction(det, im)
        if frac < HAND_MIN_FRAC:
            no_hand += 1
            continue
        u, parts = uncertainty(sess, im, size)
        scored.append({'path': p, 'hash': h, 'score': u, 'parts': parts,
                       'hand': frac})
        if (i + 1) % 100 == 0:
            print(f'  просмотрено {i + 1}/{len(cands)}', flush=True)

    scored.sort(key=lambda r: -r['score'])
    print(f'без кисти или с мелкой кистью отброшено: {no_hand}')
    print(f'осталось кандидатов: {len(scored)}')

    picked = []
    for r in scored:
        if len(picked) >= args.count:
            break
        if all(int((r['hash'] != q['hash']).sum()) >= HAMMING_SELF for q in picked):
            picked.append(r)
    print(f'отобрано: {len(picked)}')

    os.makedirs(os.path.join(OUT, 'photos'), exist_ok=True)
    for sub in ('masks', 'instances', 'meta'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    task = []
    for i, r in enumerate(picked, 1):
        iid = f'{i:03d}'
        name = f'{iid}.jpg'
        im = Image.open(r['path']).convert('RGB')
        k = WORK_SIDE / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        im.save(os.path.join(OUT, 'photos', name), quality=95, subsampling=0)
        task.append({'id': iid, 'file': name, 'part': 'обучение',
                     'origin': os.path.relpath(r['path'], DECK),
                     'w': im.width, 'h': im.height,
                     'uncertainty': round(float(r['score']), 3),
                     'parts_found': r['parts'],
                     'hand_frac': round(r['hand'], 3)})
    with open(os.path.join(OUT, 'task.json'), 'w', encoding='utf-8') as fh:
        json.dump({'work_side': WORK_SIDE, 'round': 2, 'items': task},
                  fh, ensure_ascii=False, indent=1)

    u = [t['uncertainty'] for t in task]
    p5 = sum(1 for t in task if t['parts_found'] == 5)
    print(f'\nГотово: {len(task)} кадров в labels2/photos')
    print(f'  неуверенность: от {min(u):.2f} до {max(u):.2f}')
    print(f'  кадров, где модель нашла ровно пять областей: {p5}')
    print('\nДальше:  python label_tool.py --work labels2')


if __name__ == '__main__':
    main()

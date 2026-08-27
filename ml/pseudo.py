"""Разметка наших 245 работ: маски ногтей без ручного труда.

Почему это вообще возможно. Обычно разметка — самая дорогая часть, потому что
надо руками обвести каждый ноготь. У нас есть то, чего нет у чужих наборов:
**для каждой работы известен цвет лака**. В `app/data.js` у карты лежит массив
`hexes`, и работа №N сделана в оттенке `hexes[N]` — это проверено по промптам
генерации на всех 245 работах.

То есть задача из «угадай, где ноготь» превращается в «найди известный цвет».
Это совсем другая задача, и решается она надёжно.

Две опоры, каждая закрывает слабость другой:

1. **Цвет.** Ищем пиксели, близкие к цветам карты. Одного этого мало: фон
   бывает того же оттенка (у card-16 фон — красный мех, у card-21 — тёмные
   ягоды).
2. **Область.** MediaPipe даёт точки кисти, вокруг кончиков пальцев строим
   коридор. Одного этого тоже мало — мы уже знаем, что точка «кончик пальца»
   у длинных ногтей приходится на их основание.

Вместе они работают: цвет отсекает кожу внутри коридора, коридор отсекает фон
того же цвета.

Одного цвета оказалось мало и во второй раз: маски выходили ЧИСТЫЕ, но
неполные — один-два ногтя из пяти. Для обучения это хуже мусора, потому что
незакрашенный ноготь учит модель считать ноготь фоном. Зато цветовые пятна —
надёжные ТОЧКИ внутри ногтей, а достроить точку до целого объекта умеет
Segment Anything. Он и делает финальную маску (--sam).

Результат — zip тех же форм, что и открытый набор (images/ и labels/), плюс
коллаж для глазной проверки: доверять такой разметке вслепую нельзя.
"""
import argparse
import io
import json
import math
import os
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))     # рядом лежат app/ и rustore-app/
APP = os.path.join(ROOT, 'app')
NET = 256
CELL = 260

TIPS = [4, 8, 12, 16, 20]
DIPS = [3, 7, 11, 15, 19]


def read_deck():
    """Достаём из data.js пары «файл работы → допустимые цвета карты».

    Разбираем регулярками, а не исполняем как JS: файл — источник правды для
    приложения, и тащить сюда движок ради пяти полей незачем.
    """
    src = io.open(os.path.join(APP, 'data.js'), encoding='utf-8').read()
    cards = re.findall(r'\{\s*front:.*?\n\s*phrase:', src, re.S)
    out = []
    for c in cards:
        hexes = re.findall(r'#[0-9a-fA-F]{6}', c)
        works = re.findall(r'"(assets/works/[^"]+)"', c)
        if not hexes or not works:
            continue
        for i, w in enumerate(works):
            main = hexes[i] if i < len(hexes) else hexes[0]
            out.append({'file': w, 'main': main, 'all': hexes})
    return out


def hex2rgb(h):
    return np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)], dtype=np.float32)


def finger_zone(shape, pts):
    """Коридор вдоль пальцев: где вообще может быть ноготь."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    zone = np.zeros((h, w), dtype=bool)
    for tip_i, dip_i in zip(TIPS, DIPS):
        tip, dip = pts[tip_i], pts[dip_i]
        tx, ty = tip[0] * w, tip[1] * h
        dx, dy = dip[0] * w, dip[1] * h
        ln = math.hypot(tx - dx, ty - dy)
        if ln < 3:
            continue
        ux, uy = (tx - dx) / ln, (ty - dy) / ln
        ax = (xx - tx) * ux + (yy - ty) * uy          # вдоль пальца
        sx = np.abs(-(xx - tx) * uy + (yy - ty) * ux)  # поперёк
        # Первый заход брал коридор втрое длиннее пальца и почти во всю его
        # ширину — и вылезал за руку на фон, который потом и размечался.
        # Ноготь назад за сустав не уходит, вбок за палец тоже.
        zone |= (ax > -0.5 * ln) & (ax < 2.0 * ln) & (sx < 0.55 * ln)
    return zone


def colour_mask(arr, colours, tol):
    """Пиксели, близкие к любому из цветов карты."""
    m = np.zeros(arr.shape[:2], dtype=bool)
    for c in colours:
        d = np.linalg.norm(arr - hex2rgb(c)[None, None, :], axis=2)
        m |= d < tol
    return m


def skin_colour(arr, pts):
    """Цвет кожи этой руки: суставы у основания пальцев и запястье.

    Там лака не бывает никогда, поэтому образец надёжный, а нужен он для
    проверки ниже: ноготь ОКРУЖЁН кожей, пятно на фоне — нет.
    """
    h, w = arr.shape[:2]
    vals = []
    for k in (0, 5, 9, 13, 17):
        x, y = int(pts[k][0] * w), int(pts[k][1] * h)
        y0, y1 = max(0, y - 3), min(h, y + 4)
        x0, x1 = max(0, x - 3), min(w, x + 4)
        if y1 > y0 and x1 > x0:
            vals.append(arr[y0:y1, x0:x1].reshape(-1, 3))
    return np.concatenate(vals).mean(axis=0) if vals else None


def surrounded_by_skin(arr, comp, skin, tol=48.0, need=0.3):
    """Какая доля кольца вокруг пятна похожа на кожу."""
    from scipy import ndimage
    ring = ndimage.binary_dilation(comp, iterations=4) & ~comp
    if not ring.any() or skin is None:
        return False
    d = np.linalg.norm(arr[ring] - skin[None, :], axis=1)
    return float((d < tol).mean()) >= need


def clean(mask, min_share=0.0004, max_share=0.2, fill_min=0.4, aspect_max=5.0,
          arr=None, skin=None):
    """Убираем всё, что не похоже на ноготь по форме.

    Ноготь — компактное выпуклое пятно. Рваные росчерки и полосы фона своего
    прямоугольника не заполняют и отсеиваются здесь же. Разметка компонент —
    через scipy: то же самое на чистом Python считалось бы минутами на кадр.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if not n:
        return mask, 0, 0
    h, w = mask.shape
    keep_ids = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = int((lab[sl] == i).sum())
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        share = area / (h * w)
        fill = area / (bh * bw)
        aspect = max(bh, bw) / max(1, min(bh, bw))
        if not (min_share < share < max_share and fill > fill_min and aspect < aspect_max):
            continue
        # Главная проверка: вокруг ногтя кожа. Куски фона того же цвета —
        # именно на этом и отсеиваются, форма их не выдаёт.
        if arr is not None and not surrounded_by_skin(arr, lab == i, skin):
            continue
        keep_ids.append(i)
    if not keep_ids:
        return np.zeros_like(mask), n, 0
    keep = np.isin(lab, keep_ids)
    # Заполняем дыры: блик на ногте темнее лака и в цветовой порог не попал,
    # но это по-прежнему ноготь.
    keep = ndimage.binary_fill_holes(keep)
    return keep, n, len(keep_ids)


def sam_masks(predictor, im, points, need_iou=0.55):
    """Достраиваем каждую точку до целого ногтя.

    Точка внутри ногтя у нас надёжная — она пришла из совпадения с известным
    цветом лака внутри коридора пальца. Чего нам не хватало, так это границы:
    её SAM и даёт.

    Берём САМУЮ МЕЛКУЮ из трёх гипотез, а не самую уверенную. SAM возвращает
    три уровня вложенности — часть, объект, целое, — и по одной точке он
    уверенно предлагает ПАЛЕЦ: для него это более естественный объект, чем
    ноготь на нём. Ноготь здесь всегда «часть», то есть наименьшая маска.
    """
    import numpy as np
    if not points:
        return None
    predictor.set_image(np.asarray(im))
    out = np.zeros(np.asarray(im).shape[:2], dtype=bool)
    used = 0
    for (px, py) in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        order = sorted(range(len(masks)), key=lambda i: masks[i].sum())
        m = None
        for i in order:
            if float(scores[i]) < need_iou:
                continue
            cand = masks[i]
            # Ноготь не бывает крупной частью кадра: так выглядит палец или
            # рука целиком, а не ноготь.
            if cand.mean() > 0.05:
                continue
            m = cand
            break
        if m is None:
            continue
        out |= m
        used += 1
    return out if used else None


def centroids(mask):
    """Центры пятен — они и станут подсказками для SAM."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if not n:
        return []
    pts = ndimage.center_of_mass(mask, lab, range(1, n + 1))
    return [(int(x), int(y)) for (y, x) in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=45.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'ours.zip'))
    ap.add_argument('--sheet', default=os.path.join(HERE, 'pseudo.png'))
    ap.add_argument('--sam', action='store_true',
                    help='достраивать пятна до целых ногтей через Segment Anything')
    ap.add_argument('--hint-tol', type=float, default=70.0,
                    help='допуск цвета для ТОЧЕК: он может быть щедрее, чем для маски')
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(HERE, 'data', 'hand_landmarker.task')
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
    ))

    predictor = None
    if args.sam:
        import torch
        from transformers import SamModel, SamProcessor

        class HFPredictor:
            """Тонкая обёртка, чтобы вызов совпадал с обычным SAM-предиктором."""

            def __init__(self):
                self.model = SamModel.from_pretrained('facebook/sam-vit-base').eval()
                self.proc = SamProcessor.from_pretrained('facebook/sam-vit-base')
                self.image = None

            def set_image(self, arr):
                self.image = Image.fromarray(arr)

            def predict(self, point_coords, point_labels, multimask_output=True):
                inp = self.proc(self.image, input_points=[[point_coords.tolist()]],
                                return_tensors='pt')
                with torch.no_grad():
                    out = self.model(**inp)
                masks = self.proc.image_processor.post_process_masks(
                    out.pred_masks.cpu(), inp['original_sizes'].cpu(),
                    inp['reshaped_input_sizes'].cpu())[0][0].numpy()
                scores = out.iou_scores.cpu().numpy().reshape(-1)
                return masks, scores, None

        predictor = HFPredictor()
        print('SAM загружен', flush=True)

    deck = read_deck()
    if args.limit:
        deck = deck[:args.limit]
    print('работ в колоде: %d' % len(deck), flush=True)

    zf = zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED)
    cells, stats = [], {'нет файла': 0, 'нет кисти': 0, 'пусто после чистки': 0, 'принято': 0}

    for item in deck:
        path = os.path.join(APP, item['file'])
        if not os.path.exists(path):
            stats['нет файла'] += 1
            continue
        im = Image.open(path).convert('RGB')
        k = min(1.0, 512 / max(im.size))
        im = im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im, dtype=np.uint8))
        res = lm.detect(mp_img)
        if not res.hand_landmarks:
            stats['нет кисти'] += 1
            continue

        zone = np.zeros(arr.shape[:2], dtype=bool)
        for hand in res.hand_landmarks:
            zone |= finger_zone(arr.shape[:2], [(p.x, p.y) for p in hand])

        skin = None
        for hand in res.hand_landmarks:
            skin = skin_colour(arr, [(p.x, p.y) for p in hand])
            break

        mask = colour_mask(arr, item['all'], args.tol) & zone
        mask, total, kept = clean(mask, arr=arr, skin=skin)

        if predictor is not None:
            # Точки берём по ЩЕДРОМУ порогу: для подсказки достаточно попасть
            # внутрь ногтя, а границу всё равно рисует SAM. Проверку «окружено
            # кожей» оставляем — она отсеивает фон, а не уточняет форму.
            hint = colour_mask(arr, item['all'], args.hint_tol) & zone
            hint, _, _ = clean(hint, arr=arr, skin=skin)
            pts = centroids(hint if hint.any() else mask)
            grown = sam_masks(predictor, im, pts)
            if grown is not None:
                mask = grown

        if not mask.any():
            stats['пусто после чистки'] += 1
            continue
        stats['принято'] += 1

        name = item['file'].replace('assets/works/', '').replace('/', '-').rsplit('.', 1)[0]
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=92)
        zf.writestr('images/%s.jpg' % name, buf.getvalue())
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, 'PNG')
        zf.writestr('labels/%s.png' % name, buf.getvalue())

        if len(cells) < 30:
            vis = np.asarray(im).copy()
            vis[mask] = (0.35 * vis[mask] + 0.65 * np.array([255, 40, 120])).astype(np.uint8)
            cells.append((name, Image.fromarray(vis)))

    zf.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    cols = 5
    rows = max(1, math.ceil(len(cells) / cols))
    sheet = Image.new('RGB', (cols * CELL, rows * (CELL + 16)), (24, 20, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(cells):
        img.thumbnail((CELL, CELL), Image.BILINEAR)
        x, y = (i % cols) * CELL, (i // cols) * (CELL + 16)
        sheet.paste(img, (x + (CELL - img.size[0]) // 2, y))
        draw.text((x + 4, y + CELL + 2), name[:30], fill=(230, 226, 236))
    sheet.save(args.sheet)
    print('коллаж: %s' % args.sheet)


if __name__ == '__main__':
    main()

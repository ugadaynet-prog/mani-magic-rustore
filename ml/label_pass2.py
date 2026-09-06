"""Первый проход по второй партии: модель ищет, MediaPipe добирает, SAM обводит.

В первый раз я расставлял точки руками по координатной сетке — на 39 кадрах
это заняло полдня, на 150 не имеет смысла. Здесь подсказки берутся из двух
источников сразу, и это не то же самое, что прежняя автоматическая разметка.

  Дообученная модель находит ногти в 88 случаях из ста и даёт рамку вокруг
  каждого. Рамка — сильная подсказка: SAM внутри неё обводит пластину точнее,
  чем модель, у которой выход в 384 пикселя на весь кадр.

  MediaPipe добирает пропущенные. Ровно те 12%, что модель не видит, — обычно
  крайние пальцы и ногти на тёмном фоне; кисть же там находится. Кончик пальца
  без единой области модели рядом превращается в отдельную подсказку.

Почему это не повторение старой ошибки. Прежняя авторазметка была последним
словом: что она выдала, на том и учились. Здесь она первое слово — каждый
кадр потом смотрит человек, а этот проход лишь избавляет его от возни с
контурами. Оценка качества по-прежнему берётся только с отложенных кадров,
размеченных руками с нуля.

    python label_pass2.py --work labels2
    python label_pass2.py --work labels2 --sheet 1   # лист для проверки глазом
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import exam
import label_pass1 as P
import label_tool as L
import onnxruntime as ort
import synth

# Насколько далеко от кончика пальца может лежать область модели, чтобы
# считаться «этим же ногтем», в долях длины дистальной фаланги. Дальше —
# значит модель этот ноготь пропустила и нужна отдельная подсказка.
TIP_NEAR = 1.2
# Мелочь в предсказании модели — не ноготь, а шум на границе.
MIN_BLOB = 120


def model_boxes(sess, im, size):
    """Рамки вокруг того, что модель считает ногтями."""
    pred = exam.run_model(sess, im, size)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < MIN_BLOB:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Рамку чуть расширяем: модель систематически недокрашивает край,
        # и обрезанная рамка не дала бы SAM дотянуться до кончика.
        pad = 0.12
        out.append(([max(0, x - w * pad), max(0, y - h * pad),
                     min(im.width, x + w * (1 + pad)),
                     min(im.height, y + h * (1 + pad))],
                    (float(cent[i][0]), float(cent[i][1]))))
    return out


def missed_tips(det, im, centers, typical):
    """Рамки на кончиках пальцев, рядом с которыми модель ничего не нашла.

    Возвращаем именно рамки, а не точки. По одной точке на кончике SAM с
    равным правом отдаёт ноготь, фалангу и всю кисть — на пробе так и вышло:
    кадр 009 закрасился целиком. Рамка ограничивает ответ по построению.

    Длину ногтя заранее не знаешь, поэтому label_deck.nail_prompts даёт
    несколько рамок вдоль пальца — от короткого ногтя до сильно наращённого.
    Выбираем ту, чья площадь ближе к типичному ногтю ЭТОГО кадра, известному
    по находкам модели; если модель не нашла ничего, берём вторую по счёту —
    она отвечает обычному ненаращённому ногтю.
    """
    from label_deck import detect_hands, nail_prompts
    rgb = np.asarray(im.convert('RGB'))
    hands, to_orig, _, _ = detect_hands(det, rgb)
    if not hands:
        return []
    out = []
    for lm in hands:
        for pr in nail_prompts(lm, to_orig):
            tx, ty = pr['tip']
            ln = pr['L']
            if any(math.hypot(cx - tx, cy - ty) <= TIP_NEAR * ln
                   for cx, cy in centers):
                continue
            cands = pr['cands']
            if not cands:
                continue
            if typical:
                def area(c):
                    b = c['box']
                    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                box = min(cands, key=lambda c: abs(area(c) - typical))['box']
            else:
                box = cands[min(1, len(cands) - 1)]['box']
            out.append([float(box[0]), float(box[1]),
                        float(box[2]), float(box[3])])
    return out


def paint(im, idx, path):
    rgb = np.asarray(im)
    out = synth.recolor(rgb.astype(np.float32) / 255.0, idx > 0, P.PAINT)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=88)


def sheet(work, page, per=24, cols=6):
    """Контактный лист покраски — чтобы просмотреть пачку разом."""
    items = [t for t in L.task_items()
             if os.path.exists(os.path.join(work, 'painted', f'{t["id"]}.jpg'))]
    items.sort(key=lambda t: t['id'])
    chunk = items[(page - 1) * per: page * per]
    if not chunk:
        raise SystemExit(f'на странице {page} пусто')
    cw, ch, pad, head = 250, 320, 24, 30
    rows = math.ceil(len(chunk) / cols)
    sh = Image.new('RGB', (cols * cw, rows * (ch + pad) + head), '#14151a')
    d = ImageDraw.Draw(sh)
    try:
        f = ImageFont.truetype('segoeui.ttf', 13)
        fb = ImageFont.truetype('seguisb.ttf', 15)
    except Exception:
        f = fb = ImageFont.load_default()
    d.text((10, 8), f'Первый проход, страница {page}: кадры '
                    f'{chunk[0]["id"]}–{chunk[-1]["id"]}', fill='#e8e9ee', font=fb)
    for i, t in enumerate(chunk):
        im = Image.open(os.path.join(work, 'painted', f'{t["id"]}.jpg'))
        im.thumbnail((cw - 12, ch - 12))
        x, y = (i % cols) * cw, head + (i // cols) * (ch + pad)
        sh.paste(im, (x + (cw - im.width) // 2, y + (ch - im.height) // 2))
        with open(os.path.join(work, 'meta', f'{t["id"]}.json'), encoding='utf-8') as fh:
            m = json.load(fh)
        n = m['nails']
        col = '#35d0a5' if n == 5 else ('#f0a23c' if 3 <= n <= 8 else '#ff5470')
        d.text((x + 10, y + ch + 4), f'{t["id"]}   {n} шт', fill=col, font=f)
    out = os.path.join(work, f'проверка-{page}.jpg')
    sh.save(out, quality=87)
    print(out, sh.size, f'кадров {len(chunk)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='labels2')
    ap.add_argument('--model', default=os.path.join('..', 'app-addons', 'tryon',
                                                    'nail-unet.onnx'))
    ap.add_argument('--only', nargs='+', help='размечать только эти кадры')
    ap.add_argument('--sheet', type=int, help='собрать лист проверки, страница N')
    ap.add_argument('--redo', action='store_true',
                    help='перерисовать даже уже размеченные')
    ap.add_argument('--tips', action='store_true',
                    help='убрать из масок всё, что не на кончике пальца')
    args = ap.parse_args()

    L.set_work(args.work)
    work = L.WORK
    os.makedirs(os.path.join(work, 'painted'), exist_ok=True)
    if args.sheet:
        return sheet(work, args.sheet)
    if args.tips:
        import make_label_task2 as M
        return keep_at_tips(work, M.make_hand_detector())

    items = L.task_items()
    if args.only:
        keep = set(args.only)
        items = [t for t in items if t['id'] in keep]

    L.load_sam()
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    size = sess.get_inputs()[0].shape[2]
    if not isinstance(size, int):
        size = 384

    import make_label_task2 as M
    det = M.make_hand_detector()

    for t in items:
        mp_path = os.path.join(work, 'meta', f'{t["id"]}.json')
        if os.path.exists(mp_path) and not args.redo:
            continue
        im = Image.open(os.path.join(work, 'photos', t['file'])).convert('RGB')
        boxes = model_boxes(sess, im, size)
        # Типичная площадь ногтя на этом кадре — по находкам самой модели.
        # Рамка модели чуть шире ногтя, отсюда поправка.
        b_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b, _ in boxes]
        typical = float(np.median(b_areas)) if b_areas else 0.0
        tips = missed_tips(det, im, [c for _, c in boxes], typical)
        prompts = [b for b, _ in boxes] + tips

        frame_px = im.width * im.height
        chosen = P.choose([P.variants(t, px, frame_px) for px in prompts]) \
            if prompts else []

        idx = np.zeros((im.height, im.width), np.uint8)
        k = 0
        for v, _clean in chosen:
            m = v['m']
            if int((m & (idx > 0)).sum()) > 0.5 * m.sum():
                continue
            k += 1
            idx[m & (idx == 0)] = k

        # Область, которая в разы крупнее соседей, — это не ноготь, а палец
        # или клякса, расползшаяся по коже. Ногти одного кадра различаются по
        # площади в разы только из-за ракурса, и порог 2.5 их не задевает.
        areas = {int(v): int((idx == v).sum()) for v in np.unique(idx) if v}
        if len(areas) >= 4:
            med = float(np.median(list(areas.values())))
            for v, a in list(areas.items()):
                if a > 2.5 * med:
                    idx[idx == v] = 0
                    del areas[v]
            # Перенумеровываем подряд, чтобы дырок в номерах не осталось.
            out = np.zeros_like(idx)
            for k2, v in enumerate(sorted(areas), 1):
                out[idx == v] = k2
            idx = out
            areas = {int(v): int((idx == v).sum()) for v in np.unique(idx) if v}

        L.save_labels(t, P.png_of(idx), {
            'pass': 'second-by-claude', 'from_model': len(boxes),
            'from_tips': len(tips),
            'marked_at': datetime.now(timezone.utc).astimezone().isoformat(
                timespec='seconds')})
        paint(im, idx, os.path.join(work, 'painted', f'{t["id"]}.jpg'))
        print(f'{t["id"]}: ногтей {len(areas)} '
              f'(модель {len(boxes)}, добрано по кисти {len(tips)}), '
              f'площади {sorted(areas.values())}', flush=True)




def keep_at_tips(work, det, tip_max=1.1):
    """Убрать из масок всё, что не сидит на кончике пальца.

    Первый проход красил губы, брови и бутылку лака: подсказки приходили от
    модели, а она выучила «гладкий блестящий овал ≈ ноготь». У ногтя есть
    свойство, которого нет ни у губ, ни у стекла, — он растёт из кончика
    пальца. MediaPipe даёт кончики, и всё, что от них дальше tip_max длин
    фаланги, из маски уходит.

    Кадры, где кисть не нашлась, не трогаем: там судить не по чему.
    """
    from label_deck import detect_hands, nail_prompts
    items = L.task_items()
    dropped_total = kept_total = skipped = 0
    report = []
    for t in items:
        ip = os.path.join(work, 'instances', f'{t["id"]}.png')
        mp_ = os.path.join(work, 'meta', f'{t["id"]}.json')
        if not os.path.exists(ip) or not os.path.exists(mp_):
            continue
        with open(mp_, encoding='utf-8') as fh:
            meta = json.load(fh)
        if meta.get('pass') != 'second-by-claude':
            continue                      # чужую разметку не трогаем
        rgb = np.asarray(Image.open(
            os.path.join(work, 'photos', t['file'])).convert('RGB'))
        hands, to_orig, _, _ = detect_hands(det, rgb)
        tips = []
        for lm in (hands or []):
            for pr in nail_prompts(lm, to_orig):
                tips.append((pr['tip'][0], pr['tip'][1], pr['L']))
        if not tips:
            skipped += 1
            continue

        idx = np.asarray(Image.open(ip)).copy()
        out = np.zeros_like(idx)
        k = 0
        dropped = 0
        for v in [int(x) for x in np.unique(idx) if x]:
            ys, xs = np.nonzero(idx == v)
            cx, cy = xs.mean(), ys.mean()
            near = min(math.hypot(cx - tx, cy - ty) / ln for tx, ty, ln in tips)
            if near <= tip_max:
                k += 1
                out[idx == v] = k
                kept_total += 1
            else:
                dropped += 1
                dropped_total += 1
        if dropped:
            report.append({'id': t['id'], 'dropped': dropped, 'left': k})
        Image.fromarray(out).save(ip)
        Image.fromarray(np.where(out > 0, 255, 0).astype(np.uint8)).save(
            os.path.join(work, 'masks', f'{t["id"]}.png'))
        meta['nails'] = k
        meta['areas'] = {str(v): int((out == v).sum())
                         for v in np.unique(out) if v}
        meta['tips_filtered'] = dropped
        with open(mp_, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        paint(Image.fromarray(rgb), out,
              os.path.join(work, 'painted', f'{t["id"]}.jpg'))
    print(f'оставлено {kept_total}, убрано {dropped_total}, '
          f'кадров без кисти пропущено {skipped}')
    for r in report[:40]:
        print(f'  {r["id"]}: убрано {r["dropped"]}, осталось {r["left"]}')


if __name__ == '__main__':
    main()

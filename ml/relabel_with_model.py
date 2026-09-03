"""Переразметка колоды обученной моделью: второй круг.

Зачем. Первый круг размечал через MediaPipe: он находит кисть, по суставам
считается положение ногтя, дальше SAM обводит контур. Там, где кисть обрезана
рукавом или в кадре вторая рука, MediaPipe ошибается — и вместе с ним ошибается
всё остальное. Из 245 работ колоды в набор прошли 74.

Здесь кисть не ищется вовсе. Ногти находит модель, обученная на чистом наборе
первого круга, а SAM только уточняет край. Модель видит ноготь как ноготь, ей
всё равно, видно ли ладонь и сколько в кадре рук.

Права остаются чистыми: модель обучена только на CC0 и нашей колоде, SAM —
Apache-2.0. Подробности в PROVENANCE.md.

    python relabel_with_model.py --model nail-unet.onnx --src ../../app/assets/works \\
        --out deck_round2 --sam mobile_sam.pt

Выход тот же, что у label_deck.py, поэтому review_page.py принимает его без
изменений — приёмка идёт по той же странице.
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Какой SAM обводит контур. MobileSAM (vit_t) быстрый, большая vit_b точнее по
# краю — а три четверти ошибки модели лежат именно у края ногтя.
# Каждая сборка идёт со своим классом-предиктором; смешивать их нельзя,
# поэтому храним пару «конструктор модели, предиктор» на каждый тип.
SAM_REGISTRY = {}
try:
    from mobile_sam import (SamPredictor as _MobilePredictor,
                            sam_model_registry as _mobile)
    SAM_REGISTRY['vit_t'] = (_mobile['vit_t'], _MobilePredictor)
except ImportError:
    pass
try:
    from segment_anything import (SamPredictor as _BigPredictor,
                                  sam_model_registry as _big)
    for _k in ('vit_b', 'vit_l', 'vit_h'):
        if _k in _big:
            SAM_REGISTRY[_k] = (_big[_k], _BigPredictor)
except ImportError:
    pass

from label_deck import largest_blob, solidity

# Порог вероятности, ниже которого пиксель не считается ногтем.
THRESH = 0.5
# Кандидат мельче этой доли кадра — шум, а не ноготь.
MIN_AREA_FRAC = 2e-4
# Насколько SAM позволено расширить пятно, которое дала модель. Больше —
# значит уехал с ногтя на палец.
GROW_MAX = 2.2
GROW_MIN = 0.45
SOLIDITY_MIN = 0.75
# Запас вокруг рамки кандидата, в долях его большей стороны.
BOX_PAD = 0.18


def letterbox(im, size):
    """Вписывает кадр в квадрат, не растягивая. Возвращает картинку и данные
    для обратного пересчёта координат."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    off = ((side - w) // 2, (side - h) // 2)
    canvas.paste(im, off)
    return canvas.resize((size, size), Image.BILINEAR), side, off


def model_mask(sess, im, size):
    """Маска модели, приведённая обратно к размеру исходного кадра."""
    sq, side, off = letterbox(im, size)
    x = np.asarray(sq, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    big = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
    w, h = im.size
    return big[off[1]:off[1] + h, off[0]:off[0] + w]


def candidates(prob, min_area):
    """Связные области выше порога — заготовки под ногти."""
    binary = (prob > THRESH).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out.append({
            'mask': lab == i,
            'area': area,
            'box': np.array([stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                             stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]],
                            dtype=np.float32),
            'centre': (float(cent[i][0]), float(cent[i][1])),
        })
    return out


def refine(predictor, cand, W, H):
    """Уточняет край ногтя через SAM. Заготовку модели используем и как
    подсказку, и как меру: маска, выросшая вдвое, — это уже палец."""
    box = cand['box'].copy()
    pad = BOX_PAD * max(box[2] - box[0], box[3] - box[1])
    box[0] = max(0, box[0] - pad); box[1] = max(0, box[1] - pad)
    box[2] = min(W - 1, box[2] + pad); box[3] = min(H - 1, box[3] + pad)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None

    pts = np.array([cand['centre']], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=np.array([1], dtype=np.int32),
        box=box[None, :], multimask_output=True)

    seed = cand['mask']
    best, best_key = None, None
    for m, s in zip(masks, scores):
        m = largest_blob(m.astype(bool))
        if m is None or m.sum() < 30:
            continue
        grow = float(m.sum()) / cand['area']
        if not (GROW_MIN <= grow <= GROW_MAX):
            continue
        sol = solidity(m)
        if sol < SOLIDITY_MIN:
            continue
        inter = np.logical_and(m, seed).sum()
        cover = float(inter) / cand['area']   # насколько накрыта заготовка
        if cover < 0.5:
            continue
        key = float(s) + 0.6 * sol + 0.8 * cover - 0.3 * abs(grow - 1.0)
        if best_key is None or key > best_key:
            best, best_key = m, key
    # Если SAM не дал ничего разумного, оставляем заготовку модели: она
    # грубее по краю, но по месту верна.
    return best if best is not None else seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='ONNX чистой модели')
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sam', required=True)
    ap.add_argument('--sam-type', default='vit_t', choices=sorted(SAM_REGISTRY),
                    help='vit_t — MobileSAM (быстро), vit_b — большая (точнее край)')
    ap.add_argument('--skip', default=None,
                    help='файл со списком имён, которые уже в наборе')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    skip = set()
    if args.skip and os.path.exists(args.skip):
        skip = {x.strip() for x in open(args.skip, encoding='utf-8').read()
                .replace(',', '\n').split() if x.strip()}
        print(f'уже в наборе: {len(skip)} — пропускаю')

    files = []
    for root, _, names in os.walk(args.src):
        for n in sorted(names):
            if n.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                p = os.path.join(root, n)
                key = os.path.relpath(p, args.src).replace('\\', '/')
                key = key.replace('/', '-').rsplit('.', 1)[0]
                if key not in skip:
                    files.append((key, p))
    files.sort()
    if args.every > 1:
        files = files[::args.every]
    if args.limit:
        files = files[:args.limit]
    print(f'к переразметке: {len(files)}', flush=True)

    for sub in ('masks', 'images', 'overlay'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    size = sess.get_inputs()[0].shape[2]
    if not isinstance(size, int):
        size = 384
    build, Predictor = SAM_REGISTRY[args.sam_type]
    sam = build(checkpoint=args.sam)
    print(f'SAM: {args.sam_type}', flush=True)
    sam.eval()
    predictor = Predictor(sam)

    report = []
    for i, (key, path) in enumerate(files, 1):
        im = Image.open(path).convert('RGB')
        rgb = np.array(im)
        H, W = rgb.shape[:2]

        prob = model_mask(sess, im, size)
        cands = candidates(prob, MIN_AREA_FRAC * W * H)
        if not cands:
            report.append({'file': key, 'nails': 0, 'expected': 0,
                           'complete': False, 'status': 'модель ничего не нашла'})
            print(f'{i:4d}/{len(files)}  {key}: модель ничего не нашла', flush=True)
            continue

        predictor.set_image(rgb)
        full = np.zeros((H, W), dtype=bool)
        kept = 0
        for c in cands:
            m = refine(predictor, c, W, H)
            if m is not None and m.any():
                full |= m
                kept += 1

        mask = full.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cv2.imwrite(os.path.join(args.out, 'masks', key + '.png'), mask)
        im.save(os.path.join(args.out, 'images', key + '.jpg'), quality=92)

        ov = rgb.copy()
        ov[mask > 0] = (0.45 * ov[mask > 0] + 0.55 * np.array([255, 40, 90])).astype(np.uint8)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cont, -1, (255, 255, 255), 2)
        Image.fromarray(ov).save(os.path.join(args.out, 'overlay', key + '.jpg'), quality=88)

        # Уверенность модели внутри итоговой маски: по ней удобно сортировать
        # приёмку — низкая почти всегда означает спорный кадр.
        conf = float(prob[full].mean()) if full.any() else 0.0
        report.append({'file': key, 'nails': kept, 'expected': len(cands),
                       'complete': kept == len(cands), 'conf': round(conf, 3),
                       'status': 'ok'})
        print(f'{i:4d}/{len(files)}  {key}: ногтей {kept} из {len(cands)} '
              f'(уверенность {conf:.2f})', flush=True)

    ok = sum(1 for r in report if r['status'] == 'ok')
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump({'total': len(files), 'ok': ok,
                   'complete': sum(1 for r in report if r.get('complete')),
                   'items': report}, f, ensure_ascii=False, indent=1)
    print(f'\nГотово: {ok} из {len(files)}')


if __name__ == '__main__':
    main()

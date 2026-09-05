"""Экзамен модели по ручной разметке.

Первый замер в этом проекте, которому можно верить. Всё, что мерялось до
сих пор, сравнивалось с масками MediaPipe+SAM — то есть с тем же браком,
который модель и выучила. Здесь эталон обведён руками (label_tool.py), и
цифра наконец означает то, что написано.

Три числа, и каждое отвечает на свою жалобу:

  найдено ногтей   «ногти пропускает». Эталонный ноготь считается найденным,
                   если предсказание накрыло хотя бы FOUND_MIN его площади.
                   Это главное число: ноготь либо покрашен, либо нет.
  форма            «форму ногтя не распознаёт». Средний IoU по найденным
                   ногтям: насколько контур совпал с настоящей пластиной.
  лишнее           «красит не то». Связные пятна предсказания, не попавшие
                   ни в один эталонный ноготь, и сколько это площади.

Контроль и студия считаются раздельно. Контроль — реальный сценарий
приложения (человек снимает свою руку), студия — наша чистая генерация.
Смешивать их в одно число нельзя: лёгкая половина замажет тяжёлую.

    python exam.py --model out/nail-unet.onnx
    python exam.py --model a.onnx b.onnx --raw     # сравнить, без постфильтра
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import clean_masks
import synth

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'labels')

# Доля площади эталонного ногтя, которую предсказание должно накрыть, чтобы
# ноготь считался найденным. Половина — это «ноготь узнаваемо покрашен»:
# меньше выглядит как мазок с краю, больше — уже требование к форме, а форму
# отдельно меряет IoU.
FOUND_MIN = 0.5
# Пятна мельче этого в предсказании — крошка на границе, а не «покрасил не то».
MIN_BLOB = 60
PAINT = np.array([0.05, 0.78, 0.45], dtype=np.float32)


def letterbox(im, size):
    """Вписывает кадр в квадрат, не растягивая: так же готовит вход приложение."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def unletterbox(mask, w, h):
    """Обратное преобразование: из квадрата модели — в размер фотографии."""
    side = max(w, h)
    m = cv2.resize(mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
    x, y = (side - w) // 2, (side - h) // 2
    return m[y:y + h, x:x + w].astype(bool)


def run_model(sess, im, size):
    x = np.asarray(letterbox(im, size), dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    return unletterbox(prob > 0.5, im.width, im.height)


def score_frame(gt_idx, pred):
    """Сравнить предсказание с эталоном одного кадра.

    Считаем по ногтям, а не по пикселям: пользователь видит не IoU, а то,
    покрашен ли конкретный ноготь.
    """
    ids = [int(v) for v in np.unique(gt_idx) if v]
    found, ious = 0, []
    for v in ids:
        nail = gt_idx == v
        cover = float((nail & pred).sum()) / float(nail.sum())
        if cover >= FOUND_MIN:
            found += 1
            union = (nail | pred_component_of(pred, nail)).sum()
            ious.append(float((nail & pred).sum()) / float(union) if union else 0.0)

    # Лишнее — связные куски предсказания, не задевшие ни одного эталона.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), 8)
    stray, stray_px = 0, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB:
            continue
        if not (gt_idx[lab == i] > 0).any():
            stray += 1
            stray_px += area
    return {'nails': len(ids), 'found': found,
            'iou': round(float(np.mean(ious)), 3) if ious else None,
            'stray': stray, 'stray_px': stray_px}


def pred_component_of(pred, nail):
    """Часть предсказания, относящаяся к этому ногтю.

    Брать всё предсказание кадра нельзя: тогда IoU одного ногтя штрафовался
    бы за все остальные ногти на фотографии.
    """
    n, lab = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    hit = set(int(v) for v in np.unique(lab[nail]) if v)
    return np.isin(lab, list(hit)) if hit else np.zeros_like(pred)


def contour_of(mask):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask.astype(np.uint8), ker).astype(bool) & ~mask.astype(bool)


def preview(rgb, gt_idx, pred, path):
    """Покраска предсказания + белый контур эталона: где промах, видно сразу."""
    out = synth.recolor(rgb.astype(np.float32) / 255.0, pred, PAINT)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    out[contour_of(gt_idx > 0)] = (255, 255, 255)
    Image.fromarray(out).save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', nargs='+', required=True)
    ap.add_argument('--raw', action='store_true',
                    help='без постфильтра (по умолчанию как в приложении — с ним)')
    ap.add_argument('--out', default=os.path.join(WORK, 'exam'))
    args = ap.parse_args()

    with open(os.path.join(WORK, 'task.json'), encoding='utf-8') as fh:
        items = json.load(fh)['items']
    ready = [t for t in items
             if os.path.exists(os.path.join(WORK, 'instances', f'{t["id"]}.png'))]
    if not ready:
        raise SystemExit('Нет ни одной размеченной фотографии — сначала label_tool.py')
    if len(ready) < len(items):
        print(f'ВНИМАНИЕ: размечено {len(ready)} из {len(items)}, '
              f'считаю по размеченным\n')

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for mp in args.model:
        sess = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        size = sess.get_inputs()[0].shape[2]
        if not isinstance(size, int):
            size = 384
        name = os.path.splitext(os.path.basename(mp))[0]
        sub = os.path.join(args.out, name)
        os.makedirs(sub, exist_ok=True)

        rows = []
        for t in ready:
            im = Image.open(os.path.join(WORK, 'photos', t['file'])).convert('RGB')
            rgb = np.asarray(im)
            gt_idx = np.asarray(Image.open(
                os.path.join(WORK, 'instances', f'{t["id"]}.png')))
            pred = run_model(sess, im, size)
            if not args.raw:
                pred = clean_masks.clean(rgb, pred.astype(np.uint8))[0].astype(bool)
            r = score_frame(gt_idx, pred)
            r.update(id=t['id'], part=t['part'], file=t['file'])
            rows.append(r)
            preview(rgb, gt_idx, pred, os.path.join(sub, f'{t["id"]}.jpg'))

        report[name] = rows
        print(f'══ {name}{"  (без постфильтра)" if args.raw else ""}')
        # Перебираем те части, что реально размечены, а не жёсткий список:
        # иначе новая часть задания молча не попала бы в отчёт.
        for part in sorted({r['part'] for r in rows}):
            p = [r for r in rows if r['part'] == part]
            # Кадры без единого эталонного ногтя не участвуют в «найдено» и в
            # форме — там нечего искать. А вот в «лишнем» участвуют все: кадр,
            # где красить нечего, а модель что-то покрасила, — тоже промах.
            withn = [r for r in p if r['nails']]
            stray = sum(r['stray'] for r in p)
            if not withn:
                print(f'  {part:9} кадров {len(p):2}  ногтей в эталоне нет  '
                      f'лишних пятен {stray}')
                continue
            nails = sum(r['nails'] for r in withn)
            found = sum(r['found'] for r in withn)
            ious = [r['iou'] for r in withn if r['iou'] is not None]
            print(f'  {part:9} кадров {len(p):2}  '
                  f'найдено {found:3}/{nails:<3} = {100 * found / nails:5.1f}%  '
                  f'форма IoU {np.mean(ious) if ious else float("nan"):.3f}  '
                  f'лишних пятен {stray}')
        worst = sorted(rows, key=lambda r: r['found'] / max(1, r['nails']))[:5]
        print('  хуже всего: ' + ', '.join(
            f'{r["id"]} ({r["found"]}/{r["nails"]})' for r in worst))
        print(f'  покраска с контуром эталона: {os.path.relpath(sub, HERE)}\n')

    with open(os.path.join(args.out, 'exam.json'), 'w', encoding='utf-8') as fh:
        json.dump({'found_min': FOUND_MIN, 'raw': args.raw, 'models': report},
                  fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

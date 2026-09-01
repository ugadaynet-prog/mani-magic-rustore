"""Замер модели на контрольных фото, которых она не видела.

Считает IoU по эталонным маскам, если они есть, и в любом случае собирает
контактный лист: цифра говорит, стало ли лучше, а глаз — где именно ломается.

Можно сравнивать несколько моделей сразу:
    python evaluate.py --models out/старая.onnx best.onnx --src ../../тест-распознавания
"""
import argparse
import json
import math
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def letterbox(im, size):
    """Вписывает кадр в квадрат, не растягивая: так же готовит вход приложение."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def run_model(sess, im, size):
    x = np.asarray(letterbox(im, size), dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: x})[0]
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    return prob > 0.5


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--src', required=True, help='папка с контрольными фото')
    ap.add_argument('--gt', default=None, help='папка с эталонными масками (png)')
    ap.add_argument('--out', default='eval-out')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    os.makedirs(args.out, exist_ok=True)

    report = {}
    for mp in args.models:
        sess = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        size = sess.get_inputs()[0].shape[2]
        if not isinstance(size, int):
            size = 384
        # В имя листа берём и папку: файлы часто зовутся одинаково,
        # и один лист молча затирал бы другой.
        parent = os.path.basename(os.path.dirname(os.path.abspath(mp)))
        tag = f'{parent}--{os.path.splitext(os.path.basename(mp))[0]}'
        rows, ious = [], []

        tiles = []
        for f in files:
            im = Image.open(os.path.join(args.src, f)).convert('RGB')
            pred = run_model(sess, im, size)
            cover = float(pred.mean())

            gt_iou = None
            if args.gt:
                gp = os.path.join(args.gt, os.path.splitext(f)[0] + '.png')
                if os.path.exists(gp):
                    g = np.asarray(letterbox(
                        Image.open(gp).convert('RGB'), size))[:, :, 0] > 127
                    gt_iou = iou(pred, g)
                    ious.append(gt_iou)

            rows.append({'file': f, 'cover_pct': round(cover * 100, 2),
                         'iou': None if gt_iou is None else round(gt_iou, 4)})

            base = letterbox(im, size).convert('RGB')
            ov = np.asarray(base).copy()
            ov[pred] = (0.45 * ov[pred] + 0.55 * np.array([255, 40, 90])).astype(np.uint8)
            tile = Image.fromarray(ov)
            d = ImageDraw.Draw(tile)
            label = f'{os.path.splitext(f)[0]}  {cover*100:.1f}%'
            if gt_iou is not None:
                label += f'  IoU {gt_iou:.2f}'
            d.rectangle([2, 2, 2 + 8 * len(label), 20], fill='white')
            d.text((5, 5), label, fill='black')
            tiles.append(tile)

        cols = 6
        rws = math.ceil(len(tiles) / cols)
        sheet = Image.new('RGB', (cols * size, rws * size), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * size, (i // cols) * size))
        sheet_path = os.path.join(args.out, f'{tag}.jpg')
        sheet.save(sheet_path, quality=85)

        blind = [r['file'] for r in rows if r['cover_pct'] < 1.0]
        report[tag] = {'model': mp, 'input': size, 'items': rows,
                       'mean_iou': round(float(np.mean(ious)), 4) if ious else None,
                       'almost_blind': blind, 'sheet': sheet_path}
        print(f'\n=== {tag} (вход {size}) ===')
        if ious:
            print(f'  средний IoU: {np.mean(ious):.4f}')
        print(f'  почти ничего не нашла на {len(blind)} из {len(files)}: '
              f'{", ".join(blind) if blind else "—"}')
        print(f'  лист: {sheet_path}')

    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

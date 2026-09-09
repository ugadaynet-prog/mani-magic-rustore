"""Приёмы, которые ничего не переобучают, а меняют только способ запуска.

Замер 9 сентября показал неожиданное: те же веса под входом 640 вместо 512
находят на контрольных снимках 124 ногтя из 133 против 115. Рычаг оказался не
в данных, а в том, как модель запускают. Здесь проверяются ещё два приёма из
того же семейства — оба реализуемы в приложении и оба ничего не стоят, кроме
времени инференса.

  зеркало   Кадр прогоняется дважды: как есть и отражённым, вероятности
            усредняются. Сеть не симметрична, и на отражении она ошибается в
            других местах; усреднение убирает часть случайных промахов.

  два прохода  Первый проход находит ногти, по ним считается рамка кисти,
            кадр обрезается по ней с запасом и прогоняется второй раз. Ноготь
            получает больше пикселей — ровно то, из-за чего помогло 640.
            Пропущенный на первом проходе ноготь (обычно большой палец) при
            этом попадает в вырезку и может найтись со второго.

    python tta.py --model ../app-addons/tryon/nail-unet.onnx
    python tta.py --model m.onnx --work labels3
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import clean_masks
import exam

HERE = os.path.dirname(os.path.abspath(__file__))
# Насколько расширяем рамку найденных ногтей, чтобы в вырезку попала вся кисть
# и пропущенный ноготь рядом. Меньше — вырезка режет соседний палец, больше —
# теряется смысл: масштаб почти не меняется.
GROW = 1.9
# Если ногти и так занимают заметную долю кадра, второй проход не нужен:
# увеличивать уже некуда, а лишний прогон стоит времени.
SKIP_IF_FRAC = 0.10


def probs(sess, im, size):
    """Карта вероятностей в СВОЁМ разрешении модели, size×size.

    Наверх её тянуть нельзя: exam.py и приложение сначала режут по порогу, а
    потом растягивают ближайшим соседом. Если растянуть вероятности линейно и
    резать уже наверху, граница пятна получается другой — я на этом обжёгся
    9 сентября и намерил лишнего вдвое больше, чем показывает экзамен.
    Комбинировать проходы (зеркало, вырезка) надо здесь, до порога.
    """
    x = np.asarray(exam.letterbox(im, size), np.float32) / 255.0
    lg = sess.run(None, {sess.get_inputs()[0].name:
                         np.transpose(x, (2, 0, 1))[None]})[0][0, 0]
    return 1.0 / (1.0 + np.exp(-lg))


def to_frame(p, im):
    """Порог и возврат в размер фотографии — тот же путь, что в exam.py."""
    return exam.unletterbox(p > exam.THRESHOLD, im.width, im.height)


def mirrored(sess, im, size):
    a = probs(sess, im, size)
    b = probs(sess, im.transpose(Image.FLIP_LEFT_RIGHT), size)[:, ::-1]
    return np.maximum(a, b), (a + b) / 2


def crop_box(mask, w, h):
    """Рамка вокруг найденного, расширенная так, чтобы влезла кисть."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * GROW
    side = max(side, 64)
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def two_pass(sess, im, size, base_mask):
    """Второй проход по вырезке вокруг найденного, уже в системе координат кадра."""
    if not base_mask.any() or base_mask.mean() > SKIP_IF_FRAC:
        return base_mask
    box = crop_box(base_mask, im.width, im.height)
    if box is None:
        return base_mask
    x0, y0, x1, y1 = box
    sub = im.crop(box)
    m2 = to_frame(probs(sess, sub, size), sub)
    out = base_mask.copy()
    out[y0:y1, x0:x1] |= m2
    return out


def score(work, sess, size, mode):
    with open(os.path.join(work, 'task.json'), encoding='utf-8') as fh:
        items = json.load(fh)['items']
    rows = []
    for t in items:
        ip = os.path.join(work, 'instances', f'{t["id"]}.png')
        if not os.path.exists(ip):
            continue
        im = Image.open(os.path.join(work, 'photos', t['file'])).convert('RGB')
        rgb = np.asarray(im)
        gt = np.asarray(Image.open(ip))
        if mode == 'как есть':
            m = to_frame(probs(sess, im, size), im)
        elif mode == 'зеркало-макс':
            m = to_frame(mirrored(sess, im, size)[0], im)
        elif mode == 'зеркало-средн':
            m = to_frame(mirrored(sess, im, size)[1], im)
        elif mode == 'два прохода':
            m = two_pass(sess, im, size, to_frame(probs(sess, im, size), im))
        elif mode == 'зеркало+два':
            m = two_pass(sess, im, size, to_frame(mirrored(sess, im, size)[1], im))
        else:
            raise SystemExit('неизвестный режим ' + mode)
        pred = clean_masks.clean(rgb, m.astype(np.uint8))[0].astype(bool)
        r = exam.score_frame(gt, pred)
        r['part'] = t['part']
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--work', default='labels')
    ap.add_argument('--modes', nargs='+',
                    default=['как есть', 'зеркало-макс', 'зеркало-средн',
                             'два прохода', 'зеркало+два'])
    args = ap.parse_args()
    work = os.path.join(HERE, args.work)
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    size = sess.get_inputs()[0].shape[2]
    if not isinstance(size, int):
        size = 512
    print(f'модель {os.path.basename(args.model)}, вход {size}, задание {args.work}\n')
    for mode in args.modes:
        rows = score(work, sess, size, mode)
        print(f'══ {mode}')
        for part in sorted({r['part'] for r in rows}):
            p = [r for r in rows if r['part'] == part and r['nails']]
            if not p:
                continue
            nails = sum(r['nails'] for r in p)
            found = sum(r['found'] for r in p)
            ious = [r['iou'] for r in p if r['iou'] is not None]
            stray = sum(r['stray_px'] for r in rows if r['part'] == part)
            print(f'  {part:9} найдено {found:3}/{nails:<3} = {100 * found / nails:5.1f}%  '
                  f'форма {np.mean(ious):.3f}  лишнее {stray:>6} px')


if __name__ == '__main__':
    main()

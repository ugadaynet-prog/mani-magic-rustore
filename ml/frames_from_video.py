"""Отобрать из видео кадры, пригодные для ручной разметки.

Нарезать видео подряд бессмысленно: соседние кадры почти одинаковы и ничего
не добавляют, а кадр, снятый в движении, смазан и не размечается. Поэтому из
каждого ролика берутся несколько кадров, которые:

  резкие     — средний квадрат лапласиана не ниже порога; это та же мера, что
               в подсказке «кадр смазан» в приложении;
  разные     — перцептивный хэш далеко от уже взятых кадров этого ролика и от
               всех прочих; иначе двенадцать свотч-роликов с одного стола дали
               бы сотню копий одного снимка;
  не экзамен — хэш далеко от всех кадров labels и labels3, иначе экзамен
               перестал бы быть независимым от обучения.

    python frames_from_video.py --src "../../11/видео" --out "../../11/кадры"
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

import make_label_task2 as L2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = {'.mov', '.mp4', '.m4v'}
# Сколько кадров в секунду просматривать. Больше незачем: руку в свотч-ролике
# держат по нескольку секунд, и за треть секунды поза почти не меняется.
FPS = 3
# Сколько кадров оставлять с одного ролика не больше чем.
PER_VIDEO = 3
# Мягче этого кадр не размечается. Порог выше, чем у подсказки в приложении
# (0.0025): там мы предупреждаем про заведомо плохой снимок, а здесь берём
# только уверенно резкие.
SHARP_MIN = 0.004
# Ближе этого по хэшу — уже есть такой кадр.
HAMMING_SELF = 16
WORK_SIDE = 1600


def sharpness(im):
    small = im.convert('L').resize((96, 96), Image.BILINEAR)
    g = np.asarray(small, np.float32) / 255.0
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float((lap ** 2).mean())


def frames(path, tmp):
    """Кадры ролика с частотой FPS — через ffmpeg, который понимает и HEVC."""
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', path,
                    '-vf', f'fps={FPS}', os.path.join(tmp, '%04d.jpg')],
                   check=True)
    return sorted(os.path.join(tmp, f) for f in os.listdir(tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip', nargs='*', default=[],
                    help='ролики, которые не брать (без расширения)')
    args = ap.parse_args()
    src = args.src if os.path.isabs(args.src) else os.path.join(HERE, args.src)
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)

    gold = []
    for w in ('labels', 'labels3'):
        d = os.path.join(HERE, w, 'photos')
        gold += [L2.ahash(os.path.join(d, f)) for f in os.listdir(d)]
    det = L2.make_hand_detector()

    taken_hashes, log = [], []
    tmp = tempfile.mkdtemp(prefix='frames-')
    for name in sorted(os.listdir(src)):
        base, ext = os.path.splitext(name)
        if ext.lower() not in VIDEO_EXT or base in args.skip:
            continue
        cands = []
        for p in frames(os.path.join(src, name), tmp):
            im = Image.open(p).convert('RGB')
            s = sharpness(im)
            if s < SHARP_MIN:
                continue
            cands.append((s, p))
        # Самые резкие первыми: из двух почти одинаковых остаётся лучший.
        cands.sort(reverse=True)
        kept = 0
        for s, p in cands:
            if kept >= PER_VIDEO:
                break
            im = Image.open(p).convert('RGB')
            h = L2.ahash(im)
            if any(int((h != g).sum()) < L2.HAMMING_GOLD for g in gold):
                continue
            if any(int((h != t).sum()) < HAMMING_SELF for t in taken_hashes):
                continue
            if L2.hand_fraction(det, im) < L2.HAND_MIN_FRAC:
                continue
            k = WORK_SIDE / max(im.size)
            if k < 1:
                im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
            fn = f'{base}-{kept + 1}.jpg'
            im.save(os.path.join(out, fn), quality=93, subsampling=0)
            taken_hashes.append(h)
            kept += 1
            log.append({'file': fn, 'video': name, 'sharp': round(s, 5)})
        print(f'{name:14} кандидатов {len(cands):3}  взято {kept}', flush=True)

    with open(os.path.join(out, 'кадры.json'), 'w', encoding='utf-8') as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    print(f'\nвсего кадров: {len(log)} → {out}')


if __name__ == '__main__':
    main()

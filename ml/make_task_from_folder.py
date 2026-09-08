"""Сделать задание на разметку из обычной папки со снимками.

Третий круг разметки собирается уже не отбором из общей кучи (для этого есть
make_label_task2.py), а из готовой подборки: владелец приносит папку под
конкретную дыру — например, руки без лака, — и её надо просто превратить в
задание.

Проверка на пересечение с эталоном тут тоже нужна: если снимок совпадёт с
одним из 40 экзаменационных кадров, экзамен перестанет быть независимым.

    python make_task_from_folder.py --src "../../ногти-без-лака" --work labels3

Дальше — как обычно:
    python label_tool.py --work labels3 --precompute
    python label_pass1.py --work labels3 --grid 01 02 ...
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PHOTOS = os.path.join(HERE, 'labels', 'photos')
EXT = {'.jpg', '.jpeg', '.png', '.webp'}
WORK_SIDE = 1024
HAMMING_GOLD = 20


def ahash(path_or_img, n=12):
    im = (Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img)
    a = np.asarray(im.convert('L').resize((n, n), Image.BILINEAR), np.float32)
    return (a > a.mean()).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='папка со снимками')
    ap.add_argument('--work', required=True, help='куда положить задание')
    ap.add_argument('--part', default='без лака', help='как назвать часть в отчётах')
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(HERE, args.src)
    work = args.work if os.path.isabs(args.work) else os.path.join(HERE, args.work)

    gold = [ahash(os.path.join(GOLD_PHOTOS, f)) for f in os.listdir(GOLD_PHOTOS)]
    for sub in ('photos', 'masks', 'instances', 'meta'):
        os.makedirs(os.path.join(work, sub), exist_ok=True)

    files = sorted(f for f in os.listdir(src)
                   if os.path.splitext(f)[1].lower() in EXT)
    task, skipped = [], 0
    for f in files:
        im = Image.open(os.path.join(src, f)).convert('RGB')
        k = WORK_SIDE / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        if min(int((ahash(im) != g).sum()) for g in gold) < HAMMING_GOLD:
            skipped += 1
            continue
        iid = f'{len(task) + 1:02d}'
        name = f'{iid}.jpg'
        im.save(os.path.join(work, 'photos', name), quality=95, subsampling=0)
        task.append({'id': iid, 'file': name, 'part': args.part,
                     'origin': f, 'w': im.width, 'h': im.height})

    with open(os.path.join(work, 'task.json'), 'w', encoding='utf-8') as fh:
        json.dump({'work_side': WORK_SIDE, 'items': task}, fh,
                  ensure_ascii=False, indent=1)
    print(f'взято {len(task)}, отброшено как совпадающие с эталоном {skipped}')
    print(f'задание в {os.path.relpath(work, HERE)}')


if __name__ == '__main__':
    main()

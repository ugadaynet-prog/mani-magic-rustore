"""Пересчитать номера ногтей по связным областям маски.

Зачем. В инструменте разметки номер ногтя присваивается в момент, когда
область принимают: кнопкой «Взять ноготь» — новый номер, кистью — номер
последнего принятого. Если ноготь дорисовывали или перерисовывали кистью,
несколько ногтей получают один и тот же номер: сама маска верная, а счёт
ногтей — нет. На кадре 16 так вышло 10 ногтей под одним номером.

Для экзамена номера важны: он считает, сколько ЭТАЛОННЫХ ногтей накрыла
модель, и слипшиеся в один номер десять ногтей превратились бы в один.

Маску скрипт не трогает вовсе — только переназначает номера по связным
областям. Ногти на фотографии почти всегда разделены кожей, так что связная
область и есть ноготь; исключения (два слипшихся ногтя) скрипт покажет как
область, вдвое крупнее соседей, — это видно в отчёте.

    python renumber_instances.py --dry-run
    python renumber_instances.py
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'labels')

# Крошка — след кисти, а не ноготь: одиночный клик оставляет пятно в
# единицы пикселей. Порог берём от САМОГО КРУПНОГО ногтя кадра, а не от
# медианы: там, где крошек больше, чем ногтей (кадр 21 — одиннадцать против
# пяти), медиана сама оказывается крошкой и порог обнуляется. Ногти одного
# кадра различаются по площади в разы, но не на порядок, поэтому двадцатая
# доля от крупнейшего отсекает мусор и не задевает настоящие ногти: самый
# мелкий из них в наборе — 1138 px на кадре 16 при пороге 570.
CRUMB_PX = 60
CRUMB_FRAC = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    with open(os.path.join(WORK, 'task.json'), encoding='utf-8') as fh:
        items = json.load(fh)['items']

    changed = 0
    for t in items:
        ip = os.path.join(WORK, 'instances', f'{t["id"]}.png')
        mp = os.path.join(WORK, 'meta', f'{t["id"]}.json')
        if not os.path.exists(ip):
            continue
        idx = np.asarray(Image.open(ip))
        was = len([v for v in np.unique(idx) if v])
        n, lab = cv2.connectedComponents((idx > 0).astype(np.uint8), connectivity=8)
        areas = [(int((lab == i).sum()), i) for i in range(1, n)]
        areas.sort(reverse=True)
        if len(areas) > 255:
            print(f'{t["id"]}: областей {len(areas)} — больше 255, пропускаю')
            continue

        biggest = areas[0][0] if areas else 0
        floor = max(CRUMB_PX, CRUMB_FRAC * biggest)
        keep, crumbs = [], []
        for a, i in areas:
            (keep if a >= floor else crumbs).append((a, i))
        crumbs = [a for a, _ in crumbs]

        out = np.zeros_like(idx)
        for k, (_, i) in enumerate(keep, 1):
            out[lab == i] = k

        note = f'  убрано крошек: {len(crumbs)} ({crumbs})' if crumbs else ''
        if was != len(keep) or note:
            changed += 1
            print(f'{t["id"]}: номеров {was} → ногтей {len(keep)}{note}')
        if args.dry_run:
            continue
        Image.fromarray(out).save(ip)
        Image.fromarray(np.where(out > 0, 255, 0).astype(np.uint8)).save(
            os.path.join(WORK, 'masks', f'{t["id"]}.png'))
        with open(mp, encoding='utf-8') as fh:
            m = json.load(fh)
        m['nails'] = len(keep)
        m['areas'] = {str(k): a for k, (a, _) in enumerate(keep, 1)}
        m['crumbs_removed'] = crumbs
        m['renumbered'] = True
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(m, fh, ensure_ascii=False, indent=1)

    print(f'\n{"Показано" if args.dry_run else "Исправлено"} кадров: {changed}')


if __name__ == '__main__':
    main()

"""Чистка масок от пятен, севших мимо руки.

Наблюдение из приёмки второго круга: модель ошибается однообразно — отдельные
пятна уезжают на розу, жемчуг, край чашки, неоновую вывеску. На самих пальцах
она работает верно. Значит брак ловится без всякого глаза: ноготь всегда сидит
на пальце, то есть по его краю есть кожа, а пятно на жемчуге окружено жемчугом.

Кожу определяем в YCbCr — там она занимает узкий и устойчивый диапазон по
цветности, почти независимо от освещения и тона кожи. Для каждой связной
области берём кольцо вокруг неё и смотрим, какая доля кольца похожа на кожу.

    python clean_masks.py --ds ../../deck_round2
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

# Классический диапазон кожи по цветности. Яркость (Y) намеренно не трогаем:
# она меняется от освещения, а Cb/Cr — почти нет.
CB_LO, CB_HI = 77, 130
CR_LO, CR_HI = 133, 177

RING_PX = 7          # ширина кольца вокруг пятна, в пикселях
RING_SKIN_MIN = 0.22  # ниже этой доли кожи в кольце считаем пятно чужим
MIN_AREA = 60        # совсем мелкие крошки убираем без разговоров

# Кольцо не спасает от предмета, зажатого В РУКЕ: вишня окружена пальцами, и
# кожи вокруг неё хватает. Зато ногти одной кисти сопоставимы по размеру, а
# вишня заметно крупнее — по этому и ловим. Порог применяем только когда
# областей достаточно, чтобы медиана что-то значила.
AREA_OUTLIER = 2.5
AREA_OUTLIER_MIN_PARTS = 4


def skin_map(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    return (cb >= CB_LO) & (cb <= CB_HI) & (cr >= CR_LO) & (cr <= CR_HI)


def clean(rgb, mask):
    """Возвращает очищенную маску и список причин по каждой удалённой области."""
    skin = skin_map(rgb)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1,) * 2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    dropped = []
    kept = []
    for i in range(1, n):
        comp = lab == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            dropped.append({'area': area, 'ring_skin': None, 'why': 'мелкая крошка'})
            continue
        ring = cv2.dilate(comp.astype(np.uint8), ker).astype(bool) & ~comp
        if ring.sum() < 20:
            dropped.append({'area': area, 'ring_skin': None, 'why': 'нет кольца'})
            continue
        frac = float(skin[ring].mean())
        if frac < RING_SKIN_MIN:
            dropped.append({'area': area, 'ring_skin': round(frac, 3),
                            'why': 'вокруг не кожа'})
            continue
        kept.append((comp, area, frac))

    # Второй проход — по размеру: он имеет смысл только когда уже известно,
    # каков в этом кадре типичный ноготь.
    if len(kept) >= AREA_OUTLIER_MIN_PARTS:
        med = float(np.median([a for _, a, _ in kept]))
        keep2 = []
        for comp, area, frac in kept:
            if area > AREA_OUTLIER * med:
                dropped.append({'area': area, 'ring_skin': frac,
                                'why': f'крупнее ногтей в {area/med:.1f} раза'})
            else:
                keep2.append((comp, area, frac))
        kept = keep2

    for comp, _, _ in kept:
        out |= comp
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, help='папка с images/ и masks/')
    ap.add_argument('--dry-run', action='store_true',
                    help='только посчитать, файлы не трогать')
    args = ap.parse_args()

    img_dir = os.path.join(args.ds, 'images')
    mask_dir = os.path.join(args.ds, 'masks')
    report = []
    tot_drop = tot_keep = emptied = 0

    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        mp = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mp):
            continue
        rgb = np.asarray(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        mk = np.asarray(Image.open(mp).convert('L'))
        if mk.shape != rgb.shape[:2]:
            mk = np.asarray(Image.fromarray(mk).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST))

        cleaned, dropped = clean(rgb, mk)
        kept = int(cv2.connectedComponentsWithStats(
            cleaned.astype(np.uint8), connectivity=8)[0]) - 1
        tot_drop += len(dropped)
        tot_keep += kept
        if kept == 0:
            emptied += 1
        if dropped:
            report.append({'file': stem, 'kept': kept, 'dropped': dropped})
            why = ', '.join(f"{d['why']}" for d in dropped)
            print(f'{stem}: убрано {len(dropped)}, осталось {kept}  ({why})')
        if not args.dry_run:
            Image.fromarray((cleaned > 0).astype(np.uint8) * 255).save(mp)

    with open(os.path.join(args.ds, 'cleanup.json'), 'w', encoding='utf-8') as f:
        json.dump({'ring_skin_min': RING_SKIN_MIN, 'kept': tot_keep,
                   'dropped': tot_drop, 'emptied': emptied,
                   'items': report}, f, ensure_ascii=False, indent=1)
    print(f'\nОбластей оставлено {tot_keep}, убрано {tot_drop}; '
          f'кадров опустело {emptied}')


if __name__ == '__main__':
    main()

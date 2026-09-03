"""Оставить в маске только то, что сидит на кончике пальца.

Зачем. Разметчик, ведомый моделью, наследует её слабость: модель выучила
«гладкий блестящий овал ≈ ноготь» и красит янтарь в кольце, стеклянный шарик
в ладони, виноградину между пальцами, пятно на блокноте. Правила из
clean_masks (кожа вокруг, размер относительно соседей) такое не ловят —
предмет и лежит на коже, и по размеру с ноготь.

Зато у ногтя есть свойство, которого нет ни у одного из этих предметов: он
растёт из кончика пальца. MediaPipe даёт кончики, и всё, что от них далеко,
из маски убирается.

Кадры, где кисть не нашлась, скрипт не трогает — там судить не по чему.

    python keep_at_tips.py --ds ../../hands40 --hands hand_landmarker.task
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from label_deck import detect_hands, nail_prompts

# Насколько далеко от кончика пальца пятну позволено находиться, в долях длины
# дистальной фаланги. Ноготь начинается прямо у кончика; предмет в руке — нет.
TIP_MAX = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, help='папка с images/ и masks/')
    ap.add_argument('--hands', required=True, help='hand_landmarker.task')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.hands),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=3, min_hand_detection_confidence=0.2))

    img_dir = os.path.join(args.ds, 'images')
    mask_dir = os.path.join(args.ds, 'masks')
    report = []
    dropped_total = kept_total = skipped = 0

    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        mp_path = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mp_path):
            continue
        rgb = np.asarray(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        mk = np.asarray(Image.open(mp_path).convert('L'))
        if mk.shape != rgb.shape[:2]:
            mk = cv2.resize(mk, (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        hands, to_orig, _, _ = detect_hands(det, rgb)
        if not hands:
            skipped += 1
            continue

        tips = []
        for lm in hands:
            for p in nail_prompts(lm, to_orig):
                tips.append((p['tip'][0], p['tip'][1], p['L']))
        if not tips:
            skipped += 1
            continue

        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (mk > 127).astype(np.uint8), connectivity=8)
        out = np.zeros(mk.shape, dtype=np.uint8)
        dropped = []
        for i in range(1, n):
            cx, cy = cent[i]
            near = min((np.hypot(cx - tx, cy - ty) / L) for tx, ty, L in tips)
            if near <= TIP_MAX:
                out[lab == i] = 255
                kept_total += 1
            else:
                dropped.append({'area': int(stats[i, cv2.CC_STAT_AREA]),
                                'tips_away': round(float(near), 2)})
                dropped_total += 1

        if dropped:
            report.append({'file': stem, 'dropped': dropped})
            print(f'{stem}: убрано {len(dropped)} '
                  f'(дальше кончика в {", ".join(str(d["tips_away"]) for d in dropped)} раза)')
        if not args.dry_run:
            Image.fromarray(out).save(mp_path)

    with open(os.path.join(args.ds, 'tips_filter.json'), 'w', encoding='utf-8') as fh:
        json.dump({'tip_max': TIP_MAX, 'kept': kept_total,
                   'dropped': dropped_total, 'skipped_no_hand': skipped,
                   'items': report}, fh, ensure_ascii=False, indent=1)
    print(f'\nОставлено {kept_total}, убрано {dropped_total}; '
          f'кадров без кисти пропущено {skipped}')


if __name__ == '__main__':
    main()

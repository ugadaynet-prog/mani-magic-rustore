"""Разметка масок ногтей для собственной колоды работ.

Зачем. Готового набора масок ногтей со свободной лицензией не существует:
из шести источников в prepare_dataset.py лицензия есть только у vpapenko
(CC0, 52 фото), остальные пять — без лицензии вообще. Зато у нас есть
245 сгенерированных нами работ колоды, права на которые наши. Не хватает
им только масок — их и делает этот скрипт.

Цепочка прав чистая, ни один шаг не опирается на нелицензионные данные:
  MediaPipe Hand Landmarker (Apache-2.0) находит кисть и суставы пальцев
  -> из кончика и сустава DIP считаем, где лежит ногтевая пластина
  -> MobileSAM (Apache-2.0) по точке и рамке обводит контур ногтя.

Выход:
  <out>/masks/<card>-<n>.png    бинарная маска, 0/255
  <out>/images/<card>-<n>.jpg   исходник рядом, чтобы набор был самодостаточным
  <out>/overlay/<card>-<n>.jpg  маска поверх фото — для приёмки глазами
  <out>/report.json             что нашлось, что нет, сколько ногтей на кадр
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mobile_sam import SamPredictor, sam_model_registry

# Кончики пальцев и ближайший к ним сустав. Вектор между ними задаёт
# и направление ногтя, и его масштаб — палец крупнее, ноготь крупнее.
TIPS = [4, 8, 12, 16, 20]
DIPS = [3, 7, 11, 15, 19]

# Гипотезы о том, где вдоль оси пальца лежит ногтевая пластина. MediaPipe
# обучен на натуральных руках и ставит «кончик» по коже, а маникюрный ноготь
# уходит дальше — иногда вдвое длиннее фаланги. Поэтому проверяем несколько
# отрезков разной длины: от короткого ногтя до очень длинного нарощенного.
# Пары (начало, конец) в долях длины дистальной фаланги, отсчёт от кончика,
# плюс — вперёд, за кончик.
SPANS = ((-0.55, 0.15), (-0.35, 0.55), (-0.20, 1.10), (-0.10, 1.80))
HALF_W = 0.45    # полуширина рамки в долях длины фаланги
NEG_DOWN = 0.55  # насколько ниже сустава ставить отрицательную точку

# Границы площади в долях КВАДРАТА длины фаланги, а не площади рамки: рамка у
# длинных ногтей большая, и доля от неё ничего не говорит.
AREA_MIN_L2 = 0.03
AREA_MAX_L2 = 1.80

# Насколько далеко назад, за кончик пальца, маске позволено заходить. Если
# заходит сильно — это уже палец, а не ноготь.
BACK_LIMIT = -0.85
# Полуширина поперёк оси пальца: ноготь не шире самого пальца.
SIDE_LIMIT = 0.70
# Ноготь растёт из пальца, а не висит рядом. Маска обязана подходить вплотную
# к точке кончика; без этого длинные гипотезы уводят её на фон за пальцем —
# на мох, мех, ткань, — и там она выглядит вполне убедительно.
ANCHOR_MAX = 0.30

# Доля пикселей маски, окрашенных как кожа. У ногтя она мала даже у нюдовых
# оттенков; у маски, уехавшей на палец, велика.
SKIN_FRAC_MAX = 0.42
SKIN_TOL = 42.0   # расстояние в RGB, ближе которого цвет считаем кожей

# Ноготь лежит на пальце, поэтому вокруг него кожа. Проверка кольцом вокруг
# маски не зависит от точек MediaPipe — а когда кисть сильно обрезана, точки
# как раз и врут. Пятно, взятое на мхе или мехе, окружено мхом и мехом.
RING_SKIN_MIN = 0.18
RING_WIDTH = 0.16   # ширина кольца в долях длины фаланги

# Ноготь — гладкое выпуклое пятно. Пятно, уехавшее на ткань или волосы, рваное.
# Порог мягче прежнего: у ногтей с блёстками и рисунком край тоже неровный.
SOLIDITY_MIN = 0.78


ROTATIONS = (0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
             cv2.ROTATE_90_COUNTERCLOCKWISE)
# Доли полей, дорисовываемых по краям. Детектор ладони MediaPipe рассчитан на
# кадр, где кисть видна целиком; в колоде она часто обрезана рукавом и упирается
# в край. Поля отодвигают её от границы, и детектор её находит.
PADS = (0.0, 0.35, 0.8)
# Сколько кистей искать. Две — обычный для колоды случай; третья бывает редко,
# но перебор дешевле пропуска.
MAX_HANDS = 3


def detect_hands(detector, rgb):
    """Ищет кисти, перебирая поля и повороты. Возвращает точки и функцию,
    переводящую их нормированные координаты обратно в исходный кадр.

    Берём не первый удачный вариант, а тот, где кистей НАЙДЕНО БОЛЬШЕ. В колоде
    хватает кадров с двумя руками, и при одном ракурсе детектор видит только
    одну; ногти второй тогда молча уходят в фон и учат модель, что это не ногти.
    """
    H, W = rgb.shape[:2]
    best = ([], None, 0.0, 0)
    for pad in PADS:
        if pad == 0.0:
            padded, ox, oy = rgb, 0, 0
        else:
            px, py = int(W * pad), int(H * pad)
            padded = cv2.copyMakeBorder(rgb, py, py, px, px, cv2.BORDER_REPLICATE)
            ox, oy = px, py
        for rot in ROTATIONS:
            img = padded if rot == 0 else cv2.rotate(padded, rot)
            res = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
            if len(res.hand_landmarks) <= len(best[0]):
                continue
            rh, rw = img.shape[:2]

            def to_orig(nx, ny, _rw=rw, _rh=rh, _rot=rot, _ox=ox, _oy=oy):
                x, y = unrotate_point(nx * _rw, ny * _rh, _rot, _rw, _rh)
                return x - _ox, y - _oy

            best = (res.hand_landmarks, to_orig, pad, rot)
            if len(best[0]) >= MAX_HANDS:
                return best
        # Поля нужны там, где кисть упирается в край и не находится вовсе.
        # Если на этом уровне полей она уже нашлась, дальше не идём: следующий
        # уровень втрое больше по пикселям и стоит заметного времени.
        if best[0]:
            break
    return best


def unrotate_point(x, y, rot, w, h):
    """Переводит точку из повёрнутого кадра обратно в исходные координаты.
    w, h — размеры ПОВЁРНУТОГО кадра."""
    if rot == 0:
        return x, y
    if rot == cv2.ROTATE_90_CLOCKWISE:
        # повёрнутый (w,h) получен из исходного (h,w)
        return y, (w - 1) - x
    if rot == cv2.ROTATE_180:
        return (w - 1) - x, (h - 1) - y
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (h - 1) - y, x
    return x, y


def nail_prompts(lm, to_orig):
    """Из 21 точки кисти делает подсказки для SAM по каждому из пяти ногтей."""
    out = []
    for tip_i, dip_i in zip(TIPS, DIPS):
        tx, ty = to_orig(lm[tip_i].x, lm[tip_i].y)
        dx, dy = to_orig(lm[dip_i].x, lm[dip_i].y)

        vx, vy = tx - dx, ty - dy
        L = float(np.hypot(vx, vy))
        if L < 4:
            continue
        ux, uy = vx / L, vy / L

        # Поперечное направление — вдоль него меряем ширину ногтя.
        px_, py_ = -uy, ux

        # Отрицательная точка — ниже сустава, на средней фаланге: там гарантированно
        # кожа. Она же даёт эталон цвета кожи для выбора маски.
        nx, ny = dx - NEG_DOWN * L * ux, dy - NEG_DOWN * L * uy
        cands = []
        for s0, s1 in SPANS:
            mid = 0.5 * (s0 + s1)
            cx, cy = tx + mid * L * ux, ty + mid * L * uy
            # Рамку строим как прямоугольник вдоль пальца, а SAM отдаём его
            # габарит по осям кадра — рамки под углом он не принимает.
            corners = []
            for s in (s0, s1):
                for w in (-HALF_W, HALF_W):
                    corners.append((tx + s * L * ux + w * L * px_,
                                    ty + s * L * uy + w * L * py_))
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            cands.append({'center': (cx, cy),
                          'box': np.array([min(xs), min(ys), max(xs), max(ys)])})
        out.append({'cands': cands, 'neg': (nx, ny), 'skin': (nx, ny),
                    'tip': (tx, ty), 'axis': (ux, uy), 'perp': (px_, py_),
                    'L': L})
    return out


def skin_colour(rgb, pt, L):
    """Эталон цвета кожи: медиана в кружке на средней фаланге."""
    H, W = rgb.shape[:2]
    r = max(2, int(0.22 * L))
    x, y = int(pt[0]), int(pt[1])
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0)


def largest_blob(mask):
    """Оставляет одну связную область — самую крупную. Ноготь один, а SAM
    иногда прихватывает крошку по соседству."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def solidity(mask):
    """Площадь пятна к площади его выпуклой оболочки: 1.0 у гладкого овала,
    заметно меньше у рваного."""
    cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return 0.0
    c = max(cont, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    return float(cv2.contourArea(c) / ha) if ha > 0 else 0.0


def segment_nail(predictor, p, W, H, rgb):
    """Перебирает гипотезы о длине и положении ногтя и возвращает лучшую маску.

    Проверки нарочно привязаны к самому пальцу, а не к рамке: рамка у длинного
    нарощенного ногтя большая, и доля площади от неё ничего не значит. Мерим
    иначе — насколько далеко маска заходит назад за кончик (это был бы палец),
    не шире ли она пальца, и сколько внутри неё пикселей цвета кожи.
    """
    skin = skin_colour(rgb, p['skin'], p['L'])
    labels = np.array([1, 0], dtype=np.int32)
    L = p['L']
    ux, uy = p['axis']
    px_, py_ = p['perp']
    tx, ty = p['tip']
    best, best_key = None, None

    for c in p['cands']:
        box = c['box'].copy()
        box[0::2] = np.clip(box[0::2], 0, W - 1)
        box[1::2] = np.clip(box[1::2], 0, H - 1)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue

        pts = np.array([c['center'], p['neg']], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=labels,
            box=box[None, :], multimask_output=True)

        # Обрезаем по рамке с запасом: SAM иногда цепляет соседний палец.
        keep = np.zeros((H, W), dtype=bool)
        pad = 0.15 * L
        y0 = int(max(0, box[1] - pad)); y1 = int(min(H, box[3] + pad))
        x0 = int(max(0, box[0] - pad)); x1 = int(min(W, box[2] + pad))
        keep[y0:y1, x0:x1] = True

        for m, s in zip(masks, scores):
            m = m & keep
            if m.sum() < 30:
                continue
            m = largest_blob(m)
            if m is None or m.sum() < 30:
                continue

            area = float(m.sum()) / (L * L)
            if not (AREA_MIN_L2 <= area <= AREA_MAX_L2):
                continue

            ys, xs = np.nonzero(m)
            dx_, dy_ = xs - tx, ys - ty
            along = (dx_ * ux + dy_ * uy) / L
            side = np.abs(dx_ * px_ + dy_ * py_) / L
            # Пятый процентиль, а не минимум: одиночные пиксели по краю
            # не должны решать судьбу всей маски.
            if float(np.percentile(along, 5)) < BACK_LIMIT:
                continue
            if float(np.percentile(side, 95)) > SIDE_LIMIT:
                continue
            anchor = float(np.sqrt((dx_ * dx_ + dy_ * dy_).min())) / L
            if anchor > ANCHOR_MAX:
                continue

            sol = solidity(m)
            if sol < SOLIDITY_MIN:
                continue

            key = float(s) + 0.4 * sol
            if skin is not None:
                cols = rgb[m].astype(np.float32)
                d = np.linalg.norm(cols - skin[None, :], axis=1)
                skin_frac = float((d < SKIN_TOL).mean())
                if skin_frac > SKIN_FRAC_MAX:
                    continue

                # Кольцо берём не вокруг всей маски, а только у ОСНОВАНИЯ —
                # со стороны кисти. У длинного ногтя кончик и бока висят над
                # фоном, и кожи там нет по делу; а вот основание растёт из
                # пальца всегда.
                r = max(2, int(RING_WIDTH * L))
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
                ring = cv2.dilate(m.astype(np.uint8), ker).astype(bool) & ~m
                ry, rx = np.nonzero(ring)
                r_along = ((rx - tx) * ux + (ry - ty) * uy) / L
                base = r_along < float(np.percentile(along, 40))
                if base.sum() < 20:
                    continue
                rc = rgb[ry[base], rx[base]].astype(np.float32)
                ring_skin = float((np.linalg.norm(rc - skin[None, :], axis=1)
                                   < SKIN_TOL * 1.3).mean())
                if ring_skin < RING_SKIN_MIN:
                    continue

                key += (1.2 * float(d.mean()) / 441.0
                        - 0.6 * skin_frac + 0.5 * ring_skin)
            if best_key is None or key > best_key:
                best, best_key = m, key
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='папка с работами колоды')
    ap.add_argument('--out', required=True, help='куда складывать набор')
    ap.add_argument('--sam', required=True, help='mobile_sam.pt')
    ap.add_argument('--hands', required=True, help='hand_landmarker.task')
    ap.add_argument('--limit', type=int, default=0, help='взять только N первых')
    ap.add_argument('--every', type=int, default=1,
                    help='брать каждый N-й кадр — чтобы проба была разнообразной')
    args = ap.parse_args()

    files = []
    for root, _, names in os.walk(args.src):
        for n in sorted(names):
            if n.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, n))
    files.sort()
    if args.every > 1:
        files = files[::args.every]
    if args.limit:
        files = files[:args.limit]
    print(f'Кадров к разметке: {len(files)}', flush=True)

    for sub in ('masks', 'images', 'overlay'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.hands),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS, min_hand_detection_confidence=0.2))

    sam = sam_model_registry['vit_t'](checkpoint=args.sam)
    sam.eval()
    predictor = SamPredictor(sam)

    report = []
    for i, path in enumerate(files, 1):
        rel = os.path.relpath(path, args.src).replace('\\', '/')
        name = rel.replace('/', '-').rsplit('.', 1)[0]
        rgb = np.array(Image.open(path).convert('RGB'))
        H, W = rgb.shape[:2]

        hands, to_orig, pad, rot = detect_hands(detector, rgb)
        if not hands:
            report.append({'file': rel, 'hands': 0, 'nails': 0, 'status': 'кисть не найдена'})
            print(f'{i:4d}/{len(files)}  {rel}: кисть не найдена', flush=True)
            continue

        predictor.set_image(rgb)
        found = []
        expected = 0
        for lm in hands:
            for p in nail_prompts(lm, to_orig):
                expected += 1
                m = segment_nail(predictor, p, W, H, rgb)
                if m is not None and m.any():
                    found.append(m)

        # Ногти одной кисти сопоставимы по размеру. Грубый выброс — это почти
        # всегда не ноготь, а пятно, случайно прошедшее проверки.
        if len(found) >= 4:
            areas = np.array([int(m.sum()) for m in found], dtype=float)
            med = float(np.median(areas))
            found = [m for m, a in zip(found, areas)
                     if 0.30 * med <= a <= 2.5 * med]

        full = np.zeros((H, W), dtype=bool)
        for m in found:
            full |= m
        nails = len(found)

        if nails == 0:
            report.append({'file': rel, 'hands': len(hands), 'nails': 0, 'status': 'ногти не выделились'})
            print(f'{i:4d}/{len(files)}  {rel}: ногти не выделились', flush=True)
            continue

        mask = (full.astype(np.uint8) * 255)
        # Смыкаем щели внутри ногтя и убираем крошку по краю.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cv2.imwrite(os.path.join(args.out, 'masks', name + '.png'), mask)
        Image.fromarray(rgb).save(os.path.join(args.out, 'images', name + '.jpg'), quality=92)

        ov = rgb.copy()
        ov[mask > 0] = (0.45 * ov[mask > 0] + 0.55 * np.array([255, 40, 90])).astype(np.uint8)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cont, -1, (255, 255, 255), 2)
        Image.fromarray(ov).save(os.path.join(args.out, 'overlay', name + '.jpg'), quality=88)

        # Полным считаем кадр, где найдены все ногти всех найденных кистей.
        # Неполный кадр учит модель, что пропущенный ноготь — это фон, а такой
        # пример вреднее, чем его отсутствие. В обучение идут только полные.
        complete = nails == expected and expected > 0
        report.append({'file': rel, 'hands': len(hands), 'nails': nails,
                       'expected': expected, 'complete': complete,
                       'area': int(mask.sum() // 255), 'status': 'ok'})
        print(f'{i:4d}/{len(files)}  {rel}: ногтей {nails}', flush=True)

    ok = sum(1 for r in report if r['status'] == 'ok')
    five = sum(1 for r in report if r.get('complete'))
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump({'total': len(files), 'ok': ok, 'complete': five,
                   'items': report}, f, ensure_ascii=False, indent=1)
    print(f'\nГотово: {ok} из {len(files)} размечено, из них {five} с пятью и более ногтями')


if __name__ == '__main__':
    main()

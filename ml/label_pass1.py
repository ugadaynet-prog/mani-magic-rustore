"""Первый проход разметки: точки ставит человек, контур даёт SAM.

Инструмент для того, кто размечает не мышкой, а списком координат — то есть
для меня. Логика та же, что в label_tool.html, и файлы получаются те же
(instances/masks/meta), поэтому потом любой кадр открывается в браузерном
инструменте и правится руками.

    python label_pass1.py --grid 01 02 03      # фото с координатной сеткой
    python label_pass1.py --mark points.json   # прогнать SAM и сохранить
    python label_pass1.py --check 01 02        # покраска для проверки глазом
    python label_pass1.py --sheet              # контактный лист всех готовых

points.json: {"01": [[0.42, 0.13], [0.35, 0.28]], ...} — доли ширины и
высоты. Точка = «здесь ноготь». Можно передать четыре числа [x, y, x2, y2] —
тогда это рамка вокруг ногтя, она надёжнее там, где ногти слиплись.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import label_tool as L
import synth

PAINT = np.array([0.05, 0.78, 0.45], dtype=np.float32)

# Грубые границы правдоподобия для ногтя, в долях площади кадра. Верхняя
# нарочно щедрая: на макро-кадре (13, 16, 18) один ноготь занимает до 5%
# кадра, и жёсткие 3% отбраковывали как раз правильные варианты. Отсев
# «палец целиком» и «вся кисть» делает не она, а второй проход в choose —
# по размеру относительно остальных ногтей ЭТОГО кадра.
AREA_MIN, AREA_MAX = 0.0004, 0.10
# Выпуклость нужна только точечным подсказкам — там она отсекает «палец
# целиком» и «вся кисть». Замер по настоящим ногтям: 0.62..0.85, потому что
# у свободного края маска слегка вогнутая, а по контуру всегда есть зубцы.
# Прежние 0.80 отбраковывали как раз правильные варианты на кадрах 20-23.
SOLIDITY_MIN = 0.60
# Во сколько раз вариант может отличаться от типичного ногтя кадра.
SCALE_LO, SCALE_HI = 0.25, 3.0


def photo(item):
    return Image.open(os.path.join(L.PHOTOS, item['file'])).convert('RGB')


def solidity(mask):
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return 0.0
    big = max(c, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(big))
    return float(cv2.contourArea(big) / hull) if hull else 0.0


def at_point(mask, prompt):
    """Оставить только ту связную часть маски, где стоит подсказка.

    SAM охотно соединяет два соседних ногтя тонкой перемычкой по блику. Без
    этого второй ноготь получал бы уже занятые пиксели и превращался в
    обрезок в пару сотен точек.
    """
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    if len(prompt) == 2:
        x, y = int(prompt[0]), int(prompt[1])
    else:
        x, y = int((prompt[0] + prompt[2]) / 2), int((prompt[1] + prompt[3]) / 2)
    x = min(max(x, 0), mask.shape[1] - 1)
    y = min(max(y, 0), mask.shape[0] - 1)
    v = lab[y, x]
    if v == 0:                       # подсказка вне маски — берём крупнейшее
        areas = [(int((lab == i).sum()), i) for i in range(1, n)]
        v = max(areas)[1]
    return lab == v


def variants(item, prompt, frame_px):
    """Все три варианта SAM с пометкой, похож ли вариант на ноготь.

    В рамку ставим не одну точку, а три вдоль длинной стороны. С одной
    точкой SAM отдаёт не весь ноготь, а его кусок до блика: глянцевое пятно
    для неё самостоятельный объект, и на нюдовом маникюре (кадры 15, 16, 17)
    она рвёт по нему пластину пополам. Три точки заставляют накрыть всё.
    """
    is_box = len(prompt) == 4
    if is_box:
        box = list(prompt)
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # Точки по всей длине, включая края: у френча и у двухцветного
        # маникюра свободный край — для SAM отдельный объект, и без точки
        # прямо в нём она отдаёт только «тело» ногтя до границы цвета.
        ts = (0.15, 0.32, 0.5, 0.68, 0.85)
        if (y1 - y0) >= (x1 - x0):                 # ноготь вытянут по вертикали
            pts = [[cx, y0 + (y1 - y0) * t, 1] for t in ts]
        else:
            pts = [[x0 + (x1 - x0) * t, cy, 1] for t in ts]
    else:
        box, pts = None, [[prompt[0], prompt[1], 1]]
    out = []
    for v in range(3):
        m, score, _ = L.predict(item, pts, box, v)
        m = at_point(m, prompt)
        inside = 1.0
        if is_box:
            # Режем маску рамкой. Рамка — это утверждение человека «ноготь
            # вот здесь и не дальше», и его надо соблюсти: на кадре 40
            # (чёрный хром в тёмных волосах) SAM иначе уводила краску в
            # волосы, потому что по цвету волосы от ногтя не отличаются.
            x0, y0, x1, y1 = [int(round(c)) for c in box]
            keep = np.zeros_like(m)
            keep[max(y0, 0):y1, max(x0, 0):x1] = True
            before = int(m.sum())
            m = m & keep
            inside = float(m.sum()) / before if before else 0.0
        area = int(m.sum())
        if is_box:
            # Рамка сама и есть утверждение человека «ноготь вот здесь»,
            # поэтому от варианта требуется только не вылезать из неё.
            ok = AREA_MIN <= area / frame_px <= AREA_MAX and inside >= 0.85
        else:
            ok = (AREA_MIN <= area / frame_px <= AREA_MAX
                  and solidity(m) >= SOLIDITY_MIN)
        out.append({'m': m, 'area': area, 'score': score, 'ok': ok,
                    'box': is_box, 'inside': inside})
    return out


def choose(all_variants):
    """Выбрать вариант на каждый ноготь, опираясь на весь кадр сразу.

    Поодиночке решить нельзя: по клику внутри ногтя SAM одинаково законно
    отдаёт и пластину, и палец, и два слипшихся ногтя, и уверенность у этих
    ответов близкая. Зато ногти одного кадра примерно одного размера —
    поэтому сначала берём грубую оценку типичной площади по бесспорным
    ногтям, а потом выбираем вариант, который к ней ближе.
    """
    first = []
    for vs in all_variants:
        good = [v for v in vs if v['ok']]
        first.append(min(good, key=lambda v: -v['score'])['area'] if good else None)
    known = [a for a in first if a]
    med = float(np.median(known)) if known else None

    chosen = []
    for vs in all_variants:
        good = [v for v in vs if v['ok']]
        if not good:
            chosen.append((vs[0], False))
            continue
        if good[0]['box']:
            # Рамку человек рисует вокруг ровно одного ногтя, поэтому вопрос
            # «что имелось в виду» тут уже снят, и можно взять ОБЪЕДИНЕНИЕ
            # вариантов, не вылезающих из рамки. Это лечит главную беду
            # двухцветных ногтей: чёрный низ и прозрачный верх, чёрный лак и
            # белый френч SAM считает разными объектами и отдаёт по половине
            # пластины. Границей служит сама рамка.
            fit = [v for v in good if v['inside'] >= 0.9]
            if fit:
                u = fit[0]['m'].copy()
                for v in fit[1:]:
                    u |= v['m']
                chosen.append(({'m': u, 'area': int(u.sum()),
                                'score': max(v['score'] for v in fit),
                                'box': True, 'inside': 1.0}, True))
            else:
                chosen.append((max(good, key=lambda v: v['area']), False))
        elif med:
            near = [v for v in good
                    if SCALE_LO * med <= v['area'] <= SCALE_HI * med]
            good = sorted(near or good, key=lambda v: abs(v['area'] - med))
            chosen.append((good[0], bool(near)))
        else:
            chosen.append((max(good, key=lambda v: v['score']), True))
    return chosen


def mark(points_by_id):
    items = {t['id']: t for t in L.task_items()}
    report = []
    for iid, prompts in points_by_id.items():
        it = items[iid]
        im = photo(it)
        w, h = im.size
        frame_px = w * h
        pxs = [[p[0] * w, p[1] * h] if len(p) == 2 else
               [p[0] * w, p[1] * h, p[2] * w, p[3] * h] for p in prompts]
        chosen = choose([variants(it, px, frame_px) for px in pxs])

        idx = np.zeros((h, w), np.uint8)
        doubtful, k = [], 0
        for i, (v, clean) in enumerate(chosen, 1):
            m = v['m']
            taken = int((m & (idx > 0)).sum())
            if taken > 0.5 * m.sum():
                # Эта маска почти целиком лежит на уже размеченном ногте —
                # значит две подсказки попали в один и тот же ноготь.
                doubtful.append(f'{i}: дубль')
                continue
            k += 1
            idx[m & (idx == 0)] = k
            if not clean:
                doubtful.append(f'{i}: не похоже на ноготь')
        areas = {int(v): int((idx == v).sum()) for v in np.unique(idx) if v}
        # Обрезок в разы мельче соседей — почти наверняка промах подсказки.
        if len(areas) >= 3:
            med = float(np.median(list(areas.values())))
            for v, a in areas.items():
                if a < 0.3 * med:
                    doubtful.append(f'{v}: обрезок {a} при типичных {med:.0f}')
        L.save_labels(it, png_of(idx), {
            'pass': 'first-by-claude', 'prompts': prompts,
            'doubtful': doubtful,
            'marked_at': datetime.now(timezone.utc).astimezone().isoformat(
                timespec='seconds')})
        report.append({'id': iid, 'nails': len(areas), 'areas': areas,
                       'doubtful': doubtful})
        line = f'{iid}: ногтей {len(areas)}, площади {sorted(areas.values())}'
        if doubtful:
            line += '\n    ← ' + '; '.join(doubtful)
        print(line, flush=True)
    return report


def png_of(idx):
    """Индексную карту — в base64 PNG, как её присылает браузер."""
    import base64
    import io as _io
    rgb = np.zeros((*idx.shape, 3), np.uint8)
    rgb[..., 0] = idx
    buf = _io.BytesIO()
    Image.fromarray(rgb).save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def grid_image(item, out):
    im = photo(item).copy()
    w, h = im.size
    d = ImageDraw.Draw(im, 'RGBA')
    try:
        f = ImageFont.truetype('segoeui.ttf', 13)
    except Exception:
        f = ImageFont.load_default()
    for i in range(1, 20):
        x, y = round(w * i / 20), round(h * i / 20)
        big = i % 2 == 0
        d.line([(x, 0), (x, h)], fill=(255, 255, 255, 110 if big else 45), width=1)
        d.line([(0, y), (w, y)], fill=(255, 255, 255, 110 if big else 45), width=1)
        if big:
            d.text((x + 2, 2), f'{i / 20:.1f}', fill=(255, 240, 60, 255), font=f)
            d.text((2, y + 1), f'{i / 20:.1f}', fill=(60, 230, 255, 255), font=f)
    im.save(out, quality=92)


def zoom_image(item, out, x0, y0, x1, y1, width=900, step=0.01):
    """Кусок кадра крупно и с мелкой сеткой.

    Нужен там, где ногти мелкие или слипшиеся: по общей сетке в 0.05 точку
    в такой ноготь не поставить, а мимо ногтя SAM отдаёт палец целиком.
    """
    im = photo(item)
    W, H = im.size
    crop = im.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    crop = crop.resize((width, round(crop.height * width / crop.width)),
                       Image.LANCZOS)
    d = ImageDraw.Draw(crop, 'RGBA')
    try:
        f = ImageFont.truetype('segoeui.ttf', 14)
    except Exception:
        f = ImageFont.load_default()
    i = 1
    while x0 + i * step < x1:
        fx = x0 + i * step
        x = round((fx - x0) / (x1 - x0) * crop.width)
        d.line([(x, 0), (x, crop.height)],
               fill=(255, 255, 255, 200 if i % 5 == 0 else 80))
        if i % 5 == 0:
            d.text((x + 2, 2), f'{fx:.2f}', fill=(255, 240, 60, 255), font=f)
        i += 1
    i = 1
    while y0 + i * step < y1:
        fy = y0 + i * step
        y = round((fy - y0) / (y1 - y0) * crop.height)
        d.line([(0, y), (crop.width, y)],
               fill=(255, 255, 255, 200 if i % 5 == 0 else 80))
        if i % 5 == 0:
            d.text((2, y + 1), f'{fy:.2f}', fill=(60, 230, 255, 255), font=f)
        i += 1
    crop.save(out, quality=92)


def check_image(item, out):
    """Покраска по нашей разметке + номер каждого ногтя."""
    rgb = np.asarray(photo(item))
    idx = np.asarray(Image.open(os.path.join(L.INSTANCES, f'{item["id"]}.png')))
    painted = synth.recolor(rgb.astype(np.float32) / 255.0, idx > 0, PAINT)
    im = Image.fromarray((np.clip(painted, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype('seguisb.ttf', 16)
    except Exception:
        f = ImageFont.load_default()
    for v in [int(x) for x in np.unique(idx) if x]:
        ys, xs = np.nonzero(idx == v)
        d.text((xs.mean(), ys.mean()), str(v), fill=(255, 60, 60), font=f,
               anchor='mm', stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(out, quality=92)


def sheet(out, only=None):
    import math
    items = [t for t in L.task_items()
             if os.path.exists(os.path.join(L.INSTANCES, f'{t["id"]}.png'))
             and (only is None or t['id'] in only)]
    if not items:
        raise SystemExit('нечего показывать')
    cols, cw, ch, pad = 8, 190, 250, 22
    rows = math.ceil(len(items) / cols)
    sh = Image.new('RGB', (cols * cw, rows * (ch + pad)), '#14151a')
    dr = ImageDraw.Draw(sh)
    try:
        f = ImageFont.truetype('segoeui.ttf', 12)
    except Exception:
        f = ImageFont.load_default()
    tmp = os.path.join(L.WORK, '_tmp.jpg')
    for i, t in enumerate(items):
        check_image(t, tmp)
        im = Image.open(tmp)
        im.thumbnail((cw - 10, ch - 10))
        x, y = (i % cols) * cw, (i // cols) * (ch + pad)
        sh.paste(im, (x + (cw - im.width) // 2, y + (ch - im.height) // 2))
        with open(os.path.join(L.META, f'{t["id"]}.json'), encoding='utf-8') as fh:
            m = json.load(fh)
        bad = m.get('doubtful') or []
        dr.text((x + 8, y + ch + 2), f'{t["id"]}  {m["nails"]} шт' +
                (f'  ?{",".join(map(str, bad))}' if bad else ''),
                fill='#f0a23c' if bad else '#35d0a5', font=f)
    os.remove(tmp)
    sh.save(out, quality=88)
    print(out, sh.size, f'кадров {len(items)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', nargs='+')
    ap.add_argument('--mark')
    ap.add_argument('--check', nargs='+')
    ap.add_argument('--zoom', nargs=5, metavar=('ID', 'X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--sheet', nargs='?', const='all')
    ap.add_argument('--out', default=os.path.join(L.WORK, 'pass1'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = {t['id']: t for t in L.task_items()}

    if args.grid:
        for iid in args.grid:
            p = os.path.join(args.out, f'grid-{iid}.jpg')
            grid_image(items[iid], p)
            print(p)
        return
    if args.mark:
        L.load_sam()
        with open(args.mark, encoding='utf-8') as fh:
            mark(json.load(fh))
        return
    if args.zoom:
        iid, *box = args.zoom
        p = os.path.join(args.out, f'zoom-{iid}.jpg')
        zoom_image(items[iid], p, *[float(v) for v in box])
        print(p)
        return
    if args.check:
        for iid in args.check:
            p = os.path.join(args.out, f'check-{iid}.jpg')
            check_image(items[iid], p)
            print(p)
        return
    if args.sheet:
        only = None if args.sheet == 'all' else set(args.sheet.split(','))
        sheet(os.path.join(args.out, 'проверка.jpg'), only)
        return
    ap.error('нужен один из --grid / --mark / --check / --sheet')


if __name__ == '__main__':
    main()

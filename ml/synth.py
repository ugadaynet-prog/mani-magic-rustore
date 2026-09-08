"""Синтез тёмного маникюра из уже размеченных фото.

Зачем. В наборе из 52 фото самый тёмный ноготь имеет яркость L=0.33, ниже 0.25
нет вовсе — чёрного и тёмного лака там просто не существует. Ровно на нём
модель и слепа: на тёмных кадрах она находит 0.5–1.7% площади вместо ногтей.
Это дыра в данных, а не в обучении, и закрыть её можно без единого клика
разметчика: **маски у нас уже есть**, значит цвет ногтя внутри маски можно
заменить, а маска останется верной.

Как красим. Не заливаем ровным цветом — плоская наклейка ничему не научит.
Берём СВОЙ рельеф ногтя (его собственную яркость, растянутую по его же
перцентилям) и накладываем на новый цвет: тень остаётся тенью, блик бликом.
Блик тянем к белому отдельно — лак глянцевый, и на чёрном ногте белая полоса
видна даже ярче, чем на светлом.

Край маски размываем. Идеально резкая граница цвета — это подсказка, по
которой сеть могла бы находить ногти вместо того, чтобы учить их форму.
"""
import numpy as np

# Яркость по Rec. 709 — та же формула, что в примерке цвета.
LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def hsv_to_rgb(h, s, v):
    """Векторно, без matplotlib: (n,) → (n,3) в 0..1."""
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def targets(n, rng):
    """Цвета лака со смещением в тёмный конец — туда, где у набора дыра.

    Половина — чёрный и графит (V 0.03–0.15): их в наборе нет совсем.
    Треть — тёмные насыщенные (вишня, индиго, изумруд): их тоже нет.
    Остальное — любой оттенок, чтобы не сузить набор до одних тёмных.
    """
    h = rng.random(n).astype(np.float32)
    s = (0.10 + rng.random(n) * 0.90).astype(np.float32)
    kind = rng.random(n)
    v = np.where(kind < 0.50, 0.03 + rng.random(n) * 0.12,
        np.where(kind < 0.80, 0.15 + rng.random(n) * 0.22,
                              0.25 + rng.random(n) * 0.65)).astype(np.float32)
    # У почти чёрного насыщенность бессмысленна и даёт грязный цветной шум.
    s = np.where(v < 0.10, s * 0.35, s).astype(np.float32)
    return hsv_to_rgb(h, s, v)


def _blur3(a):
    """Коробочное размытие 3×3 без scipy."""
    p = np.pad(a, 1, mode='edge')
    out = np.zeros_like(a)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / 9.0


def recolor(img, mask, color):
    """img: (H,W,3) 0..1, mask: (H,W) 0/1, color: (3,) 0..1 → (H,W,3)."""
    sel = mask > 0.5
    if sel.sum() < 20:
        return img
    lum = img @ LUM
    nail = lum[sel]
    lo, hi = np.percentile(nail, 5), np.percentile(nail, 95)
    rng_l = max(float(hi - lo), 1e-3)

    shade = np.clip((lum - lo) / rng_l, 0, 1)          # свой рельеф, 0..1
    # Блик берём по СВОЕЙ верхушке, а не по фиксированному порогу: у
    # французского маникюра белый край занимает пол-ногтя, и порог 0.88
    # раздувал его в белое пятно. Квантиль держит блик в пределах ~8% ногтя
    # независимо от того, насколько ноготь был светлым изначально.
    q = float(np.percentile(shade[sel], 92))
    spec = np.clip((shade - q) / max(1.0 - q, 1e-3), 0, 1)
    base = 0.30 + 0.70 * shade                         # тень не в ноль, иначе плоско

    new = color[None, None, :] * base[..., None]
    new = new + (1.0 - color)[None, None, :] * (spec * 0.85)[..., None]
    new = np.clip(new, 0, 1)

    soft = _blur3(mask.astype(np.float32))[..., None]
    return (img * (1 - soft) + new * soft).astype(np.float32)


def recolor_batch(X, Y, rng, p=0.55):
    """X: (B,H,W,3) 0..1, Y: (B,H,W) 0/1. Красит примерно p долю пачки."""
    out = X.copy()
    cols = targets(X.shape[0], rng)
    take = rng.random(X.shape[0]) < p
    for i in range(X.shape[0]):
        if take[i]:
            out[i] = recolor(X[i], Y[i], cols[i])
    return out


# ─────────────────────────── узоры на ногте ───────────────────────────
#
# Зачем. recolor заливает ноготь РОВНЫМ цветом, и это учит модель ровно тому,
# от чего она ломается на снимках нейл-арта: «ноготь — гладкое однотонное
# пятно, отличное по цвету от кожи». На чёрном с бантом и стразами, на нюде с
# красными завитками, на мраморе такого пятна нет, и модель не видит ничего.
#
# Здесь внутри той же маски рисуется узор. Маска не меняется ни на пиксель,
# значит разметка остаётся верной, а модели приходится опираться на форму и на
# то, что ноготь сидит на кончике пальца, — а не на однородность цвета.
#
# Все узоры сделаны на numpy: train.py не должен зависеть от cv2, иначе старый
# рабочий процесс обучения (train-nails-clean.yml) перестанет ставиться.

def _axis(rng):
    """Случайное направление и перпендикуляр к нему."""
    a = float(rng.random()) * np.pi
    return np.array([np.cos(a), np.sin(a)], np.float32)


def _proj(shape, u):
    """Проекция координат каждого пикселя на направление u, нормированная 0..1."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    p = xs * u[0] + ys * u[1]
    return (p - p.min()) / max(float(p.max() - p.min()), 1e-3)


def _lowfreq(shape, rng, cells=6):
    """Плавный шум: маленькая случайная сетка, растянутая на кадр."""
    h, w = shape
    small = rng.random((cells, cells)).astype(np.float32)
    ys = np.linspace(0, cells - 1, h)
    xs = np.linspace(0, cells - 1, w)
    y0 = np.clip(ys.astype(int), 0, cells - 2)
    x0 = np.clip(xs.astype(int), 0, cells - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = small[y0][:, x0]
    b = small[y0][:, x0 + 1]
    c = small[y0 + 1][:, x0]
    d = small[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy).astype(np.float32)


def pattern_field(shape, rng):
    """Второй цвет и маска-узор: где именно он ложится поверх основного.

    Узоры повторяют то, что реально встречается на снимках: френч и обратный
    френч, полосы, разводы под мрамор, точки и стразы, блёстки, градиент.
    Берём один-два — на настоящем маникюре их редко больше.
    """
    kinds = ['french', 'stripes', 'marble', 'dots', 'glitter', 'gradient']
    n = 1 if rng.random() < 0.65 else 2
    chosen = list(rng.choice(kinds, size=n, replace=False))
    field = np.zeros(shape, np.float32)

    for kind in chosen:
        u = _axis(rng)
        if kind == 'french':
            # Полоса у одного края ногтя — свободный край или лунка.
            t = 0.55 + float(rng.random()) * 0.30
            p = _proj(shape, u)
            band = (p > t) if rng.random() < 0.5 else (p < 1 - t)
            field = np.maximum(field, band.astype(np.float32))
        elif kind == 'stripes':
            k = 12 + float(rng.random()) * 40
            ph = float(rng.random()) * 6.28
            s = np.sin(_proj(shape, u) * k + ph)
            field = np.maximum(field, (s > float(rng.random()) * 0.6).astype(np.float32))
        elif kind == 'marble':
            nz = _lowfreq(shape, rng, cells=int(4 + rng.random() * 6))
            field = np.maximum(field, np.clip((nz - 0.45) * 4, 0, 1))
        elif kind == 'dots':
            h, w = shape
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            r = max(1.5, min(h, w) * (0.03 + float(rng.random()) * 0.05))
            # Центры считаем разом: цикл с полноразмерной сеткой на каждую
            # точку был самой дорогой операцией во всей аугментации.
            k = int(4 + rng.random() * 10)
            cy = rng.random(k) * h
            cx = rng.random(k) * w
            d = ((xs[None] - cx[:, None, None]) ** 2
                 + (ys[None] - cy[:, None, None]) ** 2)
            field = np.maximum(field, (d.min(axis=0) < r * r).astype(np.float32))
        elif kind == 'glitter':
            field = np.maximum(field, (rng.random(shape) < 0.02).astype(np.float32))
        elif kind == 'gradient':
            field = np.maximum(field, _proj(shape, u))
    return np.clip(field, 0, 1), chosen


def textured(img, mask, rng, base=None, second=None):
    """Как recolor, но внутри маски лежит узор, а не ровный цвет.

    Рельеф и блик считаются по самому ногтю — так же, как в recolor, иначе
    узор выглядел бы наклейкой, и сеть научилась бы искать наклейки.

    Считаем только внутри рамки, охватывающей ногти: они занимают несколько
    процентов кадра, и полноразмерные сетки под узор стоили 236 мс на снимок —
    больше, чем сам шаг обучения.
    """
    sel = mask > 0.5
    if sel.sum() < 40:
        return img
    ys, xs = np.nonzero(sel)
    pad = 3
    y0 = max(0, int(ys.min()) - pad); y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad); x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    if (y1 - y0) * (x1 - x0) < mask.size * 0.85:
        out = img.copy()
        out[y0:y1, x0:x1] = textured(img[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                                     rng, base, second)
        return out
    if base is None:
        base = targets(1, rng)[0]
    if second is None:
        # Второй цвет либо контрастный, либо почти белый — как в жизни: белый
        # френч, серебряные стразы, тёмный рисунок на нюде.
        second = (np.array([1, 1, 1], np.float32) if rng.random() < 0.45
                  else targets(1, rng)[0])

    lum = img @ LUM
    nail = lum[sel]
    lo, hi = np.percentile(nail, 5), np.percentile(nail, 95)
    rng_l = max(float(hi - lo), 1e-3)
    shade = np.clip((lum - lo) / rng_l, 0, 1)
    q = float(np.percentile(shade[sel], 92))
    spec = np.clip((shade - q) / max(1.0 - q, 1e-3), 0, 1)
    relief = 0.30 + 0.70 * shade

    field, _ = pattern_field(img.shape[:2], rng)
    col = (base[None, None, :] * (1 - field[..., None])
           + second[None, None, :] * field[..., None])

    new = col * relief[..., None]
    new = new + (1.0 - col) * (spec * 0.85)[..., None]
    new = np.clip(new, 0, 1)

    soft = _blur3(mask.astype(np.float32))[..., None]
    return (img * (1 - soft) + new * soft).astype(np.float32)


def _blobs(mask, side=112):
    """Разделить маску на отдельные ногти. Без cv2: его нет в чистом обучении.

    Метки распространяются итеративно по уменьшенной маске. Ногтей в кадре
    единицы, на стороне 112 они разнесены далеко, и десятка проходов хватает;
    считать связность в полном разрешении ради выбора «какие ногти взять» ни
    к чему.
    """
    h, w = mask.shape
    k = side / max(h, w)
    if k < 1:
        sh, sw = max(1, round(h * k)), max(1, round(w * k))
        iy = np.minimum((np.arange(sh) / k).astype(np.int32), h - 1)
        ix = np.minimum((np.arange(sw) / k).astype(np.int32), w - 1)
        small = mask[iy][:, ix]
    else:
        sh, sw, small = h, w, mask
    if not small.any():
        return []
    BIG = np.int32(1 << 30)
    lab = np.where(small, np.arange(small.size, dtype=np.int32).reshape(small.shape), BIG)
    for _ in range(4 * (sh + sw)):
        prev = lab
        up = np.full_like(lab, BIG); up[:-1] = lab[1:]
        dn = np.full_like(lab, BIG); dn[1:] = lab[:-1]
        lf = np.full_like(lab, BIG); lf[:, :-1] = lab[:, 1:]
        rt = np.full_like(lab, BIG); rt[:, 1:] = lab[:, :-1]
        lab = np.where(small, np.minimum.reduce([lab, up, dn, lf, rt]), BIG)
        if np.array_equal(lab, prev):
            break
    # Обратно в полный размер: индексная карта, ближайший сосед.
    if k < 1:
        by = np.minimum((np.arange(h) * k).astype(np.int32), sh - 1)
        bx = np.minimum((np.arange(w) * k).astype(np.int32), sw - 1)
        big = lab[by][:, bx]
    else:
        big = lab
    return [(big == v) & mask for v in np.unique(lab) if v != BIG]


def _grow(m, n=2):
    """Расширить маску на n пикселей сдвигами — вместо cv2.dilate."""
    out = m.copy()
    for _ in range(n):
        g = out.copy()
        g[:-1] |= out[1:]; g[1:] |= out[:-1]
        g[:, :-1] |= out[:, 1:]; g[:, 1:] |= out[:, :-1]
        out = g
    return out


def nude(img, mask, rng, share=(0.34, 0.85)):
    """Сделать ЧАСТЬ ногтей на кадре похожими на голые.

    Зачем. Замер 8 сентября: на подборке рисунков модель находит 12 ногтей из
    32, на фотографиях промахи приходятся на большой палец со светлой пластиной
    и на нюдовые ногти с рисунком. Читается это однозначно — модель идёт за
    лаком, а не за пластиной. В колоде и в CC0-наборе почти все ногти
    накрашены, и кадра «три накрашенных, один голый» там нет вовсе.

    Как. У выбранных ногтей цвет заменяется на цвет окружающей кожи, а
    собственная яркость пластины сохраняется — блик остаётся бликом, тень
    тенью, форма и край не трогаются. Маска не меняется: ноготь по-прежнему
    ноготь, просто теперь он выглядит ненакрашенным.

    Берём именно часть ногтей, а не все: кадр, где все ногти голые, у нас уже
    есть в жизни, а вот «один голый среди накрашенных» — это ровно тот случай,
    на котором модель спотыкается.
    """
    sel = mask > 0.5
    if not sel.any():
        return img
    blobs = _blobs(sel)
    if not blobs:
        return img
    take = max(1, int(round(len(blobs) * rng.uniform(*share))))
    chosen = [blobs[i] for i in rng.permutation(len(blobs))[:take]]

    lum = img @ LUM
    out = img.copy()
    for nail in chosen:
        area = int(nail.sum())
        if area < 24:
            continue
        # Кожа вокруг этого ногтя: кольцо шириной примерно в десятую его
        # ширины, из которого выброшены все ногти кадра.
        w = max(2, int(round(np.sqrt(area) * 0.35)))
        ring = _grow(nail, w) & ~_grow(sel, 1)
        if ring.sum() < 12:
            continue
        skin = np.median(img[ring], axis=0)
        own = lum[nail]
        mid = float(np.median(own))
        if mid < 1e-3:
            continue
        # Пластина обычно чуть светлее и розовее кожи, а не копия её.
        tint = np.array([rng.uniform(1.02, 1.10), 1.0, rng.uniform(0.96, 1.02)],
                        np.float32)
        base = np.clip(skin * tint * rng.uniform(1.00, 1.10), 0, 1)
        shade = np.clip(lum[nail] / mid, 0.55, 1.5)[:, None]
        out[nail] = np.clip(base[None, :] * shade, 0, 1)

    # Граница смягчается, как и в остальных синтезах: идеально резкий край —
    # подсказка, по которой сеть находила бы ноготь вместо того, чтобы учить.
    any_chosen = np.zeros_like(sel)
    for nail in chosen:
        any_chosen |= nail
    soft = _blur3(any_chosen.astype(np.float32))[..., None]
    return (img * (1 - soft) + out * soft).astype(np.float32)

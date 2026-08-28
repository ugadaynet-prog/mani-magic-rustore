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

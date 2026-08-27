"""Готовит данные к обучению: zip → квадраты 256×256 и бинарные маски.

Маски в наборе лежат в JPG, а не в PNG: сжатие с потерями размыло границы, и
значения там не 0/255, а вся шкала. Поэтому бинаризуем порогом. Порог 128 —
середина: артефакты сжатия вокруг границы уходят в фон, сам ноготь остаётся.
"""
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
THRESH = 128


def load_pairs(zip_path):
    z = zipfile.ZipFile(zip_path)
    imgs = {os.path.splitext(os.path.basename(n))[0]: n
            for n in z.namelist() if n.startswith('images/') and not n.endswith('/')}
    labs = {os.path.splitext(os.path.basename(n))[0]: n
            for n in z.namelist() if n.startswith('labels/') and not n.endswith('/')}
    keys = sorted(set(imgs) & set(labs))
    for k in keys:
        im = Image.open(io.BytesIO(z.read(imgs[k]))).convert('RGB')
        mk = Image.open(io.BytesIO(z.read(labs[k]))).convert('L')
        yield k, im, mk


def square(im, mk):
    """Приводим к квадрату, не растягивая: дополняем короткую сторону.

    Растяжение исказило бы форму ногтя, а она — главное, что мы учим.
    """
    w, h = im.size
    side = max(w, h)
    pad_im = Image.new('RGB', (side, side), (0, 0, 0))
    pad_mk = Image.new('L', (side, side), 0)
    off = ((side - w) // 2, (side - h) // 2)
    pad_im.paste(im, off)
    pad_mk.paste(mk, off)
    return (pad_im.resize((SIZE, SIZE), Image.BILINEAR),
            pad_mk.resize((SIZE, SIZE), Image.NEAREST))


def main():
    zip_path = os.path.join(HERE, 'data', 'nails.zip')
    xs, ys, names = [], [], []
    for k, im, mk in load_pairs(zip_path):
        im, mk = square(im, mk)
        m = (np.asarray(mk) > THRESH).astype(np.uint8)
        xs.append(np.asarray(im, dtype=np.uint8))
        ys.append(m)
        names.append(k)

    X = np.stack(xs)
    Y = np.stack(ys)
    out = os.path.join(HERE, 'data', 'prepared.npz')
    np.savez_compressed(out, X=X, Y=Y, names=np.array(names))

    fg = float(Y.mean())
    empty = int((Y.sum(axis=(1, 2)) == 0).sum())
    print(json.dumps({
        'пар': int(X.shape[0]),
        'размер': list(X.shape[1:]),
        'доля ногтей': round(fg * 100, 2),
        'пустых масок': empty,
        'файл': out,
        'мегабайт': round(os.path.getsize(out) / 1e6, 1),
    }, ensure_ascii=False, indent=2))
    if empty:
        raise SystemExit('после бинаризации %d масок оказались пустыми — порог неверен' % empty)


if __name__ == '__main__':
    main()

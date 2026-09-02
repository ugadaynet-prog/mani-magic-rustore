"""Показ разметки так, как её увидит пользователь: перекраской ногтя.

Зачем не полупрозрачная маска. Судить о маске по розовой заливке нельзя —
глаз не отличает «чуть неточный контур» от «краска села на кожу», а для
приёмки важно ровно второе. Поэтому рисуем то, ради чего маска и нужна:
перекрашиваем ноготь тем же кодом, которым красит примерка (synth.recolor),
в цвет, которого на коже не бывает. Тогда ляп виден мгновенно — покрашенный
палец или непокрашенный ноготь, — а полпикселя по краю не мозолят глаз
и правильно: на них сеть всё равно усредняет.
"""
import argparse
import os

import numpy as np
from PIL import Image

import synth

# Изумрудный: на коже, ткани и цветах такого не бывает, поэтому промах
# читается сразу. Красный или розовый для этого не годятся — сливаются.
PAINT = np.array([0.05, 0.78, 0.45], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, help='папка с images/ и masks/')
    ap.add_argument('--out', default=None, help='куда класть (по умолчанию <ds>/painted)')
    ap.add_argument('--width', type=int, default=0, help='ужать по ширине')
    args = ap.parse_args()

    img_dir = os.path.join(args.ds, 'images')
    mask_dir = os.path.join(args.ds, 'masks')
    out = args.out or os.path.join(args.ds, 'painted')
    os.makedirs(out, exist_ok=True)

    n = 0
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        mp = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mp):
            continue
        img = Image.open(os.path.join(img_dir, f)).convert('RGB')
        mk = Image.open(mp).convert('L')
        if mk.size != img.size:
            mk = mk.resize(img.size, Image.NEAREST)
        if args.width:
            h = max(1, round(img.height * args.width / img.width))
            img = img.resize((args.width, h), Image.LANCZOS)
            mk = mk.resize((args.width, h), Image.NEAREST)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        m = (np.asarray(mk) > 127).astype(np.float32)
        painted = synth.recolor(arr, m, PAINT)
        Image.fromarray((painted * 255).astype(np.uint8)).save(
            os.path.join(out, stem + '.jpg'), quality=88)
        n += 1
    print(f'перекрашено: {n} -> {out}')


if __name__ == '__main__':
    main()

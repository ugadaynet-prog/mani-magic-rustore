#!/usr/bin/env python3
"""Сборка обучающего набора ТОЛЬКО из данных, которые мы вправе использовать.

Чем отличается от prepare_dataset.py. Тот тянет шесть наборов с GitHub, и
лицензия есть лишь у одного: vpapenko, CC0. Пять остальных — PaulinaPacyna,
Zea-Zee, behroozmrd47, Golbstein, ademakdogan — выложены без лицензии вообще,
то есть остаются в собственности авторов. Веса, обученные на них, невозможно
объяснить покупателю при продаже приложения, поэтому здесь их нет.

Что берём:
  1. vpapenko/nails-segmentation-dataset — 52 фото, CC0-1.0, маски авторские.
  2. Колода MANI Magic — работы, сгенерированные нами, права наши; маски
     сделаны label_deck.py через MediaPipe и MobileSAM (обе Apache-2.0).

На выходе dataset_merged/{images,masks} в том же виде, что ждёт augment и
train, плюс manifest.json — откуда взялся каждый кадр. Манифест нужен не
только для порядка: на технической проверке при продаже происхождение
обучающих данных спрашивают первым делом.
"""
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

OUT_IMG = 'dataset_merged/images'
OUT_MASK = 'dataset_merged/masks'

VPAPENKO_URL = ('https://github.com/vpapenko/nails-segmentation-dataset/'
                'raw/master/nails_segmentation.zip')
DECK_ZIP = os.environ.get('DECK_ZIP', 'deck-dataset.zip')

count = 0
seen = set()
manifest = []


def add_pair(img, mask_arr, source, origin, group=None):
    """Кладёт пару в набор. mask_arr — булев массив того же размера, что фото.

    group — ключ, по которому кадры считаются связанными. Пять работ одной
    карты сняты по-разному, но принадлежат одной цветовой семье; если развести
    их между обучением и проверкой, IoU выйдет чуть завышенным. Дешевле
    откладывать карту целиком.
    """
    global count
    if not mask_arr.any():
        return False
    h = hashlib.md5(img.tobytes()).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    name = f'{count:05d}'
    img.convert('RGB').save(f'{OUT_IMG}/{name}.jpg', quality=95)
    Image.fromarray(mask_arr.astype(np.uint8) * 255, 'L').save(f'{OUT_MASK}/{name}.png')
    manifest.append({'name': name, 'source': source, 'origin': origin,
                     'group': group or origin})
    count += 1
    return True


def binarize(mask_img):
    arr = np.array(mask_img)
    return (arr > 0).any(axis=2) if arr.ndim == 3 else arr > 0


def take_vpapenko():
    """52 фото под CC0. Внутри архива images/ и labels/ с одинаковыми именами."""
    if not os.path.exists('vpapenko.zip'):
        print(f'Скачиваю {VPAPENKO_URL}')
        urllib.request.urlretrieve(VPAPENKO_URL, 'vpapenko.zip')
    n = 0
    with zipfile.ZipFile('vpapenko.zip') as z:
        names = [p for p in z.namelist() if not p.endswith('/')]

        def pick(folder):
            # В архиве папки лежат в корне, но бывает и обёртка верхнего уровня —
            # берём и то и другое.
            return {os.path.splitext(os.path.basename(p))[0]: p for p in names
                    if p.startswith(folder + '/') or ('/' + folder + '/') in p}

        imgs, labs = pick('images'), pick('labels')
        for k in sorted(set(imgs) & set(labs)):
            with z.open(imgs[k]) as f:
                im = Image.open(f).convert('RGB').copy()
            with z.open(labs[k]) as f:
                mk = Image.open(f).copy()
            if mk.size != im.size:
                mk = mk.resize(im.size, Image.NEAREST)
            if add_pair(im, binarize(mk), 'vpapenko-cc0', k):
                n += 1
    print(f'vpapenko (CC0-1.0): {n}')
    return n


def take_deck():
    """Колода: images/ и masks/ с совпадающими именами. Берётся из папки
    ml/deck_dataset/ либо из архива deck-dataset.zip рядом."""
    root = 'deck_dataset'
    if not os.path.isdir(root):
        if not os.path.exists(DECK_ZIP):
            print(f'{DECK_ZIP} не найден — колода пропущена')
            return 0
        print(f'Распаковываю {DECK_ZIP}')
        with zipfile.ZipFile(DECK_ZIP) as z:
            z.extractall(root)
    img_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')
    if not os.path.isdir(img_dir):
        print('в колоде нет images/ — пропускаю')
        return 0

    # Берём только кадры, размеченные полностью. Кадр, где найдены 3 ногтя из
    # пяти, учит модель, что два оставшихся — фон; такой пример хуже, чем его
    # отсутствие. Неполные отложены до следующего круга: их разметит уже
    # обученная модель, и получится это у неё лучше, чем у эвристики.
    complete = None
    rep_path = os.path.join(root, 'report.json')
    if os.path.exists(rep_path) and os.environ.get('DECK_ALL') != '1':
        with open(rep_path, encoding='utf-8') as f:
            rep = json.load(f)
        complete = {i['file'].replace('/', '-').rsplit('.', 1)[0]
                    for i in rep['items'] if i.get('complete')}
        print(f'в отчёте полных кадров: {len(complete)} из {rep["total"]}')

    n = skipped_partial = 0
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        if complete is not None and stem not in complete:
            skipped_partial += 1
            continue
        mp = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mp):
            continue
        im = Image.open(os.path.join(img_dir, f)).convert('RGB')
        mk = Image.open(mp)
        if mk.size != im.size:
            mk = mk.resize(im.size, Image.NEAREST)
        # Имя вида card-07-3: карта 7, работа 3. Группа — карта целиком.
        card = stem.rsplit('-', 1)[0] if '-' in stem else stem
        if add_pair(im, binarize(mk), 'mani-magic-deck', stem, group=card):
            n += 1
    print(f'колода MANI Magic (наша генерация): {n}'
          + (f', неполных отложено: {skipped_partial}' if skipped_partial else ''))
    return n


def main():
    shutil.rmtree('dataset_merged', ignore_errors=True)
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_MASK, exist_ok=True)

    take_vpapenko()
    take_deck()

    by_source = {}
    for m in manifest:
        by_source[m['source']] = by_source.get(m['source'], 0) + 1
    with open('dataset_merged/manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'total': count, 'by_source': by_source, 'items': manifest},
                  f, ensure_ascii=False, indent=1)
    # Отдельным файлом — только то, что нужно train.py для честного деления.
    with open('dataset_merged/groups.json', 'w', encoding='utf-8') as f:
        json.dump({m['name'] + '.jpg': m['group'] for m in manifest},
                  f, ensure_ascii=False, indent=1)

    print(f'\nВсего пар: {count}')
    for k, v in sorted(by_source.items()):
        print(f'  {k}: {v}')
    if count == 0:
        raise SystemExit('Набор пуст — обучать не на чем.')


if __name__ == '__main__':
    main()

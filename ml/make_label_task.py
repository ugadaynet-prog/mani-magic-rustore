"""Собрать задание на ручную разметку — «честный экзамен» из 40 фотографий.

Зачем это вообще нужно. До сих пор все цифры качества (IoU 0.83 и прочие)
меряны против автоматических масок MediaPipe+SAM, то есть против того же
брака, который модель и повторяет. Эталона, которому можно верить, нет ни
одного. Поэтому любой замер «стало лучше» может оказаться шумом, а телефон
показывает совсем другое, чем таблица метрик.

Экзамен собирается из двух непересекающихся частей — и считать их надо
раздельно, иначе лёгкая половина замажет тяжёлую:

  контроль (23) — снимки из «тест-распознавания», ровно тот сценарий, ради
      которого приложение существует: человек фотографирует свою руку.
      Небольшие, шумные, сжатые. Именно на них модель и провалилась.

  студия (17) — из «ВК-49-фото-новые», наша собственная генерация, крупные
      и чистые. Нужны, чтобы отделить «модель не видит ноготь» от «модель не
      справляется с плохим кадром».

Ни одна из 40 не участвовала в обучении. Для студийных это проверяется
перцептивным хэшем против dataset_merged: у нашей генерации кадры похожи
между собой, и на глаз дубликат не отличить. Всё, что ближе HAMMING_MIN к
любому обучающему кадру, из экзамена выбрасывается.

    python make_label_task.py
"""
import collections
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTROL_DIR = os.path.join(DECK, 'тест-распознавания')
STUDIO_DIR = os.path.join(DECK, 'ВК-49-фото-новые')
TRAIN_DIR = os.path.join(HERE, 'dataset_merged', 'images')
WORK = os.path.join(HERE, 'labels')

STUDIO_N = 17
# Не фотографии рук, а подборки нарисованных дизайнов. В контрольном наборе
# такая одна, и в общий счёт она идти не должна: модель учили находить ноготь
# на пальце, а тут тридцать рисунков на бумаге и ни одной руки. Кадр остаётся
# в задании отдельной частью — размечать его в последнюю очередь или не
# размечать вовсе, на число «найдено ногтей» он не повлияет.
NOT_HANDS = {'02-красный.jpg'}
# Расстояние Хэмминга (из 144 бит) до ближайшего обучающего кадра. Замер по
# всем 49 студийным: минимум 7, медиана 23 — то есть у самых близких пар
# совпадает почти вся структура кадра. Порог 20 отсекает их с запасом.
HAMMING_MIN = 20
# Длинная сторона рабочей копии. Разметка ведётся и хранится в этом размере:
# и SAM, и человек работают с одним и тем же пикселем, пересчётов нет.
WORK_SIDE = 1024
EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def ahash(path, n=12):
    """Перцептивный хэш: картинка в n×n серого, бит = «ярче среднего»."""
    a = np.asarray(Image.open(path).convert('L').resize((n, n), Image.BILINEAR),
                   dtype=np.float32)
    return (a > a.mean()).ravel()


def images_in(d):
    return sorted(f for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in EXT)


def fit(path, dest, side=WORK_SIDE):
    im = Image.open(path).convert('RGB')
    k = side / max(im.size)
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(dest, quality=95, subsampling=0)
    return im.size


def pick_studio(files, train_hashes):
    """Отобрать STUDIO_N студийных кадров, не пересекающихся с обучением.

    Из уцелевших берём равномерно по списку, а не самые далёкие подряд:
    «самые далёкие от обучения» — это кадры с нетипичной композицией, и
    экзамен из них получился бы смещённым в другую сторону.
    """
    ok, dropped = [], []
    for f in files:
        h = ahash(os.path.join(STUDIO_DIR, f))
        d = min(int((h != v).sum()) for v in train_hashes)
        (ok if d >= HAMMING_MIN else dropped).append((f, d))
    if len(ok) < STUDIO_N:
        raise SystemExit(f'После отсева осталось {len(ok)} студийных, нужно {STUDIO_N}')
    step = len(ok) / STUDIO_N
    return [ok[int(i * step)] for i in range(STUDIO_N)], dropped


if __name__ == '__main__':
    train_hashes = [ahash(os.path.join(TRAIN_DIR, f)) for f in images_in(TRAIN_DIR)]
    print(f'обучающих кадров для сверки: {len(train_hashes)}')

    studio, dropped = pick_studio(images_in(STUDIO_DIR), train_hashes)
    print(f'студийных отсеяно как близкие к обучению: {len(dropped)} '
          f'(ближайшее {min(d for _, d in dropped) if dropped else "—"})')

    os.makedirs(os.path.join(WORK, 'photos'), exist_ok=True)
    os.makedirs(os.path.join(WORK, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(WORK, 'meta'), exist_ok=True)

    items = []
    for f in images_in(CONTROL_DIR):
        part = 'подборка' if f in NOT_HANDS else 'контроль'
        items.append({'part': part, 'src_dir': CONTROL_DIR, 'src': f, 'near': None})
    for f, d in studio:
        items.append({'part': 'студия', 'src_dir': STUDIO_DIR, 'src': f, 'near': d})

    task = []
    for i, it in enumerate(items):
        num = f'{i + 1:02d}'
        name = f'{num}-{it["part"]}-{os.path.splitext(it["src"])[0]}.jpg'
        w, h = fit(os.path.join(it['src_dir'], it['src']), os.path.join(WORK, 'photos', name))
        task.append({
            'id': num, 'file': name, 'part': it['part'],
            'origin': os.path.relpath(os.path.join(it['src_dir'], it['src']), DECK),
            'w': w, 'h': h,
            'hamming_to_train': it['near'],
        })

    with open(os.path.join(WORK, 'task.json'), 'w', encoding='utf-8') as fh:
        json.dump({'work_side': WORK_SIDE, 'hamming_min': HAMMING_MIN,
                   'items': task}, fh, ensure_ascii=False, indent=1)

    by = collections.Counter(t['part'] for t in task)
    parts = ', '.join(f'{v} {k}' for k, v in by.most_common())
    print(f'\nГотово: {len(task)} фото в {os.path.relpath(WORK, HERE)}/photos — {parts}')
    print('Дальше:  python label_tool.py')

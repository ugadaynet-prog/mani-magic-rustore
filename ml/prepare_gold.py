"""Разложить ручную разметку на обучение и экзамен.

225 ногтей на 39 кадрах — единственная разметка в проекте, обведённая руками.
Соблазн отдать её в обучение целиком велик, но тогда проверять модель снова
будет нечем, и мы вернёмся ровно туда, откуда пришли: к цифрам, которые
ничего не значат. Поэтому треть кадров откладывается и в обучении не
участвует НИКОГДА.

Делим внутри каждой части отдельно (контроль, студия): иначе жребий может
отправить в экзамен одни студийные кадры, а они заметно легче, и оценка
поедет. Кадр 02 (подборка рисунков, ногтей в эталоне нет) не участвует.

    python prepare_gold.py

Пишет dataset_gold/{images,masks} и dataset_gold/split.json.
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'labels')
OUT = os.path.join(HERE, 'dataset_gold')

VAL_FRAC = 1 / 3
SEED = 11


def main():
    with open(os.path.join(WORK, 'task.json'), encoding='utf-8') as fh:
        items = json.load(fh)['items']

    usable = []
    for t in items:
        mp = os.path.join(WORK, 'meta', f'{t["id"]}.json')
        if not os.path.exists(mp):
            continue
        with open(mp, encoding='utf-8') as fh:
            m = json.load(fh)
        if not m.get('nails'):
            continue
        usable.append(dict(t, nails=m['nails']))

    rng = np.random.default_rng(SEED)
    train, val = [], []
    for part in sorted({t['part'] for t in usable}):
        grp = [t for t in usable if t['part'] == part]
        k = round(len(grp) * VAL_FRAC)
        pick = set(rng.permutation(len(grp))[:k].tolist())
        for i, t in enumerate(grp):
            (val if i in pick else train).append(t)
    train.sort(key=lambda t: t['id'])
    val.sort(key=lambda t: t['id'])

    for sub in ('images', 'masks', 'instances'):
        d = os.path.join(OUT, sub)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for t in train + val:
        shutil.copyfile(os.path.join(WORK, 'photos', t['file']),
                        os.path.join(OUT, 'images', f'{t["id"]}.jpg'))
        shutil.copyfile(os.path.join(WORK, 'masks', f'{t["id"]}.png'),
                        os.path.join(OUT, 'masks', f'{t["id"]}.png'))
        # Номера ногтей нужны проверке: она считает не пиксели, а сколько
        # ЭТАЛОННЫХ ногтей модель накрыла — это и есть жалоба пользователя.
        shutil.copyfile(os.path.join(WORK, 'instances', f'{t["id"]}.png'),
                        os.path.join(OUT, 'instances', f'{t["id"]}.png'))

    split = {'seed': SEED,
             'train': [t['id'] for t in train],
             'val': [t['id'] for t in val],
             'by_part': {p: {'train': [t['id'] for t in train if t['part'] == p],
                             'val': [t['id'] for t in val if t['part'] == p]}
                         for p in sorted({t['part'] for t in usable})}}
    with open(os.path.join(OUT, 'split.json'), 'w', encoding='utf-8') as fh:
        json.dump(split, fh, ensure_ascii=False, indent=1)

    for name, grp in (('обучение', train), ('экзамен', val)):
        by = {}
        for t in grp:
            by[t['part']] = by.get(t['part'], 0) + 1
        print(f'{name}: {len(grp)} кадров, {sum(t["nails"] for t in grp)} ногтей '
              f'({", ".join(f"{v} {k}" for k, v in sorted(by.items()))})')
    print(f'\nЭкзаменационные кадры: {", ".join(t["id"] for t in val)}')


if __name__ == '__main__':
    main()

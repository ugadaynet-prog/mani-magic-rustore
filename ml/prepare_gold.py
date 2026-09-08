"""Разложить ручную разметку на обучение и экзамен.

225 ногтей на 39 кадрах — единственная разметка в проекте, обведённая руками.
Соблазн отдать её в обучение целиком велик, но тогда проверять модель снова
будет нечем, и мы вернёмся ровно туда, откуда пришли: к цифрам, которые
ничего не значат. Поэтому треть кадров откладывается и в обучении не
участвует НИКОГДА.

Делим внутри каждой части отдельно (контроль, студия): иначе жребий может
отправить в экзамен одни студийные кадры, а они заметно легче, и оценка
поедет. Часть «подборка» (кадр 02, рисованный коллаж) в обучение не идёт —
см. SKIP_PARTS.

    python prepare_gold.py
    python prepare_gold.py --round2      # добавить годные кадры второго круга

Пишет dataset_gold/{images,masks,instances} и dataset_gold/split.json.

Второй круг (--round2) добавляется ТОЛЬКО в обучение. Отложенные тринадцать
кадров не меняются никогда: иначе «до» и «после» станут несравнимы, а вся
затея с эталоном была ради сравнимости.

Из второго круга берётся только разметка, сделанная руками. Машинную
проверили замером 6 сентября: 81 кадр, размеченный конвейером «модель +
MediaPipe + SAM», не дал ничего — полнота осталась 87.7%, форма упала с
0.843 до 0.825. Причина в том, что подсказки ставила сама модель, и там, где
она слепа, разметка утверждает, что ногтя нет. Флаг --with-machine оставлен,
чтобы повторить тот замер, но по умолчанию такие кадры не берутся.
"""
import argparse
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'labels')
OUT = os.path.join(HERE, 'dataset_gold')

VAL_FRAC = 1 / 3
SEED = 11
# Части, которые в обучение не идут никогда.
#   подборка — рисованные ногти с чужого коллажа, а не фотография. В экзамене
#     она нужна: такие картинки приносят в приложение, и надо знать, что с ними
#     происходит. Учить на рисунках фотомодель незачем.
#   контроль — чужие снимки из открытых источников, см. PROVENANCE.md. Здесь
#     они всё ещё попадают в dataset_gold: отбрасывает их finetune.py, чтобы
#     старый замер можно было повторить ключом --with-control.
SKIP_PARTS = {'подборка'}
WORK2 = os.path.join(HERE, 'labels2')
# Область, которая вдвое с лишним крупнее соседей, — не ноготь, а клякса.
BLOWN = 2.2


def round2_ids(with_machine=False):
    """Кадры второго круга, годные для обучения, и почему они годны."""
    import glob
    good, skipped = [], 0
    for p in sorted(glob.glob(os.path.join(WORK2, 'meta', '*.json'))):
        with open(p, encoding='utf-8') as fh:
            m = json.load(fh)
        areas = list(m['areas'].values())
        if not areas:
            skipped += 1
            continue
        by_hand = m.get('pass') != 'second-by-claude'
        if by_hand:
            good.append(m['id'])
            continue
        blown = len(areas) >= 3 and max(areas) > BLOWN * float(np.median(areas))
        if with_machine and m['nails'] >= 5 and not blown:
            good.append(m['id'])
        else:
            skipped += 1
    return good, skipped


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
        if not m.get('nails') or t['part'] in SKIP_PARTS:
            continue
        usable.append(dict(t, nails=m['nails']))

    ap = argparse.ArgumentParser()
    ap.add_argument('--round2', action='store_true',
                    help='добавить в обучение ручные кадры из labels2')
    ap.add_argument('--with-machine', action='store_true',
                    help='взять и машинную разметку второго круга (замером '
                         'проверено: пользы нет)')
    args = ap.parse_args()

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

    extra = []
    if args.round2:
        ids, skipped = round2_ids(args.with_machine)
        for iid in ids:
            shutil.copyfile(os.path.join(WORK2, 'photos', f'{iid}.jpg'),
                            os.path.join(OUT, 'images', f'r2-{iid}.jpg'))
            shutil.copyfile(os.path.join(WORK2, 'masks', f'{iid}.png'),
                            os.path.join(OUT, 'masks', f'r2-{iid}.png'))
            shutil.copyfile(os.path.join(WORK2, 'instances', f'{iid}.png'),
                            os.path.join(OUT, 'instances', f'r2-{iid}.png'))
            extra.append(f'r2-{iid}')
        print(f'второй круг: взято {len(extra)}, отброшено {skipped}')

    split = {'seed': SEED,
             'round2': extra,
             'train': [t['id'] for t in train] + extra,
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

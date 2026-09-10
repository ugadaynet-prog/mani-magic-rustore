"""Добавить в обучение кадры под CC0 (labels5).

Почему отдельным скриптом, а не в prepare_gold.py. Набор dataset_gold
собирается на этой машине и уезжает в CI зашифрованным (в нём фотографии
приложения и снимки владельца). А CC0-кадры лежать в открытом репозитории
могут: права на них отданы явно, шифровать нечего. Поэтому в CI они попадают
не через архив, а прямо из репозитория, и добавлять их надо к уже
распакованному набору.

    python add_cc0.py                       # добавить в dataset_gold
    python add_cc0.py --dry                 # только показать, что добавится
    python add_cc0.py --src own --prefix own- --tag own   # снимки владельца

Тот же путь годится и для снимков владельца (labels6): они приезжают в CI
зашифрованными собственным ключом OWN_KEY, расшифровываются в папку own/ и
добавляются к набору этим же скриптом.

Кадры идут ТОЛЬКО в обучение. Экзамены не трогаем: если положить их в
проверку, «до» и «после» станут несравнимы, а вся затея с эталоном была ради
сравнимости.
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'labels5')
OUT = os.path.join(HERE, 'dataset_gold')
PREFIX = 'cc0-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--prefix', default=PREFIX,
                    help='приставка к именам кадров в наборе (cc0-, own-)')
    ap.add_argument('--tag', default='cc0',
                    help='под каким ключом записать список в split.json')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    inst = os.path.join(args.src, 'instances')
    if not os.path.isdir(inst):
        raise SystemExit(f'нет {inst} — сначала разметка')
    ids = sorted(f[:-4] for f in os.listdir(inst) if f.endswith('.png'))
    if not ids:
        raise SystemExit('в labels5 нет ни одной размеченной маски')

    split_path = os.path.join(args.out, 'split.json')
    with open(split_path, encoding='utf-8') as fh:
        split = json.load(fh)

    added = []
    for iid in ids:
        name = args.prefix + iid
        if name in split['train']:
            continue
        if not args.dry:
            shutil.copyfile(os.path.join(args.src, 'photos', f'{iid}.jpg'),
                            os.path.join(args.out, 'images', f'{name}.jpg'))
            shutil.copyfile(os.path.join(args.src, 'masks', f'{iid}.png'),
                            os.path.join(args.out, 'masks', f'{name}.png'))
            shutil.copyfile(os.path.join(args.src, 'instances', f'{iid}.png'),
                            os.path.join(args.out, 'instances', f'{name}.png'))
        added.append(name)

    if not args.dry:
        split['train'] = split['train'] + added
        split[args.tag] = added
        with open(split_path, 'w', encoding='utf-8') as fh:
            json.dump(split, fh, ensure_ascii=False, indent=1)

    print(f'{"добавилось бы" if args.dry else "добавлено"} {len(added)} кадров: '
          f'{", ".join(added)}')
    print(f'в обучении станет {len(split["train"]) + (len(added) if args.dry else 0)} кадров')


if __name__ == '__main__':
    main()

"""Дообучение на ручной разметке.

Отдельный скрипт, а не режим train.py: там обучение с нуля на автоматических
масках, здесь — правка готовой модели по 26 кадрам, обведённым руками. Смешивать
это в одном файле значит запутать оба.

Три вещи, которые тут сделаны иначе, и каждая — из замера, а не из общих
соображений.

  Проверка идёт по ногтям, а не по пикселям. Считаем долю ЭТАЛОННЫХ ногтей,
  которые модель накрыла хотя бы наполовину, — ровно то, на что жаловался
  владелец («ногти пропускает»). IoU остаётся, но вторым номером: модель может
  красиво обводить три ногтя и не видеть два, и IoU этого не покажет.

  Ручные кадры повторяются в эпохе GOLD_REPEAT раз. Их 26 против 239
  автоматических, и без этого правильная разметка утонула бы в неправильной.
  С повтором она составляет примерно половину эпохи.

  Автоматические кадры всё же остаются. На 26 кадрах сеть в миллион с лишним
  весов переобучится за десяток эпох; автоматические держат общие признаки,
  ручные исправляют систематическую ошибку разметки.

    python finetune.py --init best.pt --epochs 60
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import exam
import train as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, 'dataset_gold')
AUTO = os.path.join(HERE, 'dataset_merged')

GOLD_REPEAT = 8
LR = 5e-5            # дообучение, а не обучение: шаг на порядок меньше
LR_MIN = 1e-6
BATCH = 4
DARK_P = 0.35


def gold_split():
    with open(os.path.join(GOLD, 'split.json'), encoding='utf-8') as fh:
        return json.load(fh)


def val_score(net, ids, size):
    """Прогнать отложенные кадры ровно так, как это делает приложение.

    Не переиспользуем валидацию train.py: она считает IoU по тензорам 384×384,
    а нам нужен тот же путь, что в exam.py и в плагине, — вписывание в квадрат,
    возврат маски в размер фотографии и счёт по ногтям.
    """
    net.eval()
    found = nails = stray = 0
    ious = []
    with torch.no_grad():
        for iid in ids:
            im = Image.open(os.path.join(GOLD, 'images', f'{iid}.jpg')).convert('RGB')
            gt = np.asarray(Image.open(os.path.join(GOLD, 'instances', f'{iid}.png')))
            x = np.asarray(exam.letterbox(im, size), np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            logits = net(x)[0, 0].numpy()
            pred = exam.unletterbox(1 / (1 + np.exp(-logits)) > 0.5, im.width, im.height)
            r = exam.score_frame(gt, pred)
            found += r['found']
            nails += r['nails']
            stray += r['stray']
            if r['iou'] is not None:
                ious.append(r['iou'])
    return {'recall': found / max(1, nails), 'found': found, 'nails': nails,
            'iou': float(np.mean(ious)) if ious else 0.0, 'stray': stray}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='best.pt', help='с каких весов начинать')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--out', default='best-gold.pt')
    ap.add_argument('--no-auto', action='store_true',
                    help='учиться только на ручных кадрах')
    args = ap.parse_args()

    split = gold_split()
    size = T.SIZE

    gold_ds = T.NailDataset(os.path.join(GOLD, 'images'), os.path.join(GOLD, 'masks'),
                            size, files=[f'{i}.jpg' for i in split['train']],
                            augment=True, dark_p=DARK_P)
    parts = [gold_ds] * GOLD_REPEAT
    if not args.no_auto:
        auto_files = sorted(f for f in os.listdir(os.path.join(AUTO, 'images'))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        parts.append(T.NailDataset(os.path.join(AUTO, 'images'),
                                   os.path.join(AUTO, 'masks'), size,
                                   files=auto_files, augment=True, dark_p=DARK_P))
        print(f'Автоматических кадров: {len(auto_files)}', flush=True)
    train_ds = ConcatDataset(parts)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, drop_last=True)
    print(f'Ручных кадров: {len(split["train"])} × {GOLD_REPEAT} повторов; '
          f'в эпохе {len(train_ds)} примеров, {len(loader)} пачек', flush=True)
    print(f'Отложено на проверку: {len(split["val"])} кадров '
          f'({", ".join(split["val"])})', flush=True)

    net = T.NailNet()
    sd = torch.load(args.init, map_location='cpu', weights_only=True)
    net.load_state_dict(sd)
    print(f'Начальные веса: {args.init}', flush=True)

    base = val_score(net, split['val'], size)
    print(f'ДО дообучения: найдено {base["found"]}/{base["nails"]} = '
          f'{100 * base["recall"]:.1f}%, IoU {base["iou"]:.3f}, '
          f'лишних {base["stray"]}', flush=True)

    crit = T.BCEDiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=LR_MIN)

    best = base['recall'] + 0.1 * base['iou']
    history = [dict(epoch=0, **base)]
    print(f'Порог для сохранения: {best:.4f} (счёт = полнота + 0.1×IoU)', flush=True)

    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        v = val_score(net, split['val'], size)
        # Полнота — главное, IoU лишь разводит близкие результаты.
        score = v['recall'] + 0.1 * v['iou']
        mark = ''
        if score > best:
            best = score
            torch.save(net.state_dict(), args.out)
            mark = '  ← сохранено'
        history.append(dict(epoch=ep + 1, loss=total / max(1, len(loader)), **v))
        print(f'эпоха {ep + 1:3}/{args.epochs}  потери {total / max(1, len(loader)):.4f}  '
              f'найдено {v["found"]}/{v["nails"]} = {100 * v["recall"]:5.1f}%  '
              f'IoU {v["iou"]:.3f}  лишних {v["stray"]:2}  '
              f'{time.time() - t0:.0f} с{mark}', flush=True)
        with open('metrics-gold.json', 'w', encoding='utf-8') as fh:
            json.dump({'best': best, 'base': base, 'epochs': history}, fh,
                      ensure_ascii=False, indent=1)

    print(f'\nЛучший счёт: {best:.4f}; веса в {args.out}')


if __name__ == '__main__':
    main()

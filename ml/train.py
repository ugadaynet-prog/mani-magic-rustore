"""Обучение сегментации ногтей: U-Net с энкодером MobileNetV3-Small.

Почему так, а не проще и не сложнее:

* **Предобученный энкодер обязателен.** Картинок 52 — сеть с нуля на таком
  наборе выучит эти пятьдесят две руки и больше ничего. Веса ImageNet дают
  готовые «края, текстуры, блики», и учить остаётся только «что здесь ноготь».
* **MobileNetV3-Small, а не ResNet.** Модель поедет в телефон: ResNet18-U-Net
  это ~56 МБ fp32, MobileNetV3-Small — единицы мегабайт. Внутри приложения
  каждый мегабайт виден при установке.
* **Dice рядом с BCE.** Ногти занимают 3.6% кадра. Одна BCE на таком перекосе
  сходится к «везде фон» с прекрасной точностью и нулевой пользой; Dice считает
  пересечение с маской и на пустой ответ реагирует сразу.
* **Скипы берём по разрешению, а не по номеру слоя.** Индексы слоёв
  torchvision меняет между версиями, а разрешение — нет.
* **Тёмный маникюр досоздаём из масок.** В наборе самый тёмный ноготь имеет
  яркость 0.33, ниже 0.25 нет вовсе — и ровно на тёмном модель слепа. Маски
  есть на все 52 фото, значит цвет внутри маски можно заменить, а маска
  останется верной (`synth.py`). Разметка для этого не нужна.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import synth

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
VAL_N = 10          # отложенных картинок; на 52 больше отдавать жалко
EPOCHS = int(os.environ.get('EPOCHS', 260))
BATCH = 8
LR = 3e-4
SEED = 7

torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)


# --------------------------------------------------------------------- модель
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        # На каждом разрешении держим ПОСЛЕДНИЙ выход: это самый «зрелый»
        # признак этого масштаба, его и подаём в скип.
        feats = {}
        for layer in self.features:
            x = layer(x)
            feats[int(x.shape[-1])] = x
        return feats


class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        return self.block(torch.cat([x, skip], 1))


class NailNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        with torch.no_grad():
            feats = self.enc(torch.zeros(1, 3, SIZE, SIZE))
        self.res = sorted(feats.keys())              # напр. [8, 16, 32, 64, 128]
        ch = {r: feats[r].shape[1] for r in self.res}

        widths = [16, 24, 32, 48]
        ups, c_in = [], ch[self.res[0]]
        for i, r in enumerate(self.res[1:]):
            c_out = widths[min(i, len(widths) - 1)]
            ups.append(Up(c_in, ch[r], c_out))
            c_in = c_out
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(c_in, 1, 1)

    def forward(self, x):
        feats = self.enc(x)
        y = feats[self.res[0]]
        for up, r in zip(self.ups, self.res[1:]):
            y = up(y, feats[r])
        y = self.head(y)
        return F.interpolate(y, size=(SIZE, SIZE), mode='bilinear', align_corners=False)


# ---------------------------------------------------------------- аугментации
def augment(x, y):
    """x: (B,3,H,W) float 0..1, y: (B,1,H,W) float 0/1."""
    B = x.shape[0]

    # Сначала цвет лака, потом уже геометрия и свет: так перекрашенный ноготь
    # проходит тот же поворот и ту же засветку, что и настоящий, и мягкий край
    # маски пересемплируется вместе с картинкой.
    x = torch.from_numpy(
        synth.recolor_batch(x.permute(0, 2, 3, 1).numpy(), y[:, 0].numpy(), RNG)
    ).permute(0, 3, 1, 2).contiguous()
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    if torch.rand(1).item() < 0.3:
        x, y = torch.flip(x, [2]), torch.flip(y, [2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x, y = torch.rot90(x, k, [2, 3]), torch.rot90(y, k, [2, 3])

    # Небольшой поворот и масштаб — руку снимают под произвольным углом.
    ang = (torch.rand(B) * 2 - 1) * 0.35
    scale = 1 + (torch.rand(B) * 2 - 1) * 0.25
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    shift = (torch.rand(B, 2) * 2 - 1) * 0.12
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, shift[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, shift[:, 1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')
    y = F.grid_sample(y, grid, align_corners=False, padding_mode='zeros')
    y = (y > 0.5).float()

    # Свет и цвет. Лак бывает любого оттенка, а свет в салоне — какой угодно:
    # без этого модель привяжется к цветам конкретных пятидесяти двух фото.
    x = x * (0.75 + torch.rand(B, 1, 1, 1) * 0.5)
    x = (x - 0.5) * (0.75 + torch.rand(B, 1, 1, 1) * 0.5) + 0.5
    x = x * (0.85 + torch.rand(B, 3, 1, 1) * 0.3)
    return x.clamp(0, 1), y


# ------------------------------------------------------------------- обучение
def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def dark_val(X, Y, seed=SEED + 1):
    """Отложенные фото, перекрашенные в тёмное — детерминированно, один раз.

    Нужны потому, что в исходных отложенных тёмного нет совсем: по ним не
    видно, закрылась дыра или нет. Фиксированное зерно — чтобы метрика между
    прогонами сравнивалась, а не плавала вместе со случайным цветом.
    """
    rng = np.random.default_rng(seed)
    Xd = synth.recolor_batch(X.permute(0, 2, 3, 1).numpy(), Y[:, 0].numpy(), rng, p=1.0)
    return torch.from_numpy(Xd).permute(0, 3, 1, 2).contiguous()


def iou(logits, y, t=0.5):
    p = (torch.sigmoid(logits) > t).float()
    inter = (p * y).sum(dim=(1, 2, 3))
    union = ((p + y) > 0).float().sum(dim=(1, 2, 3))
    return (inter / union.clamp(min=1)).mean().item()


def main():
    d = np.load(os.path.join(HERE, 'data', 'prepared.npz'))
    X = torch.from_numpy(d['X']).permute(0, 3, 1, 2).float() / 255
    Y = torch.from_numpy(d['Y']).unsqueeze(1).float()

    idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))
    val_i, tr_i = idx[:VAL_N], idx[VAL_N:]
    Xtr, Ytr, Xva, Yva = X[tr_i], Y[tr_i], X[val_i], Y[val_i]
    Xvd = dark_val(Xva, Yva)
    print('обучающих %d, отложенных %d (+ те же %d в тёмном)'
          % (len(tr_i), len(val_i), len(val_i)), flush=True)

    net = NailNet()
    params = sum(p.numel() for p in net.parameters())
    print('параметров: %.2f млн' % (params / 1e6), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    # Перекос классов: положительных пикселей 3.6%, поэтому в BCE им вес.
    pos_w = torch.tensor([(1 - Ytr.mean()) / Ytr.mean().clamp(min=1e-6)])
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w.clamp(max=20))

    best, best_dark, best_ep, hist = 0.0, 0.0, -1, []
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(Xtr.shape[0])
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb, yb = augment(Xtr[b], Ytr[b])
            logits = net(xb)
            loss = 0.5 * bce(logits, yb) + 0.5 * dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()

        if ep % 5 == 4 or ep == EPOCHS - 1:
            net.eval()
            with torch.no_grad():
                v = iou(net(Xva), Yva)
                vd = iou(net(Xvd), Yva)
            # Лучшую эпоху выбираем по обеим сразу. По одной светлой нельзя:
            # она не заметит, что тёмное так и не нашлось.
            score = (v + vd) / 2
            hist.append({'эпоха': ep + 1, 'потери': round(tot / len(perm), 4),
                         'IoU': round(v, 4), 'IoU тёмный': round(vd, 4)})
            print('эпоха %3d | потери %.4f | IoU %.3f | IoU тёмный %.3f | %.0f с'
                  % (ep + 1, tot / len(perm), v, vd, time.time() - t0), flush=True)
            if score > (best + best_dark) / 2:
                best, best_dark, best_ep = v, vd, ep + 1
                torch.save(net.state_dict(), os.path.join(HERE, 'best.pt'))

    json.dump({'лучший IoU': round(best, 4), 'IoU тёмный': round(best_dark, 4),
               'на эпохе': best_ep,
               'параметров, млн': round(params / 1e6, 2),
               'эпох': EPOCHS, 'история': hist},
              open(os.path.join(HERE, 'metrics.json'), 'w'), ensure_ascii=False, indent=2)
    print('лучший IoU %.3f (тёмный %.3f) на эпохе %d' % (best, best_dark, best_ep),
          flush=True)


if __name__ == '__main__':
    main()

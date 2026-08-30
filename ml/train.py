"""
Обучение сегментации ногтей: U-Net с энкодером MobileNetV3-Small.

v2: Разрешение 512×512, чекпоинты каждые 10 эпох, контроль таймаута 280 минут.
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
SIZE = 512          # Increased from 256 to 512
VAL_N = 15          # отложенных картинок
EPOCHS = int(os.environ.get('EPOCHS', 150))
BATCH = 4           # Reduced from 8 to 4 (512x512 is 4x more pixels)
LR = 3e-4
SEED = 7
TIMEOUT_MIN = 280   # 280 minutes (GitHub Actions 300 min timeout - 20 min buffer)
CHECKPOINT_EVERY = 10  # Save checkpoint every 10 epochs

torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

START_TIME = time.time()


# --------------------------------------------------------------------- модель
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features

    def forward(self, x):
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
        self.res = sorted(feats.keys())
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


# ---------------------------------------------------------------- аугментация
def augment(x, y):
    """x: (B,3,H,W) float 0..1, y: (B,1,H,W) float 0/1."""
    B = x.shape[0]
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
    if torch.rand(1).item() < 0.4:
        g = float(torch.empty(1).uniform_(0.8, 1.25))
        x = x * g
    return x, y


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        b = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        return 0.5 * b + 0.5 * dice.mean()


def iou(preds, targets, thresh=0.5):
    preds = (torch.sigmoid(preds) > thresh).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(2, 3))
    return (inter + 1) / (union + 1)


def main():
    data = np.load(os.path.join(HERE, 'data', 'prepared.npz'))
    X, Y = data['X'], data['Y']
    names = data['names']
    n = len(X)
    print(f'Loaded {n} pairs at {X.shape[1:]}')

    # Binarize masks
    Y = (Y > 0.5).astype(np.float32)

    # Normalize images to 0..1
    X = X.astype(np.float32) / 255.0

    # Train/val split
    perm = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(perm)
    val_idx = perm[:VAL_N]
    train_idx = perm[VAL_N:]

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}')

    net = NailNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Model params: {n_params:,} ({n_params/1e6:.2f}M)')

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(os.path.join(HERE, 'best.pt')):
        net.load_state_dict(torch.load(os.path.join(HERE, 'best.pt'), map_location='cpu'))
        print('Resumed from best.pt')
    
    if os.path.exists(os.path.join(HERE, 'checkpoint.pt')):
        ckpt = torch.load(os.path.join(HERE, 'checkpoint.pt'), map_location='cpu')
        net.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from checkpoint at epoch {start_epoch}')

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    if os.path.exists(os.path.join(HERE, 'checkpoint.pt')):
        ckpt = torch.load(os.path.join(HERE, 'checkpoint.pt'), map_location='cpu')
        if 'opt' in ckpt:
            opt.load_state_dict(ckpt['opt'])

    crit = BCEDiceLoss()

    Xtr_t = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous()
    Ytr_t = torch.from_numpy(Ytr).unsqueeze(1)
    Xva_t = torch.from_numpy(Xva).permute(0, 3, 1, 2).contiguous()
    Yva_t = torch.from_numpy(Yva).unsqueeze(1)

    best = 0.0
    history = []
    
    # Load previous history if resuming
    if os.path.exists(os.path.join(HERE, 'metrics.json')):
        with open(os.path.join(HERE, 'metrics.json')) as f:
            prev = json.load(f)
        if 'epochs' in prev:
            history = prev['epochs']
            best = prev.get('best_iou', 0.0)
            print(f'Previous best IoU: {best:.4f}')

    for epoch in range(start_epoch, EPOCHS):
        # Check timeout
        elapsed_min = (time.time() - START_TIME) / 60
        if elapsed_min > TIMEOUT_MIN:
            print(f'Timeout reached at {elapsed_min:.1f} min, saving and exiting')
            break

        t0 = time.time()
        net.train()
        nb = len(Xtr_t) // BATCH
        loss_sum = 0.0
        for bi in range(nb):
            idx = slice(bi * BATCH, (bi + 1) * BATCH)
            x = Xtr_t[idx]
            y = Ytr_t[idx]
            x, y = augment(x, y)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        train_loss = loss_sum / nb

        net.eval()
        with torch.no_grad():
            out = net(Xva_t)
            val_loss = crit(out, Yva_t).item()
            val_iou = iou(out, Yva_t).mean().item()

        dt = time.time() - t0
        cur_iou = val_iou
        line = f'E{epoch+1:3d}/{EPOCHS} loss={train_loss:.4f} val={val_loss:.4f} IoU={val_iou:.4f} ({dt:.0f}s, {elapsed_min:.0f}min)'
        print(line, flush=True)
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'iou': val_iou})

        if cur_iou > best:
            best = cur_iou
            torch.save(net.state_dict(), os.path.join(HERE, 'best.pt'))
            print(f'  saved best.pt (IoU={best:.4f})', flush=True)

        # Save checkpoint every CHECKPOINT_EVERY epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'opt': opt.state_dict(),
                'best_iou': best,
            }, os.path.join(HERE, 'checkpoint.pt'))
            print(f'  saved checkpoint.pt at epoch {epoch+1}', flush=True)

        with open(os.path.join(HERE, 'metrics.json'), 'w') as f:
            json.dump({'best_iou': best, 'epochs': history}, f, indent=2)

    print(f'\nBest IoU: {best:.4f}')
    with open(os.path.join(HERE, 'metrics.json'), 'w') as f:
        json.dump({'best_iou': best, 'epochs': history}, f, indent=2)


if __name__ == '__main__':
    main()

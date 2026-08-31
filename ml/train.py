"""
Обучение сегментации ногтей: U-Net с энкодером MobileNetV3-Small.

v5: Cosine annealing LR, расширенная аугментация, разрешение 384x384.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms as T
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 384           # 384x384 — больше деталей (было 256)
VAL_N = 15
EPOCHS = int(os.environ.get('EPOCHS', 150))
BATCH = 4            # Уменьшен с 8 до 4 (384x384 = 2.25x больше пикселей)
LR = 3e-4
LR_MIN = 1e-5        # Минимальный LR для cosine annealing
SEED = 7
TIMEOUT_MIN = 280
CHECKPOINT_EVERY = 5 # Чекпоинт каждые 5 эпох (было 10)

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


# ----------------------------------------------------- Dataset: ленивая загрузка
class NailDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=384, indices=None, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        all_files = sorted(f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if indices is not None:
            self.files = [all_files[i] for i in indices]
        else:
            self.files = all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')

        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.augment:
            # Расширенная аугментация
            # 1. Случайный поворот 0/90/180/270
            k = np.random.randint(0, 4)
            if k:
                img = img.rotate(90 * k)
                mask = mask.rotate(90 * k)

            # 2. Зеркалирование
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < 0.3:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # 3. Яркость
            if np.random.random() < 0.5:
                factor = np.random.uniform(0.7, 1.3)
                img = ImageEnhance.Brightness(img).enhance(factor)

            # 4. Контраст
            if np.random.random() < 0.5:
                factor = np.random.uniform(0.7, 1.4)
                img = ImageEnhance.Contrast(img).enhance(factor)

            # 5. Насыщенность цвета
            if np.random.random() < 0.4:
                factor = np.random.uniform(0.5, 1.5)
                img = ImageEnhance.Color(img).enhance(factor)

            # 6. Резкость
            if np.random.random() < 0.3:
                factor = np.random.uniform(0.5, 2.0)
                img = ImageEnhance.Sharpness(img).enhance(factor)

            # 7. Gaussian blur (для устойчивости к фокусу камеры)
            if np.random.random() < 0.3:
                radius = np.random.uniform(0.5, 1.5)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))

            # 8. Случайный сдвиг яркости
            if np.random.random() < 0.4:
                delta = np.random.uniform(-0.1, 0.1)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                img_arr = np.clip(img_arr + delta, 0, 1)
                img = Image.fromarray((img_arr * 255).astype(np.uint8))

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask) > 127).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


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
    img_dir = os.path.join(HERE, 'dataset_aug', 'images')
    mask_dir = os.path.join(HERE, 'dataset_aug', 'masks')

    if not os.path.exists(img_dir):
        raise SystemExit('dataset_aug/ not found. Run prepare.py first.')

    n_total = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    print(f'Total augmented images: {n_total}', flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    val_idx = perm[:VAL_N].tolist()
    train_idx = perm[VAL_N:].tolist()

    train_ds = NailDataset(img_dir, mask_dir, SIZE, train_idx, augment=True)
    val_ds = NailDataset(img_dir, mask_dir, SIZE, val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}', flush=True)
    print(f'Image size: {SIZE}x{SIZE}, Batch: {BATCH}', flush=True)

    net = NailNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Model params: {n_params:,} ({n_params/1e6:.2f}M)', flush=True)

    # Resume from checkpoint
    start_epoch = 0
    best = 0.0
    history = []

    checkpoint_path = os.path.join(HERE, 'checkpoint.pt')
    best_path = os.path.join(HERE, 'best.pt')

    def is_valid_torch_file(path):
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 100:
            return False
        try:
            with open(path, 'rb') as f:
                header = f.read(20)
                return len(header) >= 2
        except:
            return False

    # Note: SIZE changed from 256 to 384, so old checkpoints won't load directly
    # (model architecture is same, but we need to load weights_only for state_dict)
    if is_valid_torch_file(best_path):
        print('Loading best.pt for fine-tuning...', flush=True)
        try:
            state_dict = torch.load(best_path, map_location='cpu', weights_only=True)
            # Try to load - architecture is same regardless of SIZE
            net.load_state_dict(state_dict)
            print('Loaded best.pt successfully (fine-tuning at new resolution)', flush=True)
        except Exception as e:
            print(f'Failed to load best.pt: {e}', flush=True)
            print('Training from scratch', flush=True)

    # Load metrics history
    metrics_path = os.path.join(HERE, 'metrics.json')
    if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 10:
        try:
            with open(metrics_path) as f:
                prev = json.load(f)
            if 'epochs' in prev:
                history = prev['epochs']
                best = prev.get('best_iou', 0.0)
                print(f'Loaded {len(history)} previous epochs, best IoU={best:.4f}', flush=True)
        except Exception as e:
            print(f'Failed to load metrics.json: {e}', flush=True)
            history = []

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR_MIN
    )
    print(f'LR: {LR} -> {LR_MIN} (cosine annealing over {EPOCHS} epochs)', flush=True)

    crit = BCEDiceLoss()

    print(f'\nStarting training from epoch {start_epoch+1} to {EPOCHS}', flush=True)
    print(f'Timeout: {TIMEOUT_MIN} minutes', flush=True)
    print(f'Augmentation: rotations, flips, brightness, contrast, color, sharpness, blur', flush=True)
    print(flush=True)

    for epoch in range(start_epoch, EPOCHS):
        elapsed_min = (time.time() - START_TIME) / 60
        if elapsed_min > TIMEOUT_MIN:
            print(f'Timeout at {elapsed_min:.1f} min, saving and exiting', flush=True)
            break

        t0 = time.time()
        net.train()
        nb = len(train_loader)
        loss_sum = 0.0
        for bi, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            if bi % 100 == 0:
                cur_lr = opt.param_groups[0]['lr']
                print(f'  E{epoch+1} B{bi}/{nb} loss={loss.item():.4f} lr={cur_lr:.6f}', flush=True)

        scheduler.step()
        train_loss = loss_sum / nb

        # Validation
        net.eval()
        val_loss_sum = 0.0
        iou_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                out = net(x)
                val_loss_sum += crit(out, y).item()
                iou_sum += iou(out, y).mean().item()

        val_loss = val_loss_sum / len(val_loader)
        val_iou = iou_sum / len(val_loader)

        dt = time.time() - t0
        cur_lr = opt.param_groups[0]['lr']
        print(f'E{epoch+1:3d}/{EPOCHS} loss={train_loss:.4f} val={val_loss:.4f} IoU={val_iou:.4f} lr={cur_lr:.6f} ({dt:.0f}s, {elapsed_min:.0f}min)', flush=True)
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'iou': val_iou, 'lr': cur_lr})

        if val_iou > best:
            best = val_iou
            torch.save(net.state_dict(), best_path)
            print(f'  saved best.pt (IoU={best:.4f})', flush=True)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'opt': opt.state_dict(),
                'best_iou': best,
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)
            print(f'  saved checkpoint.pt at epoch {epoch+1}', flush=True)

        with open(metrics_path, 'w') as f:
            json.dump({'best_iou': best, 'epochs': history}, f, indent=2)

    print(f'\nBest IoU: {best:.4f}')
    with open(metrics_path, 'w') as f:
        json.dump({'best_iou': best, 'epochs': history}, f, indent=2)


if __name__ == '__main__':
    main()

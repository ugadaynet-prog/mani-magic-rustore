"""
Training script for nail segmentation U-Net.
Output: nail-unet.onnx (256x256, float32 input [1,3,256,256], float32 output [1,1,256,256])
"""
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ============================================================
# 1. U-Net architecture (lightweight, ~2M params)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

# ============================================================
# 2. Dataset
# ============================================================
class NailDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=True):
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB').resize((256, 256))
        mask_fname = fname.replace('.jpg', '.png')
        mask = Image.open(os.path.join(self.masks_dir, mask_fname)).convert('L').resize((256, 256), Image.NEAREST)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = img_arr.transpose(2, 0, 1)  # CHW

        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = np.where(mask_arr > 0.5, 1.0, 0.0)
        mask_arr = mask_arr[np.newaxis, ...]  # (1, H, W)

        return torch.from_numpy(img_arr), torch.from_numpy(mask_arr)

# ============================================================
# 3. Training
# ============================================================
def main():
    print("=== Nail Segmentation Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = NailDataset('dataset_aug/images', 'dataset_aug/masks', augment=False)
    print(f"Dataset size: {len(dataset)} images")

    # Split 80/20
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(in_ch=3, out_ch=1).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    EPOCHS = 40

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)

        # IoU metric
        with torch.no_grad():
            iou_scores = []
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = torch.sigmoid(model(imgs))
                preds = (outputs > 0.5).float()
                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum() - inter
                iou = (inter / (union + 1e-8)).item()
                iou_scores.append(iou)
            mean_iou = np.mean(iou_scores)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | IoU={mean_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'nail-unet-best.pth')
            print(f"  -> saved best model (val_loss={val_loss:.4f})")

    # ============================================================
    # 4. Export to ONNX
    # ============================================================
    print("\n=== Exporting to ONNX ===")
    model.load_state_dict(torch.load('nail-unet-best.pth', map_location='cpu'))
    model.eval()
    
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, 'nail-unet.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )
    print("Exported: nail-unet.onnx")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession('nail-unet.onnx')
    out = sess.run(None, {'input': dummy.numpy()})[0]
    print(f"ONNX output shape: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")
    print("=== Done! ===")

if __name__ == '__main__':
    main()

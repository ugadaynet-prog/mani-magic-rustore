"""
Augmentation v2: expands dataset_merged/ by 15x (rotations, flips, brightness, contrast, sharpness, color, perspective, noise)
Output: dataset_aug/images/ and dataset_aug/masks/ at 512x512 resolution.
"""
import os, shutil, random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SIZE = 384  # 384x384 — баланс между качеством и памятью

shutil.rmtree('dataset_aug', ignore_errors=True)
os.makedirs('dataset_aug/images', exist_ok=True)
os.makedirs('dataset_aug/masks', exist_ok=True)

def add_gaussian_noise(img, sigma=10):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def adjust_hue(img, shift=0.05):
    """Shift hue by converting to HSV, shifting H, converting back."""
    hsv = img.convert('HSV')
    arr = np.array(hsv)
    arr[:,:,0] = (arr[:,:,0].astype(np.int32) + int(shift * 255)) % 256
    return Image.fromarray(arr, 'HSV').convert('RGB')

augs = [
    ('orig', lambda i,m: (i,m)),
    ('rot90', lambda i,m: (i.rotate(90), m.rotate(90))),
    ('rot180', lambda i,m: (i.rotate(180), m.rotate(180))),
    ('rot270', lambda i,m: (i.rotate(270), m.rotate(270))),
    ('flip_h', lambda i,m: (i.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT))),
    ('flip_v', lambda i,m: (i.transpose(Image.FLIP_TOP_BOTTOM), m.transpose(Image.FLIP_TOP_BOTTOM))),
    ('bright_130', lambda i,m: (ImageEnhance.Brightness(i).enhance(1.3), m)),
    ('bright_70', lambda i,m: (ImageEnhance.Brightness(i).enhance(0.7), m)),
    ('contrast_140', lambda i,m: (ImageEnhance.Contrast(i).enhance(1.4), m)),
    ('sharp_200', lambda i,m: (ImageEnhance.Sharpness(i).enhance(2.0), m)),
    ('color_150', lambda i,m: (ImageEnhance.Color(i).enhance(1.5), m)),
    ('color_50', lambda i,m: (ImageEnhance.Color(i).enhance(0.5), m)),
    ('noise', lambda i,m: (add_gaussian_noise(i, 15), m)),
    ('hue_plus', lambda i,m: (adjust_hue(i, 0.05), m)),
    ('hue_minus', lambda i,m: (adjust_hue(i, -0.05), m)),
]

count = 0
img_dir = 'dataset_merged/images'
mask_dir = 'dataset_merged/masks'
for fname in sorted(os.listdir(img_dir)):
    mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
    img = Image.open(f'{img_dir}/{fname}').convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
    mask = Image.open(f'{mask_dir}/{mask_name}').convert('L').resize((SIZE, SIZE), Image.NEAREST)
    mask_arr = np.where(np.array(mask) > 127, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_arr, mode='L')
    for name, fn in augs:
        ai, am = fn(img, mask)
        ai.save(f'dataset_aug/images/{count:05d}.jpg', quality=95)
        am.save(f'dataset_aug/masks/{count:05d}.png')
        count += 1

print(f'After augmentation (15x): {count} images at {SIZE}x{SIZE}')

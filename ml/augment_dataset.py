"""
Augmentation: expands dataset_merged/ by 11x (rotations, flips, brightness, etc.)
Output: dataset_aug/images/ and dataset_aug/masks/ at 256x256 resolution.
"""
import os, shutil
from PIL import Image, ImageEnhance
import numpy as np
import random

random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

shutil.rmtree('dataset_aug', ignore_errors=True)
os.makedirs('dataset_aug/images', exist_ok=True)
os.makedirs('dataset_aug/masks', exist_ok=True)

augs = [
    ('orig', lambda i,m: (i,m)),
    ('rot90', lambda i,m: (i.rotate(90), m.rotate(90))),
    ('rot180', lambda i,m: (i.rotate(180), m.rotate(180))),
    ('rot270', lambda i,m: (i.rotate(270), m.rotate(270))),
    ('flip_h', lambda i,m: (i.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT))),
    ('flip_v', lambda i,m: (i.transpose(Image.FLIP_TOP_BOTTOM), m.transpose(Image.FLIP_TOP_BOTTOM))),
    ('bright', lambda i,m: (ImageEnhance.Brightness(i).enhance(1.3), m)),
    ('dark', lambda i,m: (ImageEnhance.Brightness(i).enhance(0.7), m)),
    ('contrast', lambda i,m: (ImageEnhance.Contrast(i).enhance(1.4), m)),
    ('sharp', lambda i,m: (ImageEnhance.Sharpness(i).enhance(2.0), m)),
    ('color', lambda i,m: (ImageEnhance.Color(i).enhance(1.5), m)),
]

count = 0
img_dir = 'dataset_merged/images'
mask_dir = 'dataset_merged/masks'
for fname in sorted(os.listdir(img_dir)):
    mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
    img = Image.open(f'{img_dir}/{fname}').convert('RGB').resize((256, 256), Image.BILINEAR)
    mask = Image.open(f'{mask_dir}/{mask_name}').convert('L').resize((256, 256), Image.NEAREST)
    mask_arr = np.where(np.array(mask) > 127, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_arr, mode='L')
    for name, fn in augs:
        ai, am = fn(img, mask)
        ai.save(f'dataset_aug/images/{count:05d}.jpg', quality=95)
        am.save(f'dataset_aug/masks/{count:05d}.png')
        count += 1

print(f'After augmentation: {count} images')

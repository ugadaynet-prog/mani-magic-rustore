#!/usr/bin/env python3
"""
Downloads and merges all available nail segmentation datasets with masks.
Produces a unified dataset at dataset_merged/images/ and dataset_merged/masks/
ready for training. Deduplicates by image content hash.

v2: Added Golbstein (193) and Ademakdogan (52) datasets.
"""
import os, sys, shutil, hashlib, zipfile, subprocess, urllib.request
from PIL import Image
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Clean slate
shutil.rmtree('dataset_merged', ignore_errors=True)
os.makedirs('dataset_merged/images', exist_ok=True)
os.makedirs('dataset_merged/masks', exist_ok=True)

count = 0
seen_hashes = set()
skipped = 0

def add_pair(img_path, mask_path):
    global count, skipped
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return
    try:
        mask = Image.open(mask_path)
        arr = np.array(mask)
        # Handle both grayscale and RGB masks
        if arr.ndim == 3:
            binary = (arr > 0).any(axis=2)
        else:
            binary = arr > 0
        if not binary.any():
            skipped += 1
            return
    except:
        return
    h = hashlib.md5(open(img_path, 'rb').read()).hexdigest()
    if h in seen_hashes:
        skipped += 1
        return
    seen_hashes.add(h)
    # Save image as JPG
    Image.open(img_path).convert('RGB').save(f'dataset_merged/images/{count:05d}.jpg', quality=95)
    # Save mask as binary PNG
    Image.fromarray(binary.astype(np.uint8) * 255, 'L').save(f'dataset_merged/masks/{count:05d}.png')
    count += 1

def download_git_zip(url, dest):
    if os.path.exists(dest):
        return True
    print(f'Downloading {url}...')
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f'  Failed: {e}')
        return False

print('=== Downloading nail segmentation datasets ===')

# 1. vpapenko (52 images, CC0) - the base
download_git_zip(
    'https://github.com/vpapenko/nails-segmentation-dataset/raw/master/nails_segmentation.zip',
    'vpapenko.zip'
)
if os.path.exists('vpapenko.zip'):
    with zipfile.ZipFile('vpapenko.zip') as z:
        z.extractall('vpapenko_d')
    for f in sorted(os.listdir('vpapenko_d/images')):
        add_pair(f'vpapenko_d/images/{f}', f'vpapenko_d/labels/{f}')
    print(f'After vpapenko: {count}')

# 2. Paulina Pacyna (53 images)
download_git_zip(
    'https://codeload.github.com/PaulinaPacyna/image-processing--nails-segmentation/zip/refs/heads/master',
    'paulina.zip'
)
if os.path.exists('paulina.zip'):
    with zipfile.ZipFile('paulina.zip') as z:
        z.extractall('paulina_d')
    base = 'paulina_d/image-processing--nails-segmentation-master/nails_segmentation'
    if os.path.exists(f'{base}/images'):
        for f in sorted(os.listdir(f'{base}/images')):
            add_pair(f'{base}/images/{f}', f'{base}/labels/{f}')
    print(f'After paulina: {count}')

# 3. Zea-Zee (large dataset with multiple splits - ~235 unique real masks)
print('Cloning Zea-Zee repo (484 MB, may take a moment)...')
if not os.path.exists('zea_d'):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Zea-Zee/nails-semantic-segmentation-pytorch.git', 'zea_d'],
                   capture_output=True, timeout=300)

if os.path.exists('zea_d'):
    zea_pairs = [
        ('zea_d/dataset-183/train/images', 'zea_d/dataset-183/train/labels'),
        ('zea_d/dataset-183/val/images', 'zea_d/dataset-183/val/labels'),
        ('zea_d/base-dataset/train/images', 'zea_d/base-dataset/train/labels'),
        ('zea_d/base-dataset/val/images', 'zea_d/base-dataset/val/labels'),
        ('zea_d/expanded_dataset/merged/images', 'zea_d/expanded_dataset/merged/labels'),
        ('zea_d/expanded_dataset/images', 'zea_d/expanded_dataset/labels'),
        ('zea_d/dataset/train/images', 'zea_d/dataset/train/labels'),
        ('zea_d/dataset/val/images', 'zea_d/dataset/val/labels'),
    ]
    for img_dir, mask_dir in zea_pairs:
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    add_pair(f'{img_dir}/{f}', f'{mask_dir}/{f}')
    print(f'After zea-zee: {count}')

# 4. behrooz Spatial_RNN_Fingernail (266 images)
download_git_zip(
    'https://codeload.github.com/behroozmrd47/Spatial_RNN_Fingernail/zip/refs/heads/main',
    'behrooz.zip'
)
if os.path.exists('behrooz.zip'):
    with zipfile.ZipFile('behrooz.zip') as z:
        z.extractall('behrooz_d')
    base = 'behrooz_d/Spatial_RNN_Fingernail-main/data/raw'
    for ds in ['dataset1', 'dataset2']:
        img_dir = f'{base}/{ds}/image'
        mask_dir = f'{base}/{ds}/mask'
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    add_pair(f'{img_dir}/{f}', f'{mask_dir}/{f}')
    print(f'After behrooz: {count}')

# 5. Golbstein Fingernails-Segmentation (193 images, RGB masks with nail indices)
print('Downloading Golbstein dataset (19MB)...')
download_git_zip(
    'https://github.com/Golbstein/Fingernails-Segmentation/raw/master/nails.tar.gz',
    'golbstein.tar.gz'
)
if os.path.exists('golbstein.tar.gz'):
    import tarfile
    with tarfile.open('golbstein.tar.gz') as t:
        t.extractall('golbstein_d')
    img_dir = 'golbstein_d/nails/raw'
    mask_dir = 'golbstein_d/nails/mask'
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        for f in sorted(os.listdir(mask_dir)):
            if not f.lower().endswith('.png'):
                continue
            raw_path = os.path.join(img_dir, f)
            if not os.path.exists(raw_path):
                continue
            add_pair(raw_path, os.path.join(mask_dir, f))
    print(f'After golbstein: {count}')

# 6. Ademakdogan nails_segmentation (52 images, DeepLabV3 dataset)
print('Cloning ademakdogan repo...')
if not os.path.exists('ademakdogan_d'):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/ademakdogan/nails_segmentation.git', 'ademakdogan_d'],
                   capture_output=True, timeout=120)
if os.path.exists('ademakdogan_d'):
    for split in ['train', 'val', 'test']:
        img_dir = f'ademakdogan_d/dataset/processed/{split}'
        mask_dir = f'ademakdogan_d/dataset/processed/{split}_labels'
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            for f in sorted(os.listdir(img_dir)):
                mask_path = os.path.join(mask_dir, f)
                if os.path.exists(mask_path):
                    add_pair(os.path.join(img_dir, f), mask_path)
    print(f'After ademakdogan: {count}')

print(f'\n=== FINAL MERGED DATASET: {count} unique images with real masks ===')
print(f'Skipped (empty masks or duplicates): {skipped}')

# Cleanup large download dirs
for d in ['vpapenko_d', 'paulina_d', 'zea_d', 'behrooz_d', 'golbstein_d', 'ademakdogan_d']:
    shutil.rmtree(d, ignore_errors=True)
for f in ['vpapenko.zip', 'paulina.zip', 'behrooz.zip', 'golbstein.tar.gz']:
    if os.path.exists(f):
        os.remove(f)

print('Done. Dataset ready at dataset_merged/')

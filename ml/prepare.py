"""
Prepares data for training.
This is the entry point called by the CI workflow.

It first downloads and merges all available datasets (prepare_dataset.py),
then augments them (augment_dataset.py), and finally packs them into
data/prepared.npz for train.py.

Falls back to data/nails.zip if the download scripts are not available.
"""
import io
import json
import os
import zipfile
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 384
THRESH = 128


def square(im, mk):
    """Pad to square without stretching, then resize to SIZE."""
    w, h = im.size
    side = max(w, h)
    pad_im = Image.new('RGB', (side, side), (0, 0, 0))
    pad_mk = Image.new('L', (side, side), 0)
    off = ((side - w) // 2, (side - h) // 2)
    pad_im.paste(im, off)
    pad_mk.paste(mk, off)
    return (pad_im.resize((SIZE, SIZE), Image.BILINEAR),
            pad_mk.resize((SIZE, SIZE), Image.NEAREST))


def load_from_augmented():
    """Load from dataset_aug/ (already 256x256, already augmented)."""
    img_dir = os.path.join(HERE, 'dataset_aug', 'images')
    mask_dir = os.path.join(HERE, 'dataset_aug', 'masks')
    if not os.path.exists(img_dir):
        return None

    xs, ys, names = [], [], []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue
        im = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        mk = Image.open(mask_path).convert('L')
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BILINEAR)
        if mk.size != (SIZE, SIZE):
            mk = mk.resize((SIZE, SIZE), Image.NEAREST)
        m = (np.asarray(mk) > THRESH).astype(np.uint8)
        xs.append(np.asarray(im, dtype=np.uint8))
        ys.append(m)
        names.append(os.path.splitext(fname)[0])
    return xs, ys, names


def load_from_zip(zip_path):
    """Load from original data/nails.zip (fallback)."""
    z = zipfile.ZipFile(zip_path)
    imgs = {os.path.splitext(os.path.basename(n))[0]: n
            for n in z.namelist() if n.startswith('images/') and not n.endswith('/')}
    labs = {os.path.splitext(os.path.basename(n))[0]: n
            for n in z.namelist() if n.startswith('labels/') and not n.endswith('/')}
    keys = sorted(set(imgs) & set(labs))
    xs, ys, names = [], [], []
    for k in keys:
        im = Image.open(io.BytesIO(z.read(imgs[k]))).convert('RGB')
        mk = Image.open(io.BytesIO(z.read(labs[k]))).convert('L')
        im, mk = square(im, mk)
        m = (np.asarray(mk) > THRESH).astype(np.uint8)
        xs.append(np.asarray(im, dtype=np.uint8))
        ys.append(m)
        names.append(k)
    return xs, ys, names


def main():
    # Step 1: Download and merge all datasets
    prepare_ds = os.path.join(HERE, 'prepare_dataset.py')
    if os.path.exists(prepare_ds):
        print('=== Downloading and merging datasets ===')
        subprocess.run(['python', prepare_ds], check=True, cwd=HERE)

    # Step 2: Augment
    augment_ds = os.path.join(HERE, 'augment_dataset.py')
    if os.path.exists(augment_ds):
        print('=== Augmenting dataset ===')
        subprocess.run(['python', augment_ds], check=True, cwd=HERE)

    # Step 3: Load and pack into prepared.npz
    result = load_from_augmented()
    if result is None:
        print('dataset_aug/ not found, falling back to data/nails.zip')
        zip_path = os.path.join(HERE, 'data', 'nails.zip')
        if not os.path.exists(zip_path):
            raise SystemExit('No dataset found.')
        result = load_from_zip(zip_path)

    xs, ys, names = result
    X = np.stack(xs)
    Y = np.stack(ys)
    out = os.path.join(HERE, 'data', 'prepared.npz')
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    np.savez_compressed(out, X=X, Y=Y, names=np.array(names))

    fg = float(Y.mean())
    empty = int((Y.sum(axis=(1, 2)) == 0).sum())
    print(json.dumps({
        'pairs': int(X.shape[0]),
        'size': list(X.shape[1:]),
        'nail_fraction': round(fg * 100, 2),
        'empty_masks': empty,
        'file': out,
        'mb': round(os.path.getsize(out) / 1e6, 1),
    }, ensure_ascii=False, indent=2))
    if empty:
        print('WARNING: %d masks are empty after binarization' % empty)


if __name__ == '__main__':
    main()

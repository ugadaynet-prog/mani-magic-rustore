"""Shared offline reference for the current Android + tryon.js mask pipeline.

Geometry, component rejection, PNG quantization and hole filling follow the
application. Rasterization uses Pillow BILINEAR instead of Android/Chromium
Skia; this is a reproducible reference, NOT verified device pixel parity.
The returned bool mask measures the rendered mask at the UI threshold (0.40),
not every partly transparent pixel in the UI's soft alpha band.
"""
import math

import cv2
import numpy as np
from PIL import Image

THRESHOLD = 0.40
FOUND_MIN = 0.5
MIN_BLOB = 60
RING_PX = 7
RING_SKIN_MIN = 0.22
HOLE_MAX_FRAC = 0.04
PIPELINE_VERSION = 'android-reference-v1-pillow-bilinear'
PIPELINE_NOTE = ('Android/JS geometry, filtering, quantization and holes; '
                 'Pillow bilinear raster approximation, device parity unverified')


def sigmoid(logits):
    """Stable float32 sigmoid; accepts a 2-D array of model logits."""
    x = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def prepare_input(im, size):
    """Return PIL RGB square; placement uses Kotlin Float32 + integer truncation.

    PIL performs the image sampling; unlike the old pad-then-resize evaluator,
    only the image rectangle is resized and black padding is added afterwards.
    """
    im = im.convert('RGB')
    size = int(size)
    if size <= 0 or im.width <= 0 or im.height <= 0:
        raise ValueError('Image dimensions and model size must be positive')
    scale = np.float32(size) / np.float32(max(im.size))
    dw = int(np.float32(im.width) * scale)
    dh = int(np.float32(im.height) * scale)
    out = Image.new('RGB', (size, size), (0, 0, 0))
    if dw and dh:
        out.paste(im.resize((dw, dh), Image.Resampling.BILINEAR),
                  ((size - dw) // 2, (size - dh) // 2))
    return out


def _rgb_float(input_rgb, shape):
    rgb = np.asarray(input_rgb)
    if rgb.shape != tuple(shape) + (3,):
        raise ValueError('input_rgb must be HWC RGB at model resolution')
    if np.issubdtype(rgb.dtype, np.integer):
        return rgb.astype(np.float32)
    rgb = rgb.astype(np.float32)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('Floating input_rgb must be normalized to [0, 1]')
    return rgb * np.float32(255.0)


def suppress_stray(prob, input_rgb):
    """Mirror the native filter, preserving rejected components' low-prob halos.

    The native filter labels >0.5 components, uses a square seven-pixel ring
    excluding ALL foreground, rejects only when ring area >=20, and has no
    minimum component size. Probabilities outside these components survive.
    """
    p = np.asarray(prob, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError('prob must be a square 2-D probability map')
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('prob must contain finite probabilities in [0, 1]')
    rgb = _rgb_float(input_rgb, p.shape)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cb = np.float32(128) - np.float32(.168736)*r - np.float32(.331264)*g + np.float32(.5)*b
    cr = np.float32(128) + np.float32(.5)*r - np.float32(.418688)*g - np.float32(.081312)*b
    skin = (cb >= 77) & (cb <= 130) & (cr >= 133) & (cr <= 177)
    fg = p > np.float32(.5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    result = p.copy()
    kernel = np.ones((2*RING_PX+1, 2*RING_PX+1), np.uint8)
    h, w = p.shape
    for cid in range(1, n):
        x, y, cw, ch, _ = stats[cid]
        x0, y0 = max(0, x-RING_PX), max(0, y-RING_PX)
        x1, y1 = min(w, x+cw+RING_PX), min(h, y+ch+RING_PX)
        region = labels[y0:y1, x0:x1] == cid
        ring = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
        ring &= ~fg[y0:y1, x0:x1]
        total = int(ring.sum())
        if total >= 20:
            fraction = np.float32(skin[y0:y1, x0:x1][ring].sum()) / np.float32(total)
            if fraction < np.float32(RING_SKIN_MIN):
                view = result[y0:y1, x0:x1]
                view[region] = sigmoid(np.array(-30, dtype=np.float32))
    return result


def fill_holes(binary):
    """tryon.js fillHoles: 4-connected holes, max(40, foreground_area*0.04)."""
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError('binary must be 2-D')
    out = binary.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~binary).astype(np.uint8), connectivity=4)
    exterior = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
    limit = max(40, int(binary.sum()) * HOLE_MAX_FRAC)
    for cid in range(1, n):
        if cid not in exterior and stats[cid, cv2.CC_STAT_AREA] <= limit:
            out[labels == cid] = True
    return out


def postprocess(prob, input_rgb, original_size):
    """Return full-resolution bool mask for (width, height).

    input_rgb is the exact prepared model input: PIL RGB, uint8 HWC, or
    normalized float HWC. JS deliberately uses fractional crop geometry,
    whereas native preprocessing uses integer placement; both are preserved.
    """
    filtered = suppress_stray(prob, input_rgb)
    # Kotlin truncates toInt; JS decodes the PNG into a Float32Array.
    png8 = np.clip(filtered * np.float32(255), 0, 255).astype(np.uint8)
    decoded = (png8.astype(np.float64) / 255.0).astype(np.float32)
    # JS compares float32 values, promoted to double, against the double .40.
    binary = fill_holes(decoded.astype(np.float64) > THRESHOLD)
    # This matches Math.max(probabilities[i], bin[i] ? 1 : 0), including the
    # app's saturation of all foreground pixels, not merely filled holes.
    field = np.where(binary, np.uint8(255), png8).astype(np.uint8)
    w, h = map(int, original_size)
    if w <= 0 or h <= 0:
        raise ValueError('original_size must have positive dimensions')
    side = field.shape[0]
    scale = side / max(w, h)
    dw, dh = w * scale, h * scale
    ox, oy = (side-dw)/2, (side-dh)/2
    resized = Image.fromarray(field).transform(
        (w, h), Image.Transform.AFFINE, (dw/w, 0, ox, 0, dh/h, oy),
        resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0 > THRESHOLD


def _boundary_score(gt, pred, tolerance):
    kernel = np.ones((3, 3), np.uint8)
    gb = gt & ~cv2.erode(gt.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                        borderValue=0).astype(bool)
    pb = pred & ~cv2.erode(pred.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)
    ng, npred = int(gb.sum()), int(pb.sum())
    if not ng or not npred:
        return 1.0 if ng == npred else 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    near_gt = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
    near_pred = cv2.dilate(pb.astype(np.uint8), k).astype(bool)
    precision = float((pb & near_gt).sum()) / npred
    recall = float((gb & near_pred).sum()) / ng
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def score_frame(gt_idx, pred, boundary_tolerance_px=None):
    """Historical nail metrics plus all outside-mask paint and boundary F1.

    Boundary tolerance defaults to 0.2% of the image diagonal, at least 1px.
    fp_rate is FP / GT area (None for an empty GT), not false-positive rate
    over negative pixels. fp_frame_rate also supports entirely negative frames.
    Historical stray_px counts only disconnected false components >=60px.
    """
    gt_idx, pred = np.asarray(gt_idx), np.asarray(pred, dtype=bool)
    if gt_idx.ndim != 2 or gt_idx.shape != pred.shape:
        raise ValueError('GT and prediction must have identical 2-D shapes')
    gt = gt_idx > 0
    ids = np.unique(gt_idx[gt])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    found, ious, per_nail = 0, [], {}
    for iid in ids:
        nail = gt_idx == iid
        intersection = int((nail & pred).sum())
        nail_area = int(nail.sum())
        if intersection / nail_area >= FOUND_MIN:
            found += 1
            hit = np.unique(labels[nail])
            hit = hit[hit != 0]
            union = nail_area + int(areas[hit].sum()) - intersection
            iou = intersection / union if union else 0.0
            ious.append(iou)
            per_nail[str(int(iid))] = round(iou, 3)
        else:
            per_nail[str(int(iid))] = None
    hit_gt = set(np.unique(labels[gt]).tolist())
    stray_ids = [i for i in range(1, n) if i not in hit_gt and areas[i] >= MIN_BLOB]
    tp, fp, fn = int((gt & pred).sum()), int((~gt & pred).sum()), int((gt & ~pred).sum())
    gt_px, frame_px = int(gt.sum()), int(gt.size)
    tolerance = (max(1, round(.002 * math.hypot(*gt.shape)))
                 if boundary_tolerance_px is None else int(boundary_tolerance_px))
    if tolerance < 0:
        raise ValueError('Boundary tolerance cannot be negative')
    return {'nails': len(ids), 'found': found,
            'iou': round(float(np.mean(ious)), 3) if ious else None,
            'per_nail': per_nail, 'stray': len(stray_ids),
            'stray_px': int(areas[stray_ids].sum()) if stray_ids else 0,
            'tp': tp, 'fp': fp, 'fn': fn, 'gt_px': gt_px, 'frame_px': frame_px,
            'precision': tp/(tp+fp) if tp+fp else (1.0 if not gt_px else 0.0),
            'recall_pixels': tp/gt_px if gt_px else 1.0,
            'fp_rate': fp/gt_px if gt_px else None,
            'fp_frame_rate': fp/frame_px,
            'boundary_fscore': _boundary_score(gt, pred, tolerance),
            'boundary_tolerance_px': tolerance}


def aggregate(rows):
    """Aggregate fixed-image rows; IoU and boundary F1 are image macro means."""
    rows = list(rows)
    totals = {key: sum(int(r[key]) for r in rows)
              for key in ('found', 'nails', 'stray', 'stray_px', 'tp', 'fp', 'fn', 'gt_px', 'frame_px')}
    ious = [r['iou'] for r in rows if r['iou'] is not None]
    boundaries = [r['boundary_fscore'] for r in rows]
    tp, fp, gt_px = totals['tp'], totals['fp'], totals['gt_px']
    return dict(totals, frames=len(rows),
                recall=totals['found']/totals['nails'] if totals['nails'] else None,
                iou=float(np.mean(ious)) if ious else None,
                precision=tp/(tp+fp) if tp+fp else (1.0 if not gt_px else 0.0),
                recall_pixels=tp/gt_px if gt_px else 1.0,
                fp_rate=fp/gt_px if gt_px else None,
                fp_frame_rate=fp/totals['frame_px'] if totals['frame_px'] else None,
                boundary_fscore=float(np.mean(boundaries)) if boundaries else None)
